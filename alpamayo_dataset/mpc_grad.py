#!/usr/bin/env python3
"""
mpc_grad.py — Differentiable MPC Layer (IFT-based analytical gradient)

Diff-WMPC end-to-end 학습을 위한 PyTorch custom autograd 함수.
condensed LTV-MPC의 box-constrained QP에 IFT 적용:

  min_U  ½ U^T H(θ) U + g(θ)^T U    s.t.  lb ≤ U ≤ ub

θ = [long, yaw, steer_r, accel_r]  (lat = W_LAT_FIXED = 1.0 고정)
H, g 는 θ 에 대해 선형: H(θ) = Σ_i θ_i H_basis_i,  g(θ) = Σ_i θ_i g_basis_i

IFT sensitivity (paper: Jahncke et al., IEEE RA-L 2026, Eq. 2):

  비활성 케이스 (U* ∈ interior):
    dU*/dθ = -H⁻¹ V        V_i = dH/dθ_i · U* + dg/dθ_i

  활성 케이스 (U*_j = lb_j 또는 ub_j):
    KKT matrix M = [[H, C_a^T], [C_a, 0]]  (C_a: box 활성 제약 행렬)
    adjoint: M η_ext = [grad_U; 0]  →  dL/dθ = -(η_ext[:NNU] @ V)

    C_a row: +eⱼ (상한 활성), -eⱼ (하한 활성)

재사용:
  mpc.py:  _linearize, _build_prediction_matrices_tv, _solve_qp, _build_delta_matrix
"""

import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.optimize
import torch

try:
    import scipy.sparse as _sp
    import osqp as _osqp
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from mpc import (
    N_LABEL, N_EVAL, NX, NU, QP_REG, U_MIN, U_MAX, TERMINAL_Q_FACTOR,
    W_LAT_FIXED, IX, IY, IYAW, IVX, IACCEL, WB, DT, V_MIN_LIN,
    _linearize, _build_prediction_matrices_tv, _build_delta_matrix,
)

_NNU = N_LABEL * NU   # = 40  (20 스텝 × 2 입력)

_ACTIVE_TOL = 1e-4   # active set 판별 tolerance

# ── 성능 프로파일러 (모듈 레벨, train_epoch에서 reset/report) ──────────────
class _PerfAccum:
    def __init__(self):
        self._t: dict[str, float] = defaultdict(float)
        self._n: dict[str, int]   = defaultdict(int)
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._t.clear(); self._n.clear()

    def add(self, key: str, dt: float):
        with self._lock:
            self._t[key] += dt
            self._n[key] += 1

    def report(self) -> dict:
        with self._lock:
            return {k: {"total_s": self._t[k], "n": self._n[k],
                        "avg_ms": 1000 * self._t[k] / max(self._n[k], 1)}
                    for k in self._t}

PERF = _PerfAccum()   # diff_wmpc_train.py 에서 import 해서 사용


def set_executor(max_workers: int) -> None:
    """하위 호환용 stub — ThreadPoolExecutor 병렬화는 GIL 제약으로 효과 없음."""
    pass


# ══════════════════════════════════════════════════════
# 고정밀 QP 솔버 (IFT 전용)
# ══════════════════════════════════════════════════════

def _solve_qp_exact(
    H: np.ndarray, g: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2단계 고정밀 box-constrained QP 풀이.

    Phase 1 (active set 탐지):
        L-BFGS-B (ftol=1e-12, gtol=1e-10, maxiter=3000) 로 대략적 해 획득.
        활성 집합 A = {j : U_j ≈ lb_j 또는 U_j ≈ ub_j} 식별.

    Phase 2 (자유 서브공간 정밀 풀이):
        U_active = lb/ub (정확히 고정),
        H_ff dU_f = -(H_fa U_a + g_f)  → np.linalg.solve (직접, 오차 없음)
        KKT 잔차 ||H_free U + g_free|| < 1e-8 보장.

    Returns:
        U_opt    (NNU,)   최적 제어 입력
        act_lb   (NNU,) bool — U*_j = lb_j (하한 활성)
        act_ub   (NNU,) bool — U*_j = ub_j (상한 활성)
    """
    NNU = len(g)

    # ── Phase 1: 비제약 풀이 시도 ──────────────────────────────
    U_unc = np.linalg.solve(H, -g)
    if np.all(U_unc >= lb - 1e-8) and np.all(U_unc <= ub + 1e-8):
        U = np.clip(U_unc, lb, ub)
        return U, U <= lb + _ACTIVE_TOL, U >= ub - _ACTIVE_TOL

    # ── Phase 1: OSQP (GIL-free C++ solver) or L-BFGS-B fallback ──
    if _OSQP_AVAILABLE:
        try:
            P   = _sp.triu(H, format="csc")
            eye = _sp.eye(NNU, format="csc")
            sol = _osqp.OSQP()
            sol.setup(P, g, eye, lb, ub,
                      warm_starting=True, verbose=False,
                      eps_abs=1e-6, eps_rel=1e-6, polish=True)
            result   = sol.solve()
            U_approx = result.x if result.info.status_val == 1 else np.clip(U_unc, lb, ub)
        except Exception:
            U_approx = np.clip(U_unc, lb, ub)
    else:
        res = scipy.optimize.minimize(
            lambda u: (0.5 * u @ H @ u + g @ u, H @ u + g),
            np.clip(U_unc, lb, ub),
            method="L-BFGS-B", jac=True,
            bounds=scipy.optimize.Bounds(lb, ub),
            options={"ftol": 1e-8, "gtol": 1e-6, "maxiter": 500},
        )
        U_approx = res.x

    act_lb = U_approx <= lb + _ACTIVE_TOL
    act_ub = U_approx >= ub - _ACTIVE_TOL
    active = act_lb | act_ub
    free   = ~active

    if free.sum() == 0:
        return np.clip(U_approx, lb, ub), act_lb, act_ub

    # ── Phase 2: 자유 서브공간 직접 풀이 ────────────────────────
    U = np.empty(NNU)
    U[act_lb] = lb[act_lb]
    U[act_ub] = ub[act_ub]

    g_eff_free = (H[:, active] @ U[active] + g)[free]
    H_ff       = H[np.ix_(free, free)]
    try:
        U[free] = -np.linalg.solve(H_ff, g_eff_free)
        U = np.clip(U, lb, ub)
    except np.linalg.LinAlgError:
        U = U_approx

    return U, U <= lb + _ACTIVE_TOL, U >= ub - _ACTIVE_TOL


# ══════════════════════════════════════════════════════
# Scene 사전 계산 (DataLoader worker에서 호출)
# ══════════════════════════════════════════════════════

def precompute_scene(
    x0:           np.ndarray,
    gt_xy:        np.ndarray,
    gt_yaw:       np.ndarray,
    gt_speed:     np.ndarray,
    gt_lon_accel: np.ndarray,
) -> dict:
    """
    Per-sample MPC 행렬 사전 계산 (label_mpc_weights.MPCLabeler.__init__ 과 동일 로직).
    DataLoader worker 에서 h5 로드 시 함께 호출.

    Args:
        x0:           (NX=8,)       MPC 초기 상태 — compute_x0() 결과
        gt_xy:        (≥N_LABEL, 2) GT smooth xy [m] (로컬 프레임)
        gt_yaw:       (≥N_LABEL,)   GT yaw [rad]
        gt_speed:     (≥N_LABEL,)   GT 종방향 속도 [m/s]
        gt_lon_accel: (≥N_LABEL,)   GT 종방향 가속도 [m/s²]

    Returns dict:
        H_Q_basis   (3, NNU, NNU)  Q 가중치별 H 기저 [long=0, lat=1, yaw=2]
        g_Q_basis   (3, NNU)       Q 가중치별 g 기저
        R_basis     (2, NNU, NNU)  R 가중치별 기저 [steer_r=0, accel_r=1]
        H_lat       (NNU, NNU)     lat 고정 기여 (W_LAT_FIXED=1.0)
        g_lat       (NNU,)
        x_free      ((N_LABEL+1)*NX,)   U=0 자유 응답
        S_u         ((N_LABEL+1)*NX, NNU)
        gt_xy_eval  (N_EVAL, 2)    ADE 평가용 GT (N_EVAL=15 스텝)
        lb          (NNU,)         box constraint 하한
        ub          (NNU,)         box constraint 상한
    """
    # ── LTV 선형화용 명목 상태 복원 ─────────────────────────────────────
    # label_mpc_weights._build_gt_full_states 와 동일 로직 (import 순환 방지)
    xy_ext    = np.vstack([[0., 0.], gt_xy[:N_LABEL].astype(np.float64)])
    yaw_uw    = np.unwrap(
        np.concatenate([[0.], gt_yaw[:N_LABEL].astype(np.float64)])
    )
    speed_ext = np.concatenate([[float(x0[IVX])],    gt_speed[:N_LABEL].astype(np.float64)])
    accel_ext = np.concatenate([[float(x0[IACCEL])], gt_lon_accel[:N_LABEL].astype(np.float64)])
    yr_ext    = np.gradient(yaw_uw, DT)
    vl_x      = np.gradient(xy_ext[:, 0], DT)
    vl_y      = np.gradient(xy_ext[:, 1], DT)

    nom_states: list[np.ndarray] = []
    for k in range(N_LABEL):
        if k == 0:
            nom_states.append(x0.copy())
            continue
        cy, sy = np.cos(yaw_uw[k]), np.sin(yaw_uw[k])
        vx_k   = max(float(speed_ext[k]), V_MIN_LIN)
        vy_k   = float(-sy * vl_x[k] + cy * vl_y[k])
        yr_k   = float(yr_ext[k])
        nom_states.append(np.array([
            xy_ext[k, 0], xy_ext[k, 1], yaw_uw[k],
            vx_k, vy_k, yr_k,
            float(np.arctan2(yr_k * WB, vx_k)),
            float(accel_ext[k]),
        ]))

    # ── LTV condensed 예측 행렬 ──────────────────────────────────────────
    A_list, B_list = zip(*[_linearize(s) for s in nom_states])
    S_x, S_u = _build_prediction_matrices_tv(list(A_list), list(B_list))
    x_free   = S_x @ x0                                          # ((N_LABEL+1)*NX,)

    # ── 참조 궤적 ────────────────────────────────────────────────────────
    x_ref = np.zeros((N_LABEL + 1, NX))
    for k in range(1, N_LABEL + 1):
        x_ref[k, IX]   = gt_xy[k - 1, 0]
        x_ref[k, IY]   = gt_xy[k - 1, 1]
        x_ref[k, IYAW] = gt_yaw[k - 1]

    NNU        = N_LABEL * NU
    dx         = x_free - x_ref.flatten()                        # ((N_LABEL+1)*NX,)
    S_u_blocks = S_u.reshape(N_LABEL + 1, NX, NNU)
    dx_blocks  = dx.reshape(N_LABEL + 1, NX)

    # 터미널 스텝 가중치 (k=N_LABEL 에 TERMINAL_Q_FACTOR 적용)
    q_w          = np.ones(N_LABEL + 1)
    q_w[N_LABEL] = TERMINAL_Q_FACTOR

    # ── Q 기저 행렬 (long=IX, lat=IY, yaw=IYAW) ──────────────────────────
    H_Q_basis_list, g_Q_basis_list = [], []
    for idx in [IX, IY, IYAW]:
        rows   = S_u_blocks[:, idx, :]           # (N_LABEL+1, NNU)
        w_rows = rows * q_w[:, None]
        H_Q_basis_list.append(rows.T @ w_rows)                     # (NNU, NNU)
        g_Q_basis_list.append(rows.T @ (q_w * dx_blocks[:, idx]))  # (NNU,)

    H_Q_basis = np.stack(H_Q_basis_list)          # (3, NNU, NNU)
    g_Q_basis = np.stack(g_Q_basis_list)          # (3, NNU)

    # lat 고정 기여
    H_lat = W_LAT_FIXED * H_Q_basis[1]            # (NNU, NNU)
    g_lat = W_LAT_FIXED * g_Q_basis[1]            # (NNU,)

    # ── R 기저 행렬 (Δu 페널티, D.T @ R_b @ D) ───────────────────────────
    D = _build_delta_matrix(N_LABEL)
    R_basis_list = []
    for j in range(NU):
        R_b = np.zeros((NNU, NNU))
        for k in range(N_LABEL):
            R_b[k * NU + j, k * NU + j] = 1.0
        R_basis_list.append(D.T @ R_b @ D)        # (NNU, NNU)

    return {
        "H_Q_basis":  H_Q_basis,                          # (3, NNU, NNU)
        "g_Q_basis":  g_Q_basis,                          # (3, NNU)
        "R_basis":    np.stack(R_basis_list),             # (2, NNU, NNU)
        "H_lat":      H_lat,                              # (NNU, NNU)
        "g_lat":      g_lat,                              # (NNU,)
        "x_free":     x_free,                             # ((N_LABEL+1)*NX,)
        "S_u":        S_u,                                # ((N_LABEL+1)*NX, NNU)
        "gt_xy_eval": gt_xy[:N_EVAL, :2].copy(),          # (N_EVAL, 2)
        "lb":         np.tile(U_MIN, N_LABEL),            # (NNU,)
        "ub":         np.tile(U_MAX, N_LABEL),            # (NNU,)
        # extract_xy_pred 전용: 매 프레임 Python 루프 제거용 사전 추출
        "S_u_xy":     np.stack([                          # (N_EVAL*2, NNU)
            S_u[(k + 1) * NX + idx, :]
            for k in range(N_EVAL) for idx in (IX, IY)
        ]).astype(np.float32),
        "x_free_xy":  np.array([                          # (N_EVAL*2,)
            x_free[(k + 1) * NX + idx]
            for k in range(N_EVAL) for idx in (IX, IY)
        ], dtype=np.float32),
    }


# ══════════════════════════════════════════════════════
# H, g 구성 (θ linear scale)
# ══════════════════════════════════════════════════════

def _build_hg(theta: np.ndarray, scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """θ (4,) linear scale → H (NNU×NNU), g (NNU,).

    θ = [long, yaw, steer_r, accel_r]  (lat = W_LAT_FIXED 고정)
    H(θ) = H_lat + Σ_i θ_i H_basis_i  (θ 에 대해 선형)
    """
    H = (scene["H_lat"]
         + theta[0] * scene["H_Q_basis"][0]   # long
         + theta[1] * scene["H_Q_basis"][2]   # yaw
         + theta[2] * scene["R_basis"][0]      # steer_r
         + theta[3] * scene["R_basis"][1])     # accel_r
    H = 0.5 * (H + H.T) + QP_REG * np.eye(H.shape[0])

    g = (scene["g_lat"]
         + theta[0] * scene["g_Q_basis"][0]
         + theta[1] * scene["g_Q_basis"][2])

    return H, g


# ══════════════════════════════════════════════════════
# IFT 해석적 기울기
# ══════════════════════════════════════════════════════

def _ift_grad(
    grad_U:    np.ndarray,   # (NNU,)   dL/dU*
    H:         np.ndarray,   # (NNU, NNU)
    U_opt:     np.ndarray,   # (NNU,)
    scene:     dict,
    active_lb: np.ndarray,   # (NNU,) bool — U*_j = lb_j
    active_ub: np.ndarray,   # (NNU,) bool — U*_j = ub_j
) -> np.ndarray:             # (4,)   dL/dθ
    """
    IFT adjoint 법으로 dL/dθ 계산.

    V_i = dH/dθ_i · U* + dg/dθ_i  (θ linear scale → chain rule w[i] 곱 불필요)

    비활성: η = H⁻¹ grad_U          →  dL/dθ = -(η @ V)
    활성:   M η_ext = [grad_U; 0]   →  dL/dθ = -(η_ext[:NNU] @ V)
              M = [[H, C_a^T], [C_a, 0]]

    C_a 구성:
      상한 활성 (U*_j = ub_j): row = +eⱼ
      하한 활성 (U*_j = lb_j): row = -eⱼ
    """
    NNU = len(U_opt)

    # V (NNU, 4): dH/dθ_i · U* + dg/dθ_i
    V = np.column_stack([
        scene["H_Q_basis"][0] @ U_opt + scene["g_Q_basis"][0],  # long
        scene["H_Q_basis"][2] @ U_opt + scene["g_Q_basis"][2],  # yaw
        scene["R_basis"][0]   @ U_opt,                           # steer_r (g 기여 없음)
        scene["R_basis"][1]   @ U_opt,                           # accel_r
    ])

    active   = active_lb | active_ub
    n_active = int(active.sum())

    try:
        if n_active == 0:
            # ── 비활성 케이스: H adjoint solve ──────────────────────────
            η = np.linalg.solve(H, grad_U)    # (NNU,)

        else:
            # ── 활성 케이스: KKT matrix 확장 ────────────────────────────
            active_idx = np.where(active)[0]
            C_a = np.zeros((n_active, NNU))
            for i, j in enumerate(active_idx):
                C_a[i, j] = 1.0 if active_ub[j] else -1.0

            M   = np.block([[H,   C_a.T                         ],
                             [C_a, np.zeros((n_active, n_active))]])
            rhs = np.concatenate([grad_U, np.zeros(n_active)])

            sol = np.linalg.solve(M, rhs)     # (NNU + n_active,)
            η   = sol[:NNU]                   # adjoint 변수

        return -(η @ V)    # (4,)

    except np.linalg.LinAlgError:
        return np.zeros(4)


# ══════════════════════════════════════════════════════
# PyTorch Custom Autograd Function
# ══════════════════════════════════════════════════════

class MpcDiffLayer(torch.autograd.Function):
    """
    Differentiable MPC forward/backward (θ → U*).

    forward:
        theta (4,) torch.Tensor [linear scale: long, yaw, steer_r, accel_r]
        scene dict (numpy arrays) — precompute_scene() 결과
      → U_opt (NNU,) torch.Tensor

    backward:
        grad_output (NNU,) → IFT → grad_theta (4,)

    Usage:
        scene = precompute_scene(x0, gt_xy, gt_yaw, gt_speed, gt_lon_accel)
        U_opt = MpcDiffLayer.apply(theta, scene)

    학습 루프에서의 연산 그래프:
        log_theta (MLP) → exp → theta → MpcDiffLayer → U_opt
          → X_pred (= x_free + S_u @ U_opt)  → xy_pred → ADE loss
    """

    @staticmethod
    def forward(ctx, theta: torch.Tensor, scene: dict) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy().astype(np.float64)

        t0 = time.perf_counter()
        H, g = _build_hg(theta_np, scene)
        PERF.add("build_hg", time.perf_counter() - t0)

        t0 = time.perf_counter()
        U_opt, active_lb, active_ub = _solve_qp_exact(H, g, scene["lb"], scene["ub"])
        PERF.add("solve_qp", time.perf_counter() - t0)

        ctx.scene     = scene
        ctx.U_opt_np  = U_opt
        ctx.H_np      = H
        ctx.active_lb = active_lb
        ctx.active_ub = active_ub

        return torch.tensor(U_opt, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_U = grad_output.detach().cpu().numpy().astype(np.float64)

        t0 = time.perf_counter()
        grad_theta_np = _ift_grad(
            grad_U    = grad_U,
            H         = ctx.H_np,
            U_opt     = ctx.U_opt_np,
            scene     = ctx.scene,
            active_lb = ctx.active_lb,
            active_ub = ctx.active_ub,
        )
        PERF.add("ift_grad", time.perf_counter() - t0)

        return (
            torch.tensor(grad_theta_np, dtype=grad_output.dtype, device=grad_output.device),
            None,   # scene 은 텐서가 아님
        )


# ══════════════════════════════════════════════════════
# 유틸리티: 연산 그래프 상 xy 추출
# ══════════════════════════════════════════════════════

def extract_xy_pred(
    U_opt: torch.Tensor,
    scene: dict,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    U_opt (NNU,) → xy_pred (N_EVAL, 2) — autograd 연결 유지.

    X_pred = x_free + S_u @ U_opt  (선형, autograd 통과)
    xy_pred = X_pred[NX:].reshape(N_LABEL, NX)[:N_EVAL, [IX, IY]]
    """
    if device is None:
        device = U_opt.device

    # S_u 의 xy 관련 행만 추출: (N_EVAL * 2, NNU)
    # X_pred 인덱스: k 스텝 → row = (k+1)*NX ... (k+2)*NX, col [IX, IY]
    S_u_xy = []
    # S_u_xy / x_free_xy 는 precompute_scene 에서 사전 추출됨 (Python 루프 없음)
    S_u_xy_t    = torch.tensor(scene["S_u_xy"],    dtype=dtype, device=device)
    x_free_xy_t = torch.tensor(scene["x_free_xy"], dtype=dtype, device=device)

    xy_flat = x_free_xy_t + S_u_xy_t @ U_opt                               # (N_EVAL*2,)
    return xy_flat.reshape(N_EVAL, 2)                                       # (N_EVAL, 2)


# ══════════════════════════════════════════════════════
# 배치 Differentiable MPC Layer (병렬 QP)
# ══════════════════════════════════════════════════════

class BatchMpcDiffLayer(torch.autograd.Function):
    """
    theta (4,) + scenes list[dict] → U_opts (n_frames, NNU)

    세그먼트 내 K개 프레임 QP를 순서대로 풀고 gradient는 합산.
    (threading은 GIL + 짧은 QP 시간으로 효과 없음 → 제거)

    Usage:
        U_opts = BatchMpcDiffLayer.apply(theta, scenes)
        # U_opts: (n_frames, NNU),  autograd 연결 유지
    """

    @staticmethod
    def forward(ctx, theta: torch.Tensor, scenes: list) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy().astype(np.float64)

        U_opts_list = []
        active_lbs  = []
        active_ubs  = []
        H_list      = []

        for scene in scenes:
            t0 = time.perf_counter()
            H, g = _build_hg(theta_np, scene)
            PERF.add("build_hg", time.perf_counter() - t0)

            t0 = time.perf_counter()
            U_opt, act_lb, act_ub = _solve_qp_exact(H, g, scene["lb"], scene["ub"])
            PERF.add("solve_qp", time.perf_counter() - t0)

            U_opts_list.append(U_opt)
            active_lbs.append(act_lb)
            active_ubs.append(act_ub)
            H_list.append(H)

        U_opts_np = np.stack(U_opts_list)   # (n, NNU)

        ctx.scenes     = scenes
        ctx.U_opts_np  = U_opts_np
        ctx.H_list     = H_list
        ctx.active_lbs = active_lbs
        ctx.active_ubs = active_ubs

        return torch.tensor(U_opts_np, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output: (n_frames, NNU)
        grad_U_all = grad_output.detach().cpu().numpy().astype(np.float64)
        n = len(ctx.scenes)

        grads = []
        for i in range(n):
            t0 = time.perf_counter()
            g = _ift_grad(
                grad_U    = grad_U_all[i],
                H         = ctx.H_list[i],
                U_opt     = ctx.U_opts_np[i],
                scene     = ctx.scenes[i],
                active_lb = ctx.active_lbs[i],
                active_ub = ctx.active_ubs[i],
            )
            PERF.add("ift_grad", time.perf_counter() - t0)
            grads.append(g)

        grad_theta_np = np.sum(grads, axis=0)   # (4,)  — frames 내 theta 공유이므로 합산
        return (
            torch.tensor(grad_theta_np, dtype=grad_output.dtype,
                         device=grad_output.device),
            None,   # scenes 는 텐서 아님
        )


# ══════════════════════════════════════════════════════
# 검증: 유한 차분 vs IFT 기울기 비교
# ══════════════════════════════════════════════════════

def check_gradient(scene: dict, theta: np.ndarray, eps: float = 1e-4) -> None:
    """유한 차분과 IFT 기울기를 비교해 구현 정합성 검증."""
    import torch as _t

    theta_t = _t.tensor(theta, dtype=_t.float64, requires_grad=True)
    U_opt_t = MpcDiffLayer.apply(theta_t, scene)

    # ADE 스칼라 loss (검증용)
    gt_xy = _t.tensor(scene["gt_xy_eval"], dtype=_t.float64)
    S_u_t = _t.tensor(scene["S_u"], dtype=_t.float64)
    x_free_t = _t.tensor(scene["x_free"], dtype=_t.float64)
    X_pred = x_free_t + S_u_t @ U_opt_t
    xy_pred = X_pred[NX:].reshape(N_LABEL, NX)[:N_EVAL, IX:IY+1]  # (N_EVAL, 2) 슬라이스
    loss = _t.mean(_t.norm(xy_pred - gt_xy, dim=1))
    loss.backward()
    ift_grad = theta_t.grad.numpy().copy()

    # 유한 차분
    def _loss_np(th):
        H, g = _build_hg(th, scene)
        U, _, _ = _solve_qp_exact(H, g, scene["lb"], scene["ub"])
        X = scene["x_free"] + scene["S_u"] @ U
        xy = X[NX:].reshape(N_LABEL, NX)[:N_EVAL, [IX, IY]]
        e = xy - scene["gt_xy_eval"]
        return float(np.mean(np.linalg.norm(e, axis=1)))

    fd_grad = np.zeros(4)
    base = _loss_np(theta)
    for i in range(4):
        th_h = theta.copy(); th_h[i] += eps
        fd_grad[i] = (_loss_np(th_h) - base) / eps

    print(f"{'':>10} {'IFT':>12} {'FD':>12} {'rel_err':>10}")
    names = ["long", "yaw", "steer_r", "accel_r"]
    for i, n in enumerate(names):
        rel = abs(ift_grad[i] - fd_grad[i]) / (abs(fd_grad[i]) + 1e-8)
        print(f"  {n:<10} {ift_grad[i]:>12.6f} {fd_grad[i]:>12.6f} {rel:>10.4f}")


if __name__ == "__main__":
    # 간이 단위 테스트 (실 h5 없이 random scene 으로 IFT vs FD 비교)
    print("=== mpc_grad 유한 차분 검증 ===")
    rng = np.random.default_rng(0)

    # 임의 GT 생성 (직진 + 약한 커브)
    t_arr   = np.arange(1, N_LABEL + 1) * DT
    gt_yaw  = np.linspace(0, 0.2, N_LABEL)
    gt_speed = np.full(N_LABEL, 8.0)
    gt_xy    = np.column_stack([gt_speed * t_arr * np.cos(gt_yaw),
                                gt_speed * t_arr * np.sin(gt_yaw)])
    gt_lon_accel = np.zeros(N_LABEL)
    x0 = np.array([0., 0., 0., 8., 0., 0., 0., 0.])

    scene = precompute_scene(x0, gt_xy, gt_yaw, gt_speed, gt_lon_accel)

    theta = np.array([2.0, 1.0, 5.0, 1.0])   # [long, yaw, steer_r, accel_r] linear
    check_gradient(scene, theta)
