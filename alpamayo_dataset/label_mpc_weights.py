#!/usr/bin/env python3
"""
MPC 가중치 자동 레이블링 (STEP 2)

각 .h5 샘플에 대해 GT 궤적을 가장 잘 재현하는 MPC 비용함수 가중치를
역최적화(inverse optimal control)로 계산하고 h5 파일의 labels/ 그룹에 저장합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
차량 모델: Dynamic Bicycle Model (단일, 분기 없음)
  - 타이어 코너링 강성 (CF, CR) 포함
  - 저속 특이점: vx_lin = max(vx, V_MIN_LIN) 클램핑
  - 이산화: matrix exponential (ZOH)
  - 차량 파라미터: alpasim VehicleModel.Parameters 기반

MPC 구조: Condensed Linear MPC
  - 상태: [x, y, yaw, vx, vy, yaw_rate, steering, accel]
  - 입력: [steering_cmd, accel_cmd]
  - 솔버 호라이즌: N_LABEL=N=20 (2s, 10Hz) — 배포와 동일
  - ADE 평가 구간: N_EVAL=15 (1.5s) — 터미널 롤오프 5스텝 제외
  - R 페널티: 입력 변화량(Δu) 기반 — 터미널 롤오프 구조적 억제
  - 터미널 Q: TERMINAL_Q_FACTOR=5.0 (k=N에서 추가 안전마진)

가중치 (MPCGains 기준, alpasim mpc_controller.py):
  lat_position_weight            = 1.0  [고정 — 정규화 기준]
  long_position_weight           → w[0]  (기본 2.0)
  heading_weight                 → w[1]  (기본 1.0)
  rel_front_steering_angle_weight → w[2]  (기본 5.0)
  rel_acceleration_weight        → w[3]  (기본 1.0)

저장 형식 (h5 labels/ 그룹):
  mpc_weights (5,)  float32  [long, lat, heading, steer_rate, accel_rate]
  ade               float32  최적화 후 ADE [m] (N=20 스텝 기준)
  valid             bool     ADE < ade_threshold
  n_evals           int32    함수 평가 횟수
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실행 예시:
  # 동작 확인 (파일 5개)
  python /workspace/alpamayo_dataset/label_mpc_weights.py \
      --data-dir /workspace/alpamayo_dataset/data/collected \
      --dry-run

  # 전체 실행
  python /workspace/alpamayo_dataset/label_mpc_weights.py \
      --data-dir /workspace/alpamayo_dataset/data/collected \
      --workers 8 --ade-threshold 0.5 --overwrite

의존성: numpy, scipy, h5py  (pip install scipy h5py)
"""

import argparse
import logging
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

import h5py
import numpy as np
import scipy.optimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)

sys.path.insert(0, str(Path(__file__).parent))
from mpc import (
    N, N_LABEL, N_EVAL, NX, NU,
    IX, IY, IYAW,
    V_MIN_LIN, QP_REG, U_MIN, U_MAX, W_LAT_FIXED, TERMINAL_Q_FACTOR,
    _linearize, _build_prediction_matrices, _solve_qp, _build_delta_matrix,
)

# ══════════════════════════════════════════════════════
# 가중치 설정
# ══════════════════════════════════════════════════════

# 최적화 변수 순서 (log-parameterized):
#   theta[0] → long_position_weight           (default 2.0)
#   theta[1] → heading_weight                 (default 1.0)
#   theta[2] → rel_front_steering_angle_weight (default 5.0)
#   theta[3] → rel_acceleration_weight        (default 1.0)
W_DEFAULTS = np.array([2.0, 1.0, 5.0, 1.0])
THETA0 = np.log(W_DEFAULTS)

# 탐색 공간 경계 (log-scale)
# R 하한을 충분히 높게 유지해 Q>>R 퇴화 방지
#                         long   head   steer_r  accel_r
_THETA_MIN = np.log([ 0.1,   0.1,   1.0,    0.2  ])
_THETA_MAX = np.log([ 10.,   10.,   25.,    10.  ])
_BOUNDS_LOG = list(zip(_THETA_MIN, _THETA_MAX))

# Tikhonov 정규화 강도: ADE 개선이 기본값 근방 이탈 비용을 넘지 않으면 기본값 근방 유지
# λ=0.002: degenerate 경계(||Δθ||²≈35)에서 ~0.07m 추가 비용 → ADE 대비 의미 있는 패널티
_REG_LAMBDA = 0.002

# ── 승차감 페널티 가중치 ─────────────────────────────────────
# ADE [m] 와 스케일을 맞추기 위한 하이퍼파라미터 — CLI로 덮어쓰기 가능
# steer_rate_rms 단위: [rad/step],  jerk_rms 단위: [(m/s²)/step]
W_STEER_PENALTY: float = 0.5
W_JERK_PENALTY:  float = 0.5


# ══════════════════════════════════════════════════════
# 핵심 클래스: 샘플별 MPC Labeler
# ══════════════════════════════════════════════════════

class MPCLabeler:
    """
    단일 샘플에 대한 MPC 가중치 역산기.

    x0에서 한 번 선형화 후 S_x, S_u를 캐시 →
    Nelder-Mead 반복마다 QP만 재풀이 (open-loop 근사).

    솔버 호라이즌: N_LABEL=N=20 (배포와 동일)
    ADE 평가 구간: N_EVAL=15 (1.5s, 터미널 롤오프 5스텝 제외)
    R 페널티: Δu 기반 (D.T @ R_blk @ D) — 터미널 수렴 구조적 억제

    Open-loop 근사의 의미:
      - GT 궤적은 물리적으로 달성 가능한 궤적
      - 단일 linearization → 동역학이 크게 변하지 않는 3초 이내는 충분히 정확
      - 가중치 역산 목적에서 closed-loop과의 오차는 허용 범위 내
    """

    def __init__(self, v0: float, gt_xyz: np.ndarray, gt_yaw: np.ndarray):
        """
        Args:
            v0:        t0 초기 속도 [m/s]   (ego_history_ego_states[-1, 2])
            gt_xyz:    (≥N_LABEL, 3)  GT 미래 위치 (로컬 프레임, t0+0.1s부터)
            gt_yaw:    (≥N_LABEL,)    GT 미래 yaw [rad]
        """
        # 초기 상태: 로컬 프레임 원점, 초기 속도 반영
        x0 = np.array([0., 0., 0., max(v0, V_MIN_LIN), 0., 0., 0., 0.])

        A_d, B_d = _linearize(x0)
        self.S_x, self.S_u = _build_prediction_matrices(A_d, B_d, n=N_LABEL)

        # 자유 응답 (제어 입력 = 0일 때의 상태 궤적)
        self._x_free = self.S_x @ x0   # ((N_LABEL+1)·NX,)

        # 참조 궤적 구성: (N_LABEL+1) × NX
        # k=0: 초기 위치 (로컬 원점)
        # k=1..N_LABEL: GT 스텝 0..N_LABEL-1 (t0+0.1s ~ t0+2.0s)
        x_ref = np.zeros((N_LABEL + 1, NX))
        for k in range(1, N_LABEL + 1):
            x_ref[k, IX]   = gt_xyz[k - 1, 0]
            x_ref[k, IY]   = gt_xyz[k - 1, 1]
            x_ref[k, IYAW] = gt_yaw[k - 1]
        self._x_ref_flat = x_ref.flatten()

        # ADE 계산용 GT xy: 첫 N_EVAL=15 스텝만 (1.5s) — 터미널 롤오프 배제
        self._gt_xy = gt_xyz[:N_EVAL, :2].copy()

        # 입력 제약 (N_LABEL 스텝 전체)
        self._lb = np.tile(U_MIN, N_LABEL)   # (N_LABEL·NU,) = (60,)
        self._ub = np.tile(U_MAX, N_LABEL)

        # ── 가중치 기저 행렬 사전계산 ────────────────────────────────────
        # H = S_u.T @ Q_blk @ S_u + R_blk 은 가중치에 대해 선형:
        #   H = Σ_i q_i * H_Q_basis[i]  +  Σ_j r_j * R_basis[j]
        # 기저 행렬을 __init__에 한 번만 계산해두면,
        # rollout_ade 호출마다 행렬 곱(~0.36ms) → 선형결합(~0.02ms)으로 단축.
        NNU = N_LABEL * NU                                          # = 60
        dx           = self._x_free - self._x_ref_flat             # ((N_LABEL+1)·NX,)
        S_u_blocks   = self.S_u.reshape(N_LABEL + 1, NX, NNU)      # (N_LABEL+1, NX, NNU)
        dx_blocks    = dx.reshape(N_LABEL + 1, NX)                 # (N_LABEL+1, NX)

        # Q 기저: 비영 대각 인덱스 [IX, IY, IYAW] = [0, 1, 2]
        # H_Q_basis[k] = Σ_t S_u_t[idx,:]^T @ S_u_t[idx,:]  (NNU×NNU)
        # g_Q_basis[k] = Σ_t S_u_t[idx,:]^T @ dx_t[idx]     (NNU,)
        # 터미널 스텝(k=N_LABEL)에 TERMINAL_Q_FACTOR 적용
        # u[N-1]의 레버리지를 보상하여 R 터미널 편향 완화
        _q_w = np.ones(N_LABEL + 1)
        _q_w[N_LABEL] = TERMINAL_Q_FACTOR

        self._H_Q_basis: list[np.ndarray] = []
        self._g_Q_basis: list[np.ndarray] = []
        for idx in [IX, IY, IYAW]:
            rows   = S_u_blocks[:, idx, :]                       # (N_LABEL+1, NNU)
            w_rows = rows * _q_w[:, None]                        # 터미널 스텝 가중 적용
            self._H_Q_basis.append(rows.T @ w_rows)              # (NNU, NNU)
            self._g_Q_basis.append(rows.T @ (_q_w * dx_blocks[:, idx]))  # (NNU,)

        # lat(IY) 기여는 W_LAT_FIXED 고정이므로 미리 흡수
        self._H_lat = W_LAT_FIXED * self._H_Q_basis[1]
        self._g_lat = W_LAT_FIXED * self._g_Q_basis[1]

        # R 기저: Δu 페널티 D.T @ R_b_diag @ D
        # j번째 입력 채널의 절대값 기저를 차분 행렬로 변환
        # → u[N-1]이 0이 아닌 u[N-2]로 수렴 (터미널 롤오프 구조적 제거)
        _D = _build_delta_matrix(N_LABEL)
        self._R_basis: list[np.ndarray] = []
        for j in range(NU):
            R_b = np.zeros((NNU, NNU))
            for k in range(N_LABEL):
                R_b[k * NU + j, k * NU + j] = 1.0
            self._R_basis.append(_D.T @ R_b @ _D)

        # 해석적 기울기용: 예측 XY 성분에 대한 S_u 슬라이스
        # 전체 N_LABEL 행 중 첫 N_EVAL=15 행만 (ADE 평가 범위와 일치)
        self._S_u_xy = self.S_u[NX:].reshape(N_LABEL, NX, NNU)[:N_EVAL, [IX, IY], :]  # (N_EVAL, 2, NNU)

    # ── Objective ─────────────────────────────────────

    def rollout_ade(self, theta: np.ndarray) -> float:
        """
        log-parameterized 가중치 theta (4,) → 정규화된 비용 [m].

        비용 = ADE(theta) + comfort_penalty(theta) + λ * ||theta - THETA0||²

        comfort_penalty = W_STEER_PENALTY * steer_rate_rms
                        + W_JERK_PENALTY  * jerk_rms
          - steer_rate_rms: sqrt(mean(diff(U_steer)²))  [rad/step]
          - jerk_rms:       sqrt(mean(diff(U_accel)²))  [(m/s²)/step]

        NOTE: comfort_penalty 는 U_opt 의 비선형 함수이므로
        해석적 기울기(_rollout_val_and_grad)가 더 이상 성립하지 않는다.
        최적화는 수치 미분(forward-diff)으로 폴백된다.
        """
        w = np.exp(np.clip(theta, -10., 10.))

        # H = W_LAT*H[IY] + w[0]*H[IX] + w[1]*H[IYAW] + w[2]*R[steer] + w[3]*R[accel] + QP_REG*I
        H = (self._H_lat                  # lat_position (고정)
           + w[0] * self._H_Q_basis[0]   # long_position
           + w[1] * self._H_Q_basis[2])  # heading
        H += w[2] * self._R_basis[0] + w[3] * self._R_basis[1]
        H  = 0.5 * (H + H.T) + QP_REG * np.eye(N_LABEL * NU)

        g = (self._g_lat
           + w[0] * self._g_Q_basis[0]
           + w[1] * self._g_Q_basis[2])

        U_opt   = _solve_qp(H, g, self._lb, self._ub)
        X_pred  = self._x_free + self.S_u @ U_opt
        # 첫 N_EVAL=15 스텝만 ADE 계산 (터미널 롤오프 배제)
        xy_pred = X_pred[NX:].reshape(N_LABEL, NX)[:N_EVAL, [IX, IY]]   # (N_EVAL, 2)

        ade = float(np.mean(np.linalg.norm(xy_pred - self._gt_xy, axis=1)))
        reg = _REG_LAMBDA * float(np.sum((theta - THETA0) ** 2))

        # ── 승차감 페널티 ─────────────────────────────────────
        U_2d           = U_opt.reshape(N_LABEL, NU)
        steer_rate_rms = float(np.sqrt(np.mean(np.diff(U_2d[:, 0]) ** 2)))
        jerk_rms       = float(np.sqrt(np.mean(np.diff(U_2d[:, 1]) ** 2)))
        comfort_penalty = W_STEER_PENALTY * steer_rate_rms + W_JERK_PENALTY * jerk_rms

        return ade + comfort_penalty + reg

    def _compute_comfort_metrics(self, theta: np.ndarray) -> tuple[float, float]:
        """
        최적 theta에서 승차감 지표 반환.

        Returns:
            (steer_rate_rms [rad/step], jerk_rms [(m/s²)/step])
        """
        w = np.exp(np.clip(theta, -10., 10.))

        H = (self._H_lat
           + w[0] * self._H_Q_basis[0]
           + w[1] * self._H_Q_basis[2])
        H += w[2] * self._R_basis[0] + w[3] * self._R_basis[1]
        H  = 0.5 * (H + H.T) + QP_REG * np.eye(N_LABEL * NU)

        g = (self._g_lat
           + w[0] * self._g_Q_basis[0]
           + w[1] * self._g_Q_basis[2])

        U_opt          = _solve_qp(H, g, self._lb, self._ub)
        U_2d           = U_opt.reshape(N_LABEL, NU)
        steer_rate_rms = float(np.sqrt(np.mean(np.diff(U_2d[:, 0]) ** 2)))
        jerk_rms       = float(np.sqrt(np.mean(np.diff(U_2d[:, 1]) ** 2)))
        return steer_rate_rms, jerk_rms

    def _compute_pure_ade(self, theta: np.ndarray) -> float:
        """
        순수 궤적 추종 ADE만 반환 (comfort_penalty · reg 제외).

        h5 저장 및 --ade-threshold 필터링 전용.
        최적화 목적함수로는 사용하지 않는다.
        """
        w = np.exp(np.clip(theta, -10., 10.))

        H = (self._H_lat
           + w[0] * self._H_Q_basis[0]
           + w[1] * self._H_Q_basis[2])
        H += w[2] * self._R_basis[0] + w[3] * self._R_basis[1]
        H  = 0.5 * (H + H.T) + QP_REG * np.eye(N_LABEL * NU)

        g = (self._g_lat
           + w[0] * self._g_Q_basis[0]
           + w[1] * self._g_Q_basis[2])

        U_opt   = _solve_qp(H, g, self._lb, self._ub)
        X_pred  = self._x_free + self.S_u @ U_opt
        xy_pred = X_pred[NX:].reshape(N_LABEL, NX)[:N_EVAL, [IX, IY]]
        return float(np.mean(np.linalg.norm(xy_pred - self._gt_xy, axis=1)))

    # ── Gradient (numerical fallback only) ───────────────

    def _rollout_val_and_grad(
        self, theta: np.ndarray
    ) -> tuple[float, np.ndarray | None]:
        """rollout_ade 값을 반환하고 기울기는 항상 None.

        comfort_penalty = W_STEER * steer_rate_rms + W_JERK * jerk_rms 는
        U_opt 의 비선형 함수이므로 기존 해석적 기울기 공식
          dADE/dθᵢ = -wᵢ · (H⁻¹ grad_U) @ (dH/dwᵢ U* + dg/dwᵢ)
        이 더 이상 성립하지 않는다.

        → 항상 (val, None) 반환 → _obj 에서 forward-diff 수치 미분 사용.
        속도는 ~5× 감소하지만, 정답 데이터 품질을 위해 수용한다.
        """
        return self.rollout_ade(theta), None

    # ── Outer Optimization ────────────────────────────

    def optimize(self, max_iter: int = 200) -> dict:
        """
        L-BFGS-B로 4개 가중치 (log-scale) 최적화.

        해석적 기울기(jac=True) 사용:
          - 비제약 케이스(다수): 정확한 기울기 → 수치 미분 대비 ~5× 빠름
          - 제약 활성 케이스:   수치 미분 폴백 (4 추가 호출)
        수렴 실패 시 Nelder-Mead로 폴백.
        """
        t0 = time.perf_counter()

        def _obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
            val, grad = self._rollout_val_and_grad(theta)
            if grad is not None:
                return val, grad
            # 수치 미분 폴백 (제약 활성 케이스)
            eps  = 0.05
            grad = np.empty(4)
            for i in range(4):
                th_h = theta.copy(); th_h[i] += eps
                grad[i] = (self.rollout_ade(th_h) - val) / eps
            return val, grad

        res = scipy.optimize.minimize(
            _obj,
            np.clip(THETA0, _THETA_MIN, _THETA_MAX),
            method="L-BFGS-B",
            jac=True,
            bounds=_BOUNDS_LOG,
            options={
                "maxiter": max_iter,
                "ftol":    1e-8,
                "gtol":    1e-5,
            },
        )

        # 수렴 실패 시 Nelder-Mead fallback
        if not res.success and res.fun > 0.5:
            res2 = scipy.optimize.minimize(
                self.rollout_ade,
                res.x,
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4, "adaptive": True},
            )
            if res2.fun < res.fun:
                res = res2

        elapsed = time.perf_counter() - t0
        w_opt   = np.exp(np.clip(res.x, -10., 10.))

        # 저장 순서: [long, lat(fixed), heading, steer_rate, accel_rate]
        weights = np.array([
            w_opt[0], W_LAT_FIXED, w_opt[1], w_opt[2], w_opt[3],
        ], dtype=np.float32)

        # h5 저장·필터링용 순수 ADE (comfort_penalty·reg 제외)
        pure_ade = self._compute_pure_ade(res.x)

        # 최적 가중치에서의 승차감 지표
        steer_rate_rms, jerk_rms = self._compute_comfort_metrics(res.x)

        return {
            "weights":        weights,
            "ade":            pure_ade,           # 순수 궤적 추종 ADE [m]
            "objective":      float(res.fun),     # 최적화 목적함수 (ADE+comfort+reg)
            "success":        bool(res.success),
            "n_evals":        int(res.nfev),
            "elapsed_s":      elapsed,
            "steer_rate_rms": steer_rate_rms,     # [rad/step]
            "jerk_rms":       jerk_rms,           # [(m/s²)/step]
        }


# ══════════════════════════════════════════════════════
# 멀티프로세싱 Worker 초기화
# ══════════════════════════════════════════════════════

def _init_worker(w_steer: float, w_jerk: float) -> None:
    """Pool worker 프로세스에서 comfort penalty 가중치를 전역 변수로 설정."""
    global W_STEER_PENALTY, W_JERK_PENALTY
    W_STEER_PENALTY = w_steer
    W_JERK_PENALTY  = w_jerk


# ══════════════════════════════════════════════════════
# 라벨 인덱스 (h5 스캔 없이 기완료 파일 추적)
# ══════════════════════════════════════════════════════

_INDEX_FILE = "labeled_index.txt"  # data_dir 내 위치


def _load_index(data_dir: Path) -> set[str]:
    """기완료 파일명(basename) 집합 로드. 파일 없으면 빈 집합 반환."""
    idx = data_dir / _INDEX_FILE
    if not idx.exists():
        return set()
    return set(idx.read_text(encoding="utf-8").splitlines())


def _save_index(data_dir: Path, labeled: set[str]) -> None:
    """기완료 파일명 집합을 인덱스 파일에 저장."""
    (data_dir / _INDEX_FILE).write_text(
        "\n".join(sorted(labeled)), encoding="utf-8"
    )


# ══════════════════════════════════════════════════════
# Worker: 단일 h5 파일 처리
# ══════════════════════════════════════════════════════

def _label_one(args: tuple) -> dict:
    h5_path, ade_threshold, max_iter = args
    name = Path(h5_path).name
    result = {"path": str(h5_path), "status": "error", "ade": float("nan")}

    try:
        with h5py.File(h5_path, "r") as f:
            gt_xyz = f["gt/future_xyz"][:]                # (64, 3)
            gt_ego = f["gt/future_ego_states"][:]         # (64, 5) [x,y,speed,yaw,lon_accel]
            hist   = f["input/ego_history_ego_states"][:]  # (16, 5)

        # t0 초기 상태
        v0       = float(hist[-1, 2])
        gt_yaw   = gt_ego[:, 3]
        # (gt_accel / accel0 제거)
        labeler = MPCLabeler(v0, gt_xyz, gt_yaw)
        opt = labeler.optimize(max_iter=max_iter)

        # h5에 결과 저장
        with h5py.File(h5_path, "a") as f:
            grp = f.require_group("labels")
            for key in ["mpc_weights", "ade", "valid", "n_evals",
                        "steer_rate_rms", "jerk_rms"]:
                if key in grp:
                    del grp[key]
            grp.create_dataset("mpc_weights",     data=opt["weights"])
            grp.create_dataset("ade",             data=np.float32(opt["ade"]))
            grp.create_dataset("valid",           data=bool(opt["ade"] < ade_threshold))
            grp.create_dataset("n_evals",         data=np.int32(opt["n_evals"]))
            grp.create_dataset("steer_rate_rms",  data=np.float32(opt["steer_rate_rms"]))
            grp.create_dataset("jerk_rms",        data=np.float32(opt["jerk_rms"]))

        status = "ok" if opt["ade"] < ade_threshold else "filtered"

        logging.debug(
            "%s | ADE=%.4f m | steer_rate_rms=%.4f rad/step | jerk_rms=%.4f (m/s²)/step"
            " | nfev=%d | %.2fs",
            name, opt["ade"],
            opt["steer_rate_rms"], opt["jerk_rms"],
            opt["n_evals"], opt["elapsed_s"],
        )

        result.update({
            "status":          status,
            "ade":             opt["ade"],
            "steer_rate_rms":  opt["steer_rate_rms"],
            "jerk_rms":        opt["jerk_rms"],
            "n_evals":         opt["n_evals"],
            "elapsed":         opt["elapsed_s"],
        })

    except Exception as e:
        result["error"] = str(e)
        logging.exception("ERROR %s — %s", name, e)

    return result


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MPC 가중치 자동 레이블링",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="수집된 .h5 파일 디렉토리",
    )
    parser.add_argument("--workers",         type=int,   default=4,   help="병렬 프로세스 수 (default: 4)")
    parser.add_argument("--ade-threshold",  type=float, default=0.5, help="필터링 임계값 [m] (default: 0.5)")
    parser.add_argument("--max-iter",       type=int,   default=500, help="L-BFGS-B / Nelder-Mead 최대 반복 (default: 500)")
    parser.add_argument("--overwrite",      action="store_true",     help="기존 labels/ 덮어쓰기")
    parser.add_argument("--dry-run",        action="store_true",     help="랜덤 샘플 N개만 처리 (페널티 튜닝 확인용)")
    parser.add_argument("--dry-run-n",      type=int,   default=100, help="dry-run 샘플 수 (default: 100)")
    parser.add_argument("--seed",           type=int,   default=None, help="랜덤 시드 (dry-run 시나리오 비교용)")
    parser.add_argument("--w-steer-penalty", type=float, default=15,
                        help=f"steering rate RMS 페널티 가중치 (default: {15})")
    parser.add_argument("--w-jerk-penalty",  type=float, default=1.5,
                        help=f"jerk RMS 페널티 가중치 (default: {1.5})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── comfort penalty 가중치 설정 ──────────────────────────────
    W_STEER_PENALTY = args.w_steer_penalty
    W_JERK_PENALTY  = args.w_jerk_penalty
    logging.info(
        "comfort penalty  w_steer=%.3f  w_jerk=%.3f",
        W_STEER_PENALTY, W_JERK_PENALTY,
    )

    if args.dry_run:
        # 전체 파일 목록 수집 후 무작위 셔플 → 앞 N개 선택
        all_files = list(data_dir.glob("*.h5"))
        rng = random.Random(args.seed)
        rng.shuffle(all_files)
        h5_files = all_files[: args.dry_run_n]
        labeled_set: set[str] = set()
        logging.info(
            "dry-run 모드: 전체 %d개 중 무작위 %d개 처리 (seed=%s, 인덱스 스킵)",
            len(all_files), len(h5_files),
            args.seed if args.seed is not None else "None",
        )
    else:
        h5_files = sorted(data_dir.glob("*.h5"))

        # 인덱스 기반 사전 필터링 (h5 오픈 없이 기완료 파일 스킵)
        labeled_set = _load_index(data_dir)
        if not args.overwrite and labeled_set:
            before = len(h5_files)
            h5_files = [f for f in h5_files if f.name not in labeled_set]
            logging.info(
                "인덱스 (%s): %d개 기완료 → %d개 처리 대상",
                _INDEX_FILE, before - len(h5_files), len(h5_files),
            )

    logging.info(
        "처리 대상: %d 파일 | workers=%d | ade_threshold=%.2f m",
        len(h5_files), args.workers, args.ade_threshold,
    )

    tasks = [
        (p, args.ade_threshold, args.max_iter)
        for p in h5_files
    ]

    results: list[dict] = []
    t_start = time.perf_counter()

    def _run_with_progress(iterator, total: int) -> None:
        """tqdm 진행바 + 500샘플마다 요약 통계 출력."""
        ok = filtered = errors = 0
        with tqdm(total=total, unit="샘플", desc="레이블링",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                  ) as pbar:
            for r in iterator:
                results.append(r)
                if   r["status"] == "ok":       ok += 1
                elif r["status"] == "filtered": filtered += 1
                elif r["status"] == "error":    errors   += 1

                pbar.set_postfix({
                    "ok": ok, "filt": filtered, "err": errors,
                    "ADE": f"{r['ade']:.3f}" if r["status"] == "ok" else "-",
                }, refresh=False)
                pbar.update(1)


    if args.workers == 1:
        _run_with_progress((_label_one(t) for t in tasks), len(tasks))
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(W_STEER_PENALTY, W_JERK_PENALTY),
        ) as pool:
            _run_with_progress(
                pool.imap_unordered(_label_one, tasks, chunksize=1),
                len(tasks),
            )

    elapsed = time.perf_counter() - t_start

    # ── 최종 통계 ──────────────────────────────────────
    by_status = {
        s: [r for r in results if r["status"] == s]
        for s in ("ok", "filtered", "skipped", "error")
    }

    logging.info("━━━ 레이블링 완료 ━━━")
    logging.info(
        "총 %d개: ok=%d  filtered=%d  skipped=%d  error=%d",
        len(results),
        len(by_status["ok"]),
        len(by_status["filtered"]),
        len(by_status["skipped"]),
        len(by_status["error"]),
    )
    logging.info(
        "경과: %.1fs (평균 %.2fs/샘플)",
        elapsed, elapsed / max(len(tasks), 1),
    )

    valid = by_status["ok"]
    if valid:
        ades = [r["ade"] for r in valid]
        logging.info(
            "유효 샘플 ADE: mean=%.4f  median=%.4f  max=%.4f  [m]",
            np.mean(ades), np.median(ades), np.max(ades),
        )
        steer_vals = [r["steer_rate_rms"] for r in valid if "steer_rate_rms" in r]
        jerk_vals  = [r["jerk_rms"]       for r in valid if "jerk_rms"       in r]
        if steer_vals:
            logging.info(
                "steer_rate_rms: mean=%.4f  median=%.4f  max=%.4f  [rad/step]",
                np.mean(steer_vals), np.median(steer_vals), np.max(steer_vals),
            )
        if jerk_vals:
            logging.info(
                "jerk_rms:       mean=%.4f  median=%.4f  max=%.4f  [(m/s²)/step]",
                np.mean(jerk_vals), np.median(jerk_vals), np.max(jerk_vals),
            )

    if by_status["error"]:
        logging.warning("오류 파일 (최대 10개):")
        for r in by_status["error"][:10]:
            logging.warning("  %s — %s", Path(r["path"]).name, r.get("error", "?"))

    # ── 인덱스 업데이트 (dry-run 제외) ─────────────────
    if not args.dry_run:
        newly_done = {Path(r["path"]).name
                      for r in results if r["status"] in ("ok", "filtered")}
        updated = newly_done | (labeled_set if not args.overwrite else set())
        _save_index(data_dir, updated)
        logging.info("인덱스 업데이트: %d개 → %s", len(updated), data_dir / _INDEX_FILE)



if __name__ == "__main__":
    main()
