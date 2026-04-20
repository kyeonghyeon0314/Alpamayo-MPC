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
      --data-dir /workspace/alpamayo_dataset/data/prepare/train \
      --dry-run

  # 전체 실행
  python /workspace/alpamayo_dataset/label_mpc_weights.py \
      --data-dir /workspace/alpamayo_dataset/data/prepare/train \
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
    N, N_LABEL, N_EVAL, NX, NU, DT,
    IX, IY, IYAW, IVX, IVY, IYR, ISTEER, IACCEL, WB,
    V_MIN_LIN, QP_REG, U_MIN, U_MAX, W_LAT_FIXED, TERMINAL_Q_FACTOR,
    _linearize, _build_prediction_matrices_tv, _solve_qp, _build_delta_matrix,
    compute_x0,
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
# 범위를 좁혀 equifinality 억제: heading/steer_rate가 경계로 수렴하는 현상 방지
#                         long   head   steer_r  accel_r
_THETA_MIN = np.log([ 0.5,   0.5,   1.0,    0.2  ])
_THETA_MAX = np.log([10.,    5.,   10.,     5.  ])
_BOUNDS_LOG = list(zip(_THETA_MIN, _THETA_MAX))

# ── 목적함수 가중치 ──────────────────────────────────────────
# 목적함수 = W_ADE_PENALTY * (ADE_xy + W_YAW_ADE * ADE_yaw)
# ADE_xy [m], ADE_yaw [rad]
# W_YAW_ADE=0 이면 기존 xy-only ADE와 동일
W_ADE_PENALTY: float = 1.0
W_YAW_ADE:     float = 2.0   # yaw 오차 가중치 [rad → m 등가 스케일]

# ══════════════════════════════════════════════════════
# GT 명목 상태 복원 (LTV 선형화용)
# ══════════════════════════════════════════════════════

def _build_gt_full_states(
    x0:           np.ndarray,
    gt_xy:        np.ndarray,
    gt_yaw:       np.ndarray,
    gt_speed:     np.ndarray,
    gt_lon_accel: np.ndarray,
) -> list[np.ndarray]:
    """GT smooth 궤적에서 LTV 선형화용 명목 상태 복원.

    Args:
        x0:           (NX,)         t0 초기 상태 — compute_x0() 결과
        gt_xy:        (≥N_LABEL, 2) smooth 미래 xy [m]  (로컬 프레임)
        gt_yaw:       (≥N_LABEL,)   smooth 미래 yaw [rad]
        gt_speed:     (≥N_LABEL,)   종방향 속도 [m/s]  (gt/future_ego_states[:,2])
        gt_lon_accel: (≥N_LABEL,)   종방향 가속도 [m/s²] (gt/future_ego_states[:,4])

    Returns:
        states: list of (NX,) arrays, length N_LABEL
          states[0] = x0  (정확한 t0 초기 상태)
          states[k] (k≥1): t0+k*DT 시점의 NX=8 상태 추정값
            [x, y, yaw, vx, vy, yaw_rate, steering, lon_accel]
    """
    # ── t0를 포함한 연장 배열 구성 (인덱스 0=t0, 1..N_LABEL=GT) ──
    xy_ext    = np.vstack([[0., 0.], gt_xy[:N_LABEL].astype(np.float64)])          # (N_LABEL+1, 2)
    yaw_uw    = np.unwrap(
        np.concatenate([[0.], gt_yaw[:N_LABEL].astype(np.float64)])
    )                                                                               # (N_LABEL+1,)
    speed_ext = np.concatenate(
        [[float(x0[IVX])], gt_speed[:N_LABEL].astype(np.float64)]
    )                                                                               # (N_LABEL+1,)
    accel_ext = np.concatenate(
        [[float(x0[IACCEL])], gt_lon_accel[:N_LABEL].astype(np.float64)]
    )                                                                               # (N_LABEL+1,)

    # ── yaw_rate: unwrapped yaw의 중앙 차분 ──
    yr_ext = np.gradient(yaw_uw, DT)   # (N_LABEL+1,)

    # ── local-frame 속도: 위치 중앙 차분 → body frame 변환으로 vy 추정 ──
    vl_x = np.gradient(xy_ext[:, 0], DT)  # (N_LABEL+1,)
    vl_y = np.gradient(xy_ext[:, 1], DT)  # (N_LABEL+1,)

    states: list[np.ndarray] = []
    for k in range(N_LABEL):
        if k == 0:
            # t0: compute_x0() 결과 그대로 사용 (가장 정확)
            states.append(x0)
            continue

        yaw_k   = float(yaw_uw[k])
        cy, sy  = np.cos(yaw_k), np.sin(yaw_k)
        vx_k    = max(float(speed_ext[k]), V_MIN_LIN)
        # R(-yaw) @ v_local → body-frame vy
        vy_k    = float(-sy * vl_x[k] + cy * vl_y[k])
        yr_k    = float(yr_ext[k])
        steer_k = float(np.arctan2(yr_k * WB, vx_k))

        states.append(np.array([
            xy_ext[k, 0], xy_ext[k, 1], yaw_k,
            vx_k, vy_k, yr_k, steer_k, float(accel_ext[k]),
        ]))

    return states  # length N_LABEL


# ══════════════════════════════════════════════════════
# 핵심 클래스: 샘플별 MPC Labeler
# ══════════════════════════════════════════════════════

class MPCLabeler:
    """
    단일 샘플에 대한 MPC 가중치 역산기.

    GT 궤적 각 스텝에서 선형화(LTV) → S_x, S_u를 __init__에서 1회 구성 →
    L-BFGS-B 반복마다 QP만 재풀이 (pseudo-closed-loop).

    솔버 호라이즌: N_LABEL=N=20 (배포와 동일)
    ADE 평가 구간: N_EVAL=15 (1.5s, 터미널 롤오프 5스텝 제외)
    R 페널티: Δu 기반 (D.T @ R_blk @ D) — 터미널 수렴 구조적 억제

    Pseudo-closed-loop 의미:
      - GT smooth 궤적을 명목 궤적으로 사용 → 각 스텝 k마다 A_d[k], B_d[k] 선형화
      - 코너/가속 구간에서 yaw·vx 변화에 따른 시변 dynamics 정확 반영
      - S_x, S_u는 __init__에서 고정 → 가중치 최적화 루프 구조 동일 (QP만 반복)
    """

    def __init__(
        self,
        x0:           np.ndarray,
        gt_xy:        np.ndarray,
        gt_yaw:       np.ndarray,
        gt_speed:     np.ndarray,
        gt_lon_accel: np.ndarray,
    ):
        """
        Args:
            x0:           (NX=8,)       MPC 초기 상태 — compute_x0()로 생성
            gt_xy:        (≥N_LABEL, 2) GT smooth 미래 xy (로컬 프레임)
            gt_yaw:       (≥N_LABEL,)   GT smooth 미래 yaw [rad]
            gt_speed:     (≥N_LABEL,)   GT 종방향 속도 [m/s]  (future_ego_states[:,2])
            gt_lon_accel: (≥N_LABEL,)   GT 종방향 가속도 [m/s²] (future_ego_states[:,4])
        """
        # ── LTV: GT 궤적 각 스텝에서 선형화 ────────────────────────────
        nom_states      = _build_gt_full_states(x0, gt_xy, gt_yaw, gt_speed, gt_lon_accel)
        A_list, B_list  = zip(*[_linearize(s) for s in nom_states])
        self.S_x, self.S_u = _build_prediction_matrices_tv(list(A_list), list(B_list))

        # 자유 응답 (제어 입력 = 0일 때의 상태 궤적)
        self._x_free = self.S_x @ x0   # ((N_LABEL+1)·NX,)

        # 참조 궤적 구성: (N_LABEL+1) × NX
        # k=0: 초기 위치 (로컬 원점)
        # k=1..N_LABEL: GT 스텝 0..N_LABEL-1 (t0+0.1s ~ t0+2.0s)
        x_ref = np.zeros((N_LABEL + 1, NX))
        for k in range(1, N_LABEL + 1):
            x_ref[k, IX]   = gt_xy[k - 1, 0]
            x_ref[k, IY]   = gt_xy[k - 1, 1]
            x_ref[k, IYAW] = gt_yaw[k - 1]
        self._x_ref_flat = x_ref.flatten()

        # ADE 계산용 GT: 첫 N_EVAL=15 스텝만 (1.5s) — 터미널 롤오프 배제
        self._gt_xy  = gt_xy[:N_EVAL, :2].copy()     # (N_EVAL, 2)
        self._gt_yaw = gt_yaw[:N_EVAL].copy()        # (N_EVAL,)

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

        # 해석적 기울기용: S_u 슬라이스 (첫 N_EVAL 행만)
        _S_u_blk     = self.S_u[NX:].reshape(N_LABEL, NX, NNU)[:N_EVAL]  # (N_EVAL, NX, NNU)
        self._S_u_xy  = _S_u_blk[:, [IX, IY], :]   # (N_EVAL, 2, NNU)
        self._S_u_yaw = _S_u_blk[:, IYAW, :]        # (N_EVAL, NNU)

    # ── Objective ─────────────────────────────────────

    def rollout_ade(self, theta: np.ndarray) -> float:
        """
        log-parameterized 가중치 theta (4,) → 목적함수 값 [m 등가].

        목적함수 = W_ADE_PENALTY * (ADE_xy + W_YAW_ADE * ADE_yaw)

          ADE_xy  [m]   — 첫 N_EVAL=15 스텝 xy 평균 L2 거리
          ADE_yaw [rad] — 첫 N_EVAL=15 스텝 yaw 평균 절대 오차

        승차감 페널티(steer_rate_rms, jerk_rms)는 제거됨.
        Smooth GT가 smooth한 yaw 레퍼런스를 제공하므로, 자연히 부드러운 제어가 유도됨.
        """
        val, _ = self._rollout_val_and_grad(theta)
        return val

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

    def _compute_ade_xy(self, theta: np.ndarray) -> float:
        """xy-only ADE [m] — h5 저장 및 ade-threshold 필터링 전용."""
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

    def _compute_ade_yaw(self, theta: np.ndarray) -> float:
        """yaw-only ADE [rad] — h5 저장 전용."""
        w = np.exp(np.clip(theta, -10., 10.))

        H = (self._H_lat
           + w[0] * self._H_Q_basis[0]
           + w[1] * self._H_Q_basis[2])
        H += w[2] * self._R_basis[0] + w[3] * self._R_basis[1]
        H  = 0.5 * (H + H.T) + QP_REG * np.eye(N_LABEL * NU)

        g = (self._g_lat
           + w[0] * self._g_Q_basis[0]
           + w[1] * self._g_Q_basis[2])

        U_opt    = _solve_qp(H, g, self._lb, self._ub)
        X_pred   = self._x_free + self.S_u @ U_opt
        yaw_pred = X_pred[NX:].reshape(N_LABEL, NX)[:N_EVAL, IYAW]
        return float(np.mean(np.abs(yaw_pred - self._gt_yaw)))

    # ── Value + Analytical Gradient ──────────────────────

    def _rollout_val_and_grad(
        self, theta: np.ndarray
    ) -> tuple[float, np.ndarray | None]:
        """목적함수 값과 해석적 기울기를 동시에 반환.

        목적함수:
          J = W_ADE * (ADE_xy + W_YAW * ADE_yaw) + λ||θ-θ₀||²

        해석적 기울기 (제약 비활성 케이스):
          U* = -H⁻¹ g  →  dU*/dθᵢ = -H⁻¹ vᵢ  (vᵢ = dH/dθᵢ·U* + dg/dθᵢ)
          grad_U = dJ/dU* = W_ADE*(dADE_xy/dU* + W_YAW*dADE_yaw/dU*)
          dJ/dθᵢ = -grad_U · H⁻¹vᵢ

        제약 활성 케이스: None 반환 → _obj 에서 수치 미분 폴백.
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

        U_opt  = _solve_qp(H, g, self._lb, self._ub)
        X_pred = self._x_free + self.S_u @ U_opt
        Xr     = X_pred[NX:].reshape(N_LABEL, NX)

        xy_pred  = Xr[:N_EVAL, [IX, IY]]   # (N_EVAL, 2)
        yaw_pred = Xr[:N_EVAL, IYAW]       # (N_EVAL,)

        e_xy  = xy_pred  - self._gt_xy     # (N_EVAL, 2)
        e_yaw = yaw_pred - self._gt_yaw    # (N_EVAL,)

        norms   = np.linalg.norm(e_xy, axis=1)          # (N_EVAL,)
        ade_xy  = float(np.mean(norms))
        ade_yaw = float(np.mean(np.abs(e_yaw)))
        val     = W_ADE_PENALTY * (ade_xy + W_YAW_ADE * ade_yaw)

        # ── 제약 활성 여부 확인 → 활성 시 수치 미분 폴백 ──────
        _TOL = 1e-4
        if np.any(U_opt <= self._lb + _TOL) or np.any(U_opt >= self._ub - _TOL):
            return val, None

        # ── 해석적 기울기 ──────────────────────────────────────
        NNU = N_LABEL * NU

        # dADE_xy/dU*: (NNU,)
        safe_norms  = np.maximum(norms, 1e-6)
        e_xy_normed = e_xy / safe_norms[:, None]              # (N_EVAL, 2)
        d_xy_dU     = np.einsum('ki,kij->j', e_xy_normed, self._S_u_xy) / N_EVAL

        # dADE_yaw/dU*: (NNU,)
        d_yaw_dU = (np.sign(e_yaw) @ self._S_u_yaw) / N_EVAL

        # 목적함수의 U* 기울기
        grad_U = W_ADE_PENALTY * (d_xy_dU + W_YAW_ADE * d_yaw_dU)  # (NNU,)

        # vᵢ = dH/dθᵢ · U* + dg/dθᵢ  (NNU, 4)
        V = np.column_stack([
            w[0] * (self._H_Q_basis[0] @ U_opt + self._g_Q_basis[0]),  # long
            w[1] * (self._H_Q_basis[2] @ U_opt + self._g_Q_basis[2]),  # yaw
            w[2] * (self._R_basis[0]   @ U_opt),                        # steer_r
            w[3] * (self._R_basis[1]   @ U_opt),                        # accel_r
        ])

        # dJ/dθᵢ = -grad_U · H⁻¹ vᵢ  (solve 4 RHS at once)
        grad = -(grad_U @ np.linalg.solve(H, V))   # (4,)

        return val, grad

    # ── Outer Optimization ────────────────────────────

    def optimize(self, max_iter: int = 200,
                 theta_prev: np.ndarray | None = None,
                 lambda_smooth: float = 0.0) -> dict:
        """
        L-BFGS-B로 4개 가중치 (log-scale) 최적화.

        theta_prev:    이전 프레임의 theta (클립 내 warm-start 및 smoothness 페널티용)
        lambda_smooth: 클립 내 시간적 일관성 강도. 목적함수에
                       lambda_smooth * ||θ - θ_prev||² 항 추가.

        해석적 기울기(jac=True) 사용:
          - 비제약 케이스(다수): 정확한 기울기 → 수치 미분 대비 ~5× 빠름
          - 제약 활성 케이스:   수치 미분 폴백 (4 추가 호출)
        수렴 실패 시 Nelder-Mead로 폴백.
        """
        t0 = time.perf_counter()

        use_smooth = lambda_smooth > 0.0 and theta_prev is not None

        def _obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
            val, grad = self._rollout_val_and_grad(theta)
            if use_smooth:
                diff  = theta - theta_prev
                val  += lambda_smooth * float(np.dot(diff, diff))
                if grad is not None:
                    grad = grad + 2.0 * lambda_smooth * diff
            if grad is not None:
                return val, grad
            # 수치 미분 폴백 (제약 활성 케이스)
            eps  = 0.05
            grad = np.empty(4)
            for i in range(4):
                th_h = theta.copy(); th_h[i] += eps
                grad[i] = (self.rollout_ade(th_h) - val) / eps
            return val, grad

        theta_init = np.clip(
            theta_prev if theta_prev is not None else THETA0,
            _THETA_MIN, _THETA_MAX,
        )
        res = scipy.optimize.minimize(
            _obj,
            theta_init,
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

        ade_xy  = self._compute_ade_xy(res.x)
        ade_yaw = self._compute_ade_yaw(res.x)
        steer_rate_rms, jerk_rms = self._compute_comfort_metrics(res.x)

        return {
            "weights":        weights,
            "theta_opt":      res.x.copy(),       # 클립 내 다음 프레임 warm-start용
            "ade":            ade_xy,
            "ade_yaw":        ade_yaw,
            "objective":      float(res.fun),
            "success":        bool(res.success),
            "n_evals":        int(res.nfev),
            "elapsed_s":      elapsed,
            "steer_rate_rms": steer_rate_rms,
            "jerk_rms":       jerk_rms,
        }


# ══════════════════════════════════════════════════════
# 멀티프로세싱 Worker 초기화
# ══════════════════════════════════════════════════════

def _init_worker(w_ade: float, w_yaw: float) -> None:
    """Pool worker 프로세스에서 objective 가중치를 전역 변수로 설정."""
    global W_ADE_PENALTY, W_YAW_ADE
    W_ADE_PENALTY = w_ade
    W_YAW_ADE     = w_yaw


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
    h5_path, ade_threshold, max_iter, theta_prev, lambda_smooth = args
    name = Path(h5_path).name
    result = {"path": str(h5_path), "status": "error", "ade": float("nan")}

    try:
        with h5py.File(h5_path, "r") as f:
            hist      = f["input/ego_history_ego_states"][:]  # (16, 5)
            hist_vel  = f["input/ego_history_vel"][:]          # (16, 3) global frame
            hist_curv = f["input/ego_history_curv"][:]         # (16, 1)
            hist_quat = f["input/ego_history_quat_global"][:]  # (16, 4) [qx,qy,qz,qw]

            gt_ego_raw   = f["gt/future_ego_states"][:]        # (64, 5)
            gt_speed     = gt_ego_raw[:, 2]
            gt_lon_accel = gt_ego_raw[:, 4]

            if "gt_smooth" in f:
                gt_xy  = f["gt_smooth/future_xy"][:]           # (64, 2)
                gt_yaw = f["gt_smooth/future_yaw"][:]          # (64,)
            else:
                gt_xyz = f["gt/future_xyz"][:]                 # (64, 3)
                gt_xy  = gt_xyz[:, :2]
                gt_yaw = gt_ego_raw[:, 3]

        x0      = compute_x0(hist[-1, 2], hist[-1, 4], hist_vel[-1], hist_curv[-1, 0], hist_quat[-1])
        labeler = MPCLabeler(x0, gt_xy, gt_yaw, gt_speed, gt_lon_accel)
        opt     = labeler.optimize(max_iter=max_iter,
                                   theta_prev=theta_prev,
                                   lambda_smooth=lambda_smooth)

        # h5에 결과 저장
        with h5py.File(h5_path, "a") as f:
            grp = f.require_group("labels")
            for key in ["mpc_weights", "ade", "ade_yaw", "valid", "n_evals",
                        "steer_rate_rms", "jerk_rms"]:
                if key in grp:
                    del grp[key]
            grp.create_dataset("mpc_weights",    data=opt["weights"])
            grp.create_dataset("ade",            data=np.float32(opt["ade"]))
            grp.create_dataset("ade_yaw",        data=np.float32(opt["ade_yaw"]))
            grp.create_dataset("valid",          data=bool(opt["ade"] < ade_threshold))
            grp.create_dataset("n_evals",        data=np.int32(opt["n_evals"]))
            grp.create_dataset("steer_rate_rms", data=np.float32(opt["steer_rate_rms"]))
            grp.create_dataset("jerk_rms",       data=np.float32(opt["jerk_rms"]))

        status = "ok" if opt["ade"] < ade_threshold else "filtered"

        logging.debug(
            "%s | ADE_xy=%.4f m | steer_rms=%.4f | jerk_rms=%.4f | nfev=%d | %.2fs",
            name, opt["ade"], opt["steer_rate_rms"], opt["jerk_rms"],
            opt["n_evals"], opt["elapsed_s"],
        )

        result.update({
            "status":          status,
            "ade":             opt["ade"],
            "theta_opt":       opt["theta_opt"],   # 다음 프레임 warm-start용
            "steer_rate_rms":  opt["steer_rate_rms"],
            "jerk_rms":        opt["jerk_rms"],
            "n_evals":         opt["n_evals"],
            "elapsed":         opt["elapsed_s"],
        })

    except Exception as e:
        result["error"] = str(e)
        logging.exception("ERROR %s — %s", name, e)

    return result


def _clip_sort_key(p: Path) -> int:
    """파일명 {uuid}__{timestamp}.h5 에서 타임스탬프(정수)를 추출해 시간순 정렬."""
    try:
        return int(p.stem.split("__")[1])
    except (IndexError, ValueError):
        return 0


def _label_clip(args: tuple) -> list[dict]:
    """
    같은 클립에 속한 h5 파일들을 시간순으로 처리.
    이전 프레임의 theta_opt를 다음 프레임의 warm-start 및 smoothness 기준으로 전달.
    """
    clip_files, ade_threshold, max_iter, lambda_smooth = args
    sorted_files = sorted(clip_files, key=_clip_sort_key)
    theta_prev   = None   # 첫 프레임은 THETA0에서 시작
    results      = []

    for h5_path in sorted_files:
        r = _label_one((str(h5_path), ade_threshold, max_iter, theta_prev, lambda_smooth))
        results.append(r)
        # 성공한 프레임의 theta만 다음 프레임으로 전달
        if r.get("status") in ("ok", "filtered") and "theta_opt" in r:
            theta_prev = r["theta_opt"]

    return results


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
        help="prepare/ 베이스 디렉토리 (하위 splits를 자동 탐색)",
    )
    parser.add_argument("--splits",         nargs="+", default=["train", "val", "test"],
                        metavar="SPLIT",    help="처리할 split (default: train val test)")
    parser.add_argument("--workers",        type=int,   default=8,   help="병렬 프로세스 수 (default: 4)")
    parser.add_argument("--ade-threshold",  type=float, default=0.5, help="필터링 임계값 [m] (default: 0.5)")
    parser.add_argument("--max-iter",       type=int,   default=500, help="L-BFGS-B / Nelder-Mead 최대 반복 (default: 500)")
    parser.add_argument("--overwrite",      action="store_true",     help="기존 labels/ 덮어쓰기")
    parser.add_argument("--dry-run",        action="store_true",     help="랜덤 샘플 N개만 처리 (페널티 튜닝 확인용)")
    parser.add_argument("--dry-run-n",      type=int,   default=100, help="dry-run 샘플 수 (default: 100)")
    parser.add_argument("--seed",           type=int,   default=None, help="랜덤 시드 (dry-run 시나리오 비교용)")
    parser.add_argument("--w-ade-penalty",  type=float, default=1.0,
                        help="ADE 전체 스케일 가중치 (default: 1.0)")
    parser.add_argument("--w-yaw-ade",      type=float, default=2.0,
                        help="yaw ADE 가중치 [rad → m 등가] (default: 2.0, 0=xy-only)")
    parser.add_argument("--lambda-smooth",  type=float, default=0.1,
                        help="클립 내 시간적 일관성 강도 (default: 0.1, 0=비활성)")
    parser.add_argument("--viz-n",          type=int,   default=3,
                        help="레이블링 후 시각화할 랜덤 클립 수 (default: 3, 0=생략). "
                             "선택된 클립의 모든 프레임을 시간순으로 저장.")
    parser.add_argument("--viz-dir",        type=str,   default=None,
                        help="시각화 PNG 저장 경로 (default: --data-dir/viz)")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)

    # ── objective 가중치 설정 (main 프로세스용) ──────────────────
    global W_ADE_PENALTY, W_YAW_ADE
    W_ADE_PENALTY   = args.w_ade_penalty
    W_YAW_ADE       = args.w_yaw_ade
    lambda_smooth   = args.lambda_smooth
    logging.info(
        "objective  w_ade=%.3f  w_yaw_ade=%.3f  lambda_smooth=%.4f",
        W_ADE_PENALTY, W_YAW_ADE, lambda_smooth,
    )

    # ── split별 파일 수집 + 인덱스 기반 사전 필터링 ──────────────
    # labeled_sets: split_dir → 기완료 파일명 집합 (인덱스 업데이트에 재사용)
    h5_files:    list[Path]          = []
    split_dirs:  list[Path]          = []
    labeled_sets: dict[Path, set[str]] = {}

    for split in args.splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            logging.warning("split 디렉토리 없음, 건너뜀: %s", split_dir)
            continue
        split_files = sorted(split_dir.glob("*.h5"))
        logging.info("split=%s: %d개 파일", split, len(split_files))

        if not args.dry_run:
            lset = _load_index(split_dir)
            labeled_sets[split_dir] = lset
            if not args.overwrite and lset:
                before = len(split_files)
                split_files = [f for f in split_files if f.name not in lset]
                logging.info(
                    "  인덱스: %d개 기완료 → %d개 처리 대상",
                    before - len(split_files), len(split_files),
                )
        else:
            labeled_sets[split_dir] = set()

        split_dirs.append(split_dir)
        h5_files.extend(split_files)

    if not h5_files and not args.dry_run:
        logging.error("처리할 .h5 파일 없음 (base=%s, splits=%s)", base_dir, args.splits)
        return

    if args.dry_run:
        # dry-run: 전체 목록에서 무작위 N개 선택 (인덱스 무시)
        all_files: list[Path] = []
        for split_dir in split_dirs if split_dirs else [base_dir / s for s in args.splits]:
            all_files.extend(sorted(split_dir.glob("*.h5")))
        rng = random.Random(args.seed)
        rng.shuffle(all_files)
        h5_files = all_files[: args.dry_run_n]
        logging.info(
            "dry-run 모드: 전체 %d개 중 무작위 %d개 처리 (seed=%s, 인덱스 스킵)",
            len(all_files), len(h5_files),
            args.seed if args.seed is not None else "None",
        )

    # ── 클립 단위로 그룹화 ────────────────────────────────
    clip_groups: dict[str, list[Path]] = {}
    for p in h5_files:
        clip_id = p.stem.split("__")[0]
        clip_groups.setdefault(clip_id, []).append(p)

    n_clips = len(clip_groups)
    logging.info(
        "처리 대상: %d 파일 / %d 클립 | splits=%s | workers=%d | "
        "ade_threshold=%.2f m | lambda_smooth=%.3f",
        len(h5_files), n_clips, args.splits, args.workers,
        args.ade_threshold, lambda_smooth,
    )

    clip_tasks = [
        (files, args.ade_threshold, args.max_iter, lambda_smooth)
        for files in clip_groups.values()
    ]

    results: list[dict] = []
    t_start = time.perf_counter()

    def _run_with_progress(clip_iterator, total_files: int) -> None:
        """클립 단위 iterator에서 프레임 결과를 flatten해 tqdm 진행바 표시."""
        ok = filtered = errors = 0
        with tqdm(total=total_files, unit="샘플", desc="레이블링",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                  ) as pbar:
            for clip_results in clip_iterator:
                for r in clip_results:
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
        _run_with_progress((_label_clip(t) for t in clip_tasks), len(h5_files))
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(W_ADE_PENALTY, W_YAW_ADE),
        ) as pool:
            _run_with_progress(
                pool.imap_unordered(_label_clip, clip_tasks, chunksize=1),
                len(h5_files),
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
        elapsed, elapsed / max(len(h5_files), 1),
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

    # ── 인덱스 업데이트 (dry-run 제외, split별로 저장) ─────────────
    if not args.dry_run:
        for split_dir in split_dirs:
            newly_done = {
                Path(r["path"]).name
                for r in results
                if r["status"] in ("ok", "filtered")
                and Path(r["path"]).parent == split_dir
            }
            prev = labeled_sets.get(split_dir, set())
            updated = newly_done | (prev if not args.overwrite else set())
            _save_index(split_dir, updated)
            logging.info("인덱스 업데이트 [%s]: %d개 → %s",
                         split_dir.name, len(updated), split_dir / _INDEX_FILE)

    # ── 시각화 (랜덤 N 클립, 클립 내 프레임 시간순) ──────
    if args.viz_n > 0:
        viz_dir = Path(args.viz_dir) if args.viz_dir else base_dir / "label_viz"
        _save_label_viz(
            ok_paths=[Path(r["path"]) for r in by_status["ok"]],
            viz_dir=viz_dir,
            n_clips=args.viz_n,
            seed=args.seed if args.seed is not None else 42,
        )


def _save_label_viz(
    ok_paths:  list[Path],
    viz_dir:   Path,
    n_clips:   int,
    seed:      int,
) -> None:
    """
    성공 샘플을 클립 단위로 그룹화한 뒤 랜덤으로 n_clips개 클립을 선택해
    각 클립 내 프레임을 시간순으로 연속 저장.
    출력 파일명: {clip_id}__{timestamp}_mpc_viz.png (시간순 정렬로 탐색기에서 연속 표시)
    """
    if not ok_paths:
        logging.warning("No labeled samples to visualize.")
        return

    # 클립 단위 그룹화
    clip_map: dict[str, list[Path]] = {}
    for p in ok_paths:
        clip_id = p.stem.split("__")[0]
        clip_map.setdefault(clip_id, []).append(p)

    import random as _rnd
    rng = _rnd.Random(seed)
    selected_clips = rng.sample(sorted(clip_map.keys()), min(n_clips, len(clip_map)))

    from viz_mpc_label import plot_single

    viz_dir.mkdir(parents=True, exist_ok=True)
    for clip_id in selected_clips:
        frames = sorted(clip_map[clip_id], key=_clip_sort_key)
        logging.info("클립 [%s] %d 프레임 시각화", clip_id[:8], len(frames))
        for h5_path in frames:
            try:
                plot_single(h5_path, viz_dir)
                logging.info("  Viz saved: %s", viz_dir / f"{h5_path.stem}_mpc_viz.png")
            except Exception as e:
                logging.warning("  Viz failed %s — %s", h5_path.name, e)


if __name__ == "__main__":
    main()
