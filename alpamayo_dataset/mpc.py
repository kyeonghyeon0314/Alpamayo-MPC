#!/usr/bin/env python3
"""
공유 MPC 모델 (alpasim MPCController 기반)

label_mpc_weights.py / viz_mpc_label.py / test.py 공통 사용:
  - 차량 파라미터 / MPC 상수
  - 동적 자전거 모델 선형화 · 이산화 (ZOH)
  - Condensed 예측 행렬 구성
  - QP 솔버 (unconstrained → L-BFGS-B fallback)
  - run_mpc() 고수준 인터페이스
  - compute_x0() h5 raw egomotion → MPC 초기 상태 복원

N (=20)     : 추론 / 시각화 호라이즌 (2s, 10Hz)
N_LABEL     : label_mpc_weights.py 전용 — N과 동일 (=20)
N_EVAL (=15): IOC ADE 평가 구간 — 마지막 5스텝 터미널 롤오프 제거 (1.5s)

R 구조: 입력 절대값이 아닌 변화량(Δu) 페널티 사용.
  H_R = D^T R_blk D  (D: 블록 차분 행렬, u[-1]=0)
  → u[N-1]이 0이 아닌 u[N-2] 방향으로 수렴, 터미널 롤오프 구조적 제거.
"""

import math

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.spatial.transform import Rotation as _Rotation

# ══════════════════════════════════════════════════════
# MPC 파라미터
# ══════════════════════════════════════════════════════
DT = 0.1
N       = 20    # QP 호라이즌 (2s, 10Hz) — IOC / 시각화 / 배포 모두 동일
N_LABEL = N     # label_mpc_weights.py 호환용 alias
N_EVAL  = 15    # IOC ADE 평가 구간 (1.5s) — 마지막 5스텝 터미널 롤오프 배제
TERMINAL_Q_FACTOR = 5.0  # 터미널 스텝(k=N) Q 가중치 배율 — 추가 안전마진
NX = 8     # 상태: [x, y, yaw, vx, vy, yaw_rate, steering, accel]
NU = 2     # 입력: [steering_cmd, accel_cmd]

IX, IY, IYAW, IVX, IVY, IYR, ISTEER, IACCEL = range(8)

# ══════════════════════════════════════════════════════
# 차량 파라미터 (alpasim VehicleModel.Parameters 기본값)
# ══════════════════════════════════════════════════════
MASS    = 2014.4    # [kg]
INERTIA = 3414.2    # [kg·m²]
L_CG    = 1.59      # rear axle ~ CG [m]
WB      = 2.85      # wheelbase [m]
CF      = 93534.5   # 전륜 코너링 강성 [N/rad]
CR      = 176162.1  # 후륜 코너링 강성 [N/rad]
TAU_S   = 0.1       # 조향 actuator 시상수 [s]
TAU_A   = 0.1       # 가속 actuator 시상수 [s]

V_MIN_LIN = 1.0     # 선형화 최소 속도 (특이점 방지) [m/s]
QP_REG    = 1e-6    # QP Hessian 정규화

U_MIN = np.array([-2.0, -9.0])   # [steer_cmd, accel_cmd] 하한
U_MAX = np.array([ 2.0,  6.0])   # [steer_cmd, accel_cmd] 상한

# ══════════════════════════════════════════════════════
# 가중치
# ══════════════════════════════════════════════════════
W_LAT_FIXED = 1.0   # lat_position_weight 고정 (IOC 정규화 기준)

# alpasim 기본값 [long, lat(fixed), heading, steer_r, accel_r]
WEIGHTS_DEFAULT = np.array([2.0, 1.0, 1.0, 5.0, 1.0])


# ══════════════════════════════════════════════════════
# 핵심 함수
# ══════════════════════════════════════════════════════

def _build_delta_matrix(n: int, nu: int = NU) -> np.ndarray:
    """입력 차분 행렬 D (n·nu × n·nu): Δu[k] = u[k] - u[k-1], u[-1] = 0 가정.

    D = [ I           ]
        [-I  I         ]
        [    -I  I     ]
        ...

    Δu 페널티 적용: H_R = D.T @ R_blk @ D
    → u[N-1]이 0이 아닌 u[N-2] 방향으로 수렴 (터미널 롤오프 제거).
    """
    NNU = n * nu
    D = np.eye(NNU)
    for k in range(1, n):
        D[k * nu:(k + 1) * nu, (k - 1) * nu:k * nu] = -np.eye(nu)
    return D


def _make_QR(weights5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """weights5 = [long, lat, hdg, steer_r, accel_r] → Q (NX×NX), R (NU×NU).
    IACCEL is NOT included in Q (removed to eliminate terminal convergence bias).
    """
    Q = np.diag([weights5[0], weights5[1], weights5[2],
                 0., 0., 0., 0., 0.])
    R = np.diag([weights5[3], weights5[4]])
    return Q, R


def _linearize(x_op: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    동적 자전거 모델을 x_op에서 선형화 → ZOH 이산화 (A_d, B_d).

    저속 특이점: vx_lin = max(vx, V_MIN_LIN) 클램핑.
    """
    yaw = x_op[IYAW]
    vx  = max(x_op[IVX], V_MIN_LIN)
    vy  = x_op[IVY]
    yr  = x_op[IYR]

    A = np.zeros((NX, NX))
    B = np.zeros((NX, NU))

    cy, sy  = math.cos(yaw), math.sin(yaw)
    lf      = WB - L_CG
    km      = MASS * vx
    ki      = INERTIA * vx
    lf_cf   = lf * CF
    lr_cr   = L_CG * CR

    a00 = -2 * (CF + CR) / km
    a01 = -vx - 2 * (lf_cf - lr_cr) / km
    a10 = -2 * (lf_cf - lr_cr) / ki
    a11 = -2 * (lf * lf_cf + L_CG * lr_cr) / ki
    b00 = 2 * CF / MASS
    b10 = 2 * lf_cf / INERTIA

    v_rig_y = vy - yr * L_CG
    A[IX,  IYAW]   = -vx * sy - v_rig_y * cy
    A[IX,  IVX]    =  cy;  A[IX,  IVY] = -sy;  A[IX,  IYR] =  L_CG * sy
    A[IY,  IYAW]   =  vx * cy - v_rig_y * sy
    A[IY,  IVX]    =  sy;  A[IY,  IVY] =  cy;  A[IY,  IYR] = -L_CG * cy
    A[IYAW, IYR]   =  1.0
    A[IVX,  IACCEL] = 1.0
    A[IVY, IVY] = a00;  A[IVY, IYR] = a01;  A[IVY, ISTEER] = b00
    A[IYR, IVY] = a10;  A[IYR, IYR] = a11;  A[IYR, ISTEER] = b10
    A[ISTEER, ISTEER] = -1.0 / TAU_S;  B[ISTEER, 0] = 1.0 / TAU_S
    A[IACCEL, IACCEL] = -1.0 / TAU_A;  B[IACCEL, 1] = 1.0 / TAU_A

    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = A * DT
    M[:NX, NX:] = B * DT
    eM = scipy.linalg.expm(M)
    return eM[:NX, :NX], eM[:NX, NX:]


def _build_prediction_matrices(
    A_d: np.ndarray, B_d: np.ndarray, n: int = N
) -> tuple[np.ndarray, np.ndarray]:
    """
    Condensed prediction matrices (open-loop, 단일 linearization).

    x_{0..n} = S_x @ x0 + S_u @ U
      S_x: ((n+1)·NX, NX)
      S_u: ((n+1)·NX, n·NU)
    """
    S_x = np.zeros(((n + 1) * NX, NX))
    S_u = np.zeros(((n + 1) * NX, n * NU))
    Ap = np.eye(NX)
    for k in range(n + 1):
        S_x[k * NX:(k + 1) * NX] = Ap
        if k < n:
            Ap = A_d @ Ap
    for j in range(n):
        Psi = B_d.copy()
        for k in range(j, n):
            S_u[(k + 1) * NX:(k + 2) * NX, j * NU:(j + 1) * NU] = Psi
            Psi = A_d @ Psi
    return S_x, S_u


def _build_prediction_matrices_tv(
    A_list: list[np.ndarray],
    B_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """LTV Condensed prediction matrices (step-wise linearization).

    각 스텝 k에서 A_list[k], B_list[k]로 선형화된 예측 행렬 구성:
      x_{k+1} = A_list[k] @ x_k + B_list[k] @ u_k

    S_x[k] = Φ(k,0) = A[k-1]·…·A[0]
    S_u[k,j] = Φ(k, j+1) @ B[j]  (k > j, else 0)

    Args:
        A_list: 길이 n, 각 원소 (NX, NX)
        B_list: 길이 n, 각 원소 (NX, NU)

    Returns:
        S_x: ((n+1)·NX, NX)
        S_u: ((n+1)·NX, n·NU)
    """
    n = len(A_list)
    S_x = np.zeros(((n + 1) * NX, NX))
    S_u = np.zeros(((n + 1) * NX, n * NU))

    # S_x: cumulative state transition Φ(k, 0)
    Phi = np.eye(NX)
    S_x[:NX] = Phi
    for k in range(n):
        Phi = A_list[k] @ Phi
        S_x[(k + 1) * NX:(k + 2) * NX] = Phi

    # S_u: forced response  S_u[k+1, j] = A[k]·…·A[j+1] @ B[j]
    for j in range(n):
        Psi = B_list[j].copy()                                          # Φ(j+1,j+1)@B[j] = B[j]
        S_u[(j + 1) * NX:(j + 2) * NX, j * NU:(j + 1) * NU] = Psi
        for k in range(j + 1, n):
            Psi = A_list[k] @ Psi                                       # Φ(k+1,j+1)@B[j]
            S_u[(k + 1) * NX:(k + 2) * NX, j * NU:(j + 1) * NU] = Psi

    return S_x, S_u


def step_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """단일 DT 스텝 선형 동역학 적분 (ZOH).

    Args:
        x: (NX,) 현재 상태 [x, y, yaw, vx, vy, yr, steer, accel]
        u: (NU,) 제어 입력 [steer_cmd, accel_cmd]

    Returns:
        x_next: (NX,) 다음 상태 (DT 후)
    """
    u_clipped = np.clip(u, U_MIN, U_MAX)
    A_d, B_d  = _linearize(x)
    return A_d @ x + B_d @ u_clipped


def _solve_qp(
    H: np.ndarray, g: np.ndarray,
    lb: np.ndarray, ub: np.ndarray,
) -> np.ndarray:
    """
    min  0.5 u'Hu + g'u   s.t. lb ≤ u ≤ ub

    1차: numpy 직접 풀이 (비제약 해가 feasible이면 반환)
    2차: scipy L-BFGS-B (box constraints)
    """
    try:
        U_unc = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        U_unc = np.zeros(len(g))
    if np.all(U_unc >= lb - 1e-6) and np.all(U_unc <= ub + 1e-6):
        return np.clip(U_unc, lb, ub)
    res = scipy.optimize.minimize(
        lambda u: (0.5 * u @ H @ u + g @ u, H @ u + g),
        np.clip(U_unc, lb, ub),
        method="L-BFGS-B",
        jac=True,
        bounds=scipy.optimize.Bounds(lb, ub),
        options={"ftol": 1e-9, "gtol": 1e-6, "maxiter": 200},
    )
    return res.x


def compute_x0(
    speed: float,
    lon_accel: float,
    vel_global: np.ndarray,
    curv: float,
    quat: np.ndarray,
) -> np.ndarray:
    """h5 raw egomotion → MPC 초기 상태 벡터 x0 (NX=8).

    t0 시점의 로컬 프레임 (x=y=yaw=0) 기준으로 나머지 5개 상태를 복원한다.

    Args:
        speed:      ego_history_ego_states[-1, 2]  종방향 속도 [m/s]
        lon_accel:  ego_history_ego_states[-1, 4]  종방향 가속도 [m/s²]
        vel_global: ego_history_vel[-1]             (3,) 전역 좌표계 속도 [vx,vy,vz]
        curv:       ego_history_curv[-1, 0]         경로 곡률 [1/m]
        quat:       ego_history_quat_global[-1]     (4,) [qx,qy,qz,qw] 전역 자세

    Returns:
        x0: (NX=8,) [x=0, y=0, yaw=0, vx, vy, yaw_rate, steering, lon_accel]

    Notes:
        vy       — global vel을 body frame으로 회전 변환 후 y 성분
        yaw_rate — κ × vx  (경로 곡률 × 종방향 속도)
        steering — arctan(κ × WB)  (운동학적 자전거 모델 전륜 조향각)
    """
    vx = max(speed, V_MIN_LIN)

    # global frame → vehicle body frame
    vel_body = _Rotation.from_quat(quat).inv().apply(vel_global)
    vy = float(vel_body[1])

    yaw_rate = float(curv) * vx
    steering = float(np.arctan(float(curv) * WB))

    return np.array([0., 0., 0., vx, vy, yaw_rate, steering, float(lon_accel)])


def run_mpc(
    v0: float,
    gt_xyz: np.ndarray,
    gt_yaw: np.ndarray,
    weights5: np.ndarray,
    n: int = N,
    x0_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    가중치로 MPC 궤적 계산.

    Args:
        v0:       초기 속도 [m/s]  (x0_full 제공 시 무시)
        gt_xyz:   GT 미래 궤적 (≥n, 3)  [x, y, z] — 현재 차량 기준 local frame
        gt_yaw:   GT yaw (≥n,) [rad]
        weights5: [long, lat, hdg, steer_r, accel_r]
        n:        QP 호라이즌 (기본값 N=20)
        x0_full:  (NX,) 전체 초기 상태. 제공 시 x/y/yaw는 0으로 정규화됨.
                  pseudo-closed-loop에서 velocity/actuator 상태를 이어받을 때 사용.

    Returns:
        xy_pred    (n, 2)  예측 xy 궤적 [m]
        yaw_pred   (n,)    예측 yaw [rad]
        steer_pred (n,)    예측 실제 조향각 [rad]  (actuator state, U[:,0]이 아님)
        accel_pred (n,)    예측 실제 가속도 [m/s²] (actuator state, U[:,1]이 아님)
        U_opt      (n, 2)  최적 제어 입력 (steering_cmd, accel_cmd)
        ade        float   xy ADE [m]
    """
    NNU = n * NU
    if x0_full is not None:
        x0 = x0_full.copy()
        x0[IX] = x0[IY] = x0[IYAW] = 0.0  # position/heading은 호출자가 변환 완료
    else:
        x0 = np.array([0., 0., 0., max(v0, V_MIN_LIN), 0., 0., 0., 0.])
    A_d, B_d = _linearize(x0)
    S_x, S_u = _build_prediction_matrices(A_d, B_d, n=n)

    x_free = S_x @ x0
    x_ref  = np.zeros((n + 1, NX))
    for k in range(1, n + 1):
        x_ref[k, IX]   = gt_xyz[k - 1, 0]
        x_ref[k, IY]   = gt_xyz[k - 1, 1]
        x_ref[k, IYAW] = gt_yaw[k - 1]
    x_ref_flat = x_ref.flatten()

    Q, R = _make_QR(weights5)
    Q_blk = np.zeros(((n + 1) * NX, (n + 1) * NX))
    for k in range(n + 1):
        factor = TERMINAL_Q_FACTOR if k == n else 1.0
        Q_blk[k * NX:(k + 1) * NX, k * NX:(k + 1) * NX] = factor * Q
    R_blk = scipy.linalg.block_diag(*([R] * n))
    D     = _build_delta_matrix(n)

    H = S_u.T @ Q_blk @ S_u + D.T @ R_blk @ D  # Δu 페널티
    g = S_u.T @ Q_blk @ (x_free - x_ref_flat)
    H = 0.5 * (H + H.T) + QP_REG * np.eye(NNU)

    lb = np.tile(U_MIN, n)
    ub = np.tile(U_MAX, n)
    U_opt = _solve_qp(H, g, lb, ub)

    X_pred     = x_free + S_u @ U_opt
    states     = X_pred[NX:].reshape(n, NX)
    xy_pred    = states[:, [IX, IY]]
    yaw_pred   = states[:, IYAW]
    steer_pred = states[:, ISTEER]
    accel_pred = states[:, IACCEL]
    ade        = float(np.mean(np.linalg.norm(xy_pred - gt_xyz[:n, :2], axis=1)))
    return xy_pred, yaw_pred, steer_pred, accel_pred, U_opt.reshape(n, NU), ade
