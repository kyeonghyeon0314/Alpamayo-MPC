#!/usr/bin/env python3
"""
PCL (Pseudo-Closed-Loop) 시뮬레이션 실행기

build_pcl_dataset.py로 생성한 {clip_id}.h5 파일을 읽어
클립 전체에 걸쳐 Receding Horizon PCL 시뮬레이션을 수행하고,
MLP-PCL vs Default-PCL vs GT를 비교 시각화합니다.

────────────────────────────────────────────────────────────────
핵심 루프 (N_frames 스텝):
  i=0: GT 초기 위치에서 시작 (global_origin_xy[0] = [0,0])
  for i in 0 .. N_frames-1:
    1. 현재 시뮬 전역 위치 기록
    2. 전역 상태 → 프레임 i 로컬 좌표로 변환
    3. GT reference → 시뮬 차량 기준 로컬 프레임으로 재변환
    4. cotend[i] → MLP 가중치 예측
    5. MPC 풀기 (MLP / Default 각각, Receding Horizon 원리)
    6. 첫 번째 제어 입력만 적용 → 동역학 전진
    7. 다음 전역 상태 갱신

시각적 단절(Visual Disconnect) 경고:
  cotend[i]는 GT 경로 위에서 추출됨. 시뮬 차량이 GT에서 이탈해도
  이탈 위치의 hidden state는 재생성 불가 → 오프라인 평가의 고유 한계.
  논문/보고서 작성 시 반드시 명시할 것.

실행 예시:
  python closed-loop-simul/run_pcl_sim.py \\
      --pcl-clip  closed-loop-simul/pcl_clips/3f2d0388-3e97-43b7-9268-1d8987fc243e.h5 \\
      --model-dir closed-loop-simul/mlp_out \\
      --out       closed-loop-simul/pcl_sim_out
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mpc import (
    run_mpc, step_dynamics, WEIGHTS_DEFAULT,
    N, DT, NX, IX, IY, IYAW, IVX, V_MIN_LIN,
)
from model import load_mlp, CotendMLP

_W_LAT_FIXED = 1.0
_W_NAMES = ["long_pos", "lat_pos\n(fixed)", "heading", "steer_rate", "accel_rate"]


# ══════════════════════════════════════════════════════════════
# 좌표 변환 유틸리티
# ══════════════════════════════════════════════════════════════

def _R(yaw: float) -> np.ndarray:
    """2D 회전 행렬 R(yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _global_to_local(x_global: np.ndarray, origin_xy: np.ndarray, g_yaw: float) -> np.ndarray:
    """
    전역 상태 → 프레임 로컬 상태.

    위치/헤딩만 변환. 속도·actuator 상태(vx, vy, yr, steer, accel)는
    이미 차량 바디 프레임 기준이므로 좌표계 변환 불필요.
    """
    x_local = x_global.copy()
    R_inv = _R(g_yaw).T                              # R^{-1} = R^T
    p_local = R_inv @ (x_global[:2] - origin_xy)
    x_local[IX]   = p_local[0]
    x_local[IY]   = p_local[1]
    x_local[IYAW] = x_global[IYAW] - g_yaw
    return x_local


def _local_to_global(x_local: np.ndarray, origin_xy: np.ndarray, g_yaw: float) -> np.ndarray:
    """프레임 로컬 상태 → 전역 상태."""
    x_global = x_local.copy()
    p_global = _R(g_yaw) @ x_local[:2] + origin_xy
    x_global[IX]   = p_global[0]
    x_global[IY]   = p_global[1]
    x_global[IYAW] = x_local[IYAW] + g_yaw
    return x_global


def _gt_to_sim_frame(
    gt_xyz: np.ndarray, gt_yaw: np.ndarray,
    sx: float, sy: float, syaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GT 궤적을 프레임 i 로컬 → 시뮬 차량 기준 로컬로 재변환.

    gt_xyz: (M, 3)  — 프레임 i 로컬 좌표
    gt_yaw: (M,)    — 프레임 i 로컬 heading
    sx, sy, syaw    — 시뮬 차량의 프레임 i 로컬 위치·헤딩
    """
    R_inv = _R(syaw).T                               # R^{-1} = R^T
    xy_rel = (gt_xyz[:, :2] - np.array([sx, sy])) @ R_inv.T
    yaw_rel = gt_yaw - syaw
    xyz_rel = np.column_stack([xy_rel, np.zeros(len(xy_rel))])
    return xyz_rel, yaw_rel


def _pad_gt(gt_xyz: np.ndarray, gt_yaw: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """GT 배열을 MPC 호라이즌 n 스텝 이상으로 끝 값 패딩."""
    if len(gt_xyz) >= n:
        return gt_xyz[:n], gt_yaw[:n]
    pad = n - len(gt_xyz)
    gt_xyz = np.vstack([gt_xyz, np.tile(gt_xyz[-1], (pad, 1))])
    gt_yaw = np.concatenate([gt_yaw, np.tile(gt_yaw[-1], pad)])
    return gt_xyz, gt_yaw


def compute_valid_mask(gt_xyz_local: np.ndarray, min_var: float = 0.5) -> np.ndarray:
    """
    각 프레임의 GT future가 실질적인 변화를 가지는지 판별.

    GT future의 x(전방) 분산이 min_var 이하인 프레임은
    끝 값으로 패딩된 것으로 간주 → hidden state 유효하지 않음 (open-loop 구간).

    Returns:
        valid (N_frames,) bool — True: hidden state 유효, False: open-loop
    """
    # gt_xyz_local: (N_frames, 64, 3)
    x_var = np.var(gt_xyz_local[:, :, 0], axis=1)   # 전방 x 좌표 분산
    return x_var > min_var


# ══════════════════════════════════════════════════════════════
# MLP 추론
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict_weights(model: CotendMLP, cotend: np.ndarray, device: torch.device) -> np.ndarray:
    """cotend (4096,) → weights5 (5,)  [long, lat_fixed, hdg, steer_r, accel_r]."""
    x = torch.from_numpy(cotend.astype(np.float32)).unsqueeze(0).to(device)
    log4 = model(x).squeeze(0).cpu().numpy()
    w4 = np.exp(log4)
    return np.array([w4[0], _W_LAT_FIXED, w4[1], w4[2], w4[3]])


# ══════════════════════════════════════════════════════════════
# PCL 시뮬레이션 메인 루프
# ══════════════════════════════════════════════════════════════

def run_pcl_simulation(
    cotend:           np.ndarray,   # (N_frames, 4096)
    gt_xyz_local:     np.ndarray,   # (N_frames, 64, 3)  — 프레임 i 로컬
    gt_yaw_local:     np.ndarray,   # (N_frames, 64)     — 프레임 i 로컬
    v0_arr:           np.ndarray,   # (N_frames,)
    global_origin_xy: np.ndarray,   # (N_frames, 2)
    global_yaw:       np.ndarray,   # (N_frames,)
    model:            CotendMLP,
    device:           torch.device,
) -> dict:
    """
    Receding Horizon PCL 시뮬레이션.

    MLP 가중치 / Default 가중치 두 가지를 동시에 시뮬레이션한다.

    Returns:
        xy_mlp_global  (N_frames, 2) : MLP-PCL 시뮬 전역 위치
        xy_def_global  (N_frames, 2) : Def-PCL 시뮬 전역 위치
        xy_gt_global   (N_frames, 2) : GT 전역 위치 (= global_origin_xy)
        w_mlp_hist     (N_frames, 5) : 매 스텝 MLP 예측 가중치
        u_mlp_hist     (N_frames, 2) : 매 스텝 MLP 제어 입력
        u_def_hist     (N_frames, 2) : 매 스텝 Default 제어 입력
        errors_mlp     (N_frames,)   : GT 대비 MLP-PCL 위치 오차 [m]
        errors_def     (N_frames,)   : GT 대비 Def-PCL 위치 오차 [m]
    """
    N_frames = len(cotend)

    # ── 초기 상태: frame 0 전역 원점에서 출발 ──────────────
    # global_origin_xy[0] = [0, 0],  global_yaw[0] = 0  (build_pcl_dataset.py 기준)
    x_mlp_g = np.zeros(NX)
    x_mlp_g[IVX] = max(float(v0_arr[0]), V_MIN_LIN)
    x_def_g = x_mlp_g.copy()

    xy_mlp_global = np.zeros((N_frames, 2))
    xy_def_global = np.zeros((N_frames, 2))
    w_mlp_hist    = np.zeros((N_frames, 5))
    u_mlp_hist    = np.zeros((N_frames, 2))
    u_def_hist    = np.zeros((N_frames, 2))
    vx_mlp_hist   = np.zeros(N_frames)
    vx_def_hist   = np.zeros(N_frames)

    for i in range(N_frames):
        origin = global_origin_xy[i]
        g_yaw  = float(global_yaw[i])
        v0_i   = float(v0_arr[i])

        # ① 현재 전역 위치 및 속도 기록 (step i의 '시작')
        xy_mlp_global[i] = x_mlp_g[:2]
        xy_def_global[i] = x_def_g[:2]
        vx_mlp_hist[i]   = x_mlp_g[IVX]
        vx_def_hist[i]   = x_def_g[IVX]

        # ② 전역 상태 → 프레임 i 로컬 변환
        x_mlp_l = _global_to_local(x_mlp_g, origin, g_yaw)
        x_def_l = _global_to_local(x_def_g, origin, g_yaw)

        # ③ GT reference: 프레임 i 로컬 → 각 시뮬 차량의 로컬로 재변환
        #    MPC 호라이즌 N 스텝만 사용 (패딩 포함)
        gt_ref, gt_yaw_ref = _pad_gt(gt_xyz_local[i], gt_yaw_local[i], N)

        gt_sim_m, gt_yaw_sim_m = _gt_to_sim_frame(
            gt_ref, gt_yaw_ref,
            x_mlp_l[IX], x_mlp_l[IY], x_mlp_l[IYAW],
        )
        gt_sim_d, gt_yaw_sim_d = _gt_to_sim_frame(
            gt_ref, gt_yaw_ref,
            x_def_l[IX], x_def_l[IY], x_def_l[IYAW],
        )

        # ④ MLP 가중치 예측
        w_mlp = _predict_weights(model, cotend[i], device)
        w_mlp_hist[i] = w_mlp

        # ⑤ MPC 풀기 (MLP / Default)
        #    x0_full 제공 시 run_mpc 내부에서 position/heading을 0으로 정규화
        _, U_mlp, _ = run_mpc(v0_i, gt_sim_m, gt_yaw_sim_m, w_mlp,        x0_full=x_mlp_l)
        _, U_def, _ = run_mpc(v0_i, gt_sim_d, gt_yaw_sim_d, WEIGHTS_DEFAULT, x0_full=x_def_l)

        u_mlp = U_mlp[0]   # Receding Horizon: 첫 번째 입력만 적용
        u_def = U_def[0]
        u_mlp_hist[i] = u_mlp
        u_def_hist[i] = u_def

        # ⑥ 1-step 동역학 전진 (프레임 i 로컬 내)
        x_mlp_l_next = step_dynamics(x_mlp_l, u_mlp)
        x_def_l_next = step_dynamics(x_def_l, u_def)

        # ⑦ 다음 전역 상태 갱신
        x_mlp_g = _local_to_global(x_mlp_l_next, origin, g_yaw)
        x_def_g = _local_to_global(x_def_l_next, origin, g_yaw)

    # ── 지표 계산 ────────────────────────────────────────────
    xy_gt_global = global_origin_xy.copy()
    errors_mlp   = np.linalg.norm(xy_mlp_global - xy_gt_global, axis=1)
    errors_def   = np.linalg.norm(xy_def_global - xy_gt_global, axis=1)

    return {
        "xy_mlp_global": xy_mlp_global,
        "xy_def_global": xy_def_global,
        "xy_gt_global":  xy_gt_global,
        "w_mlp_hist":    w_mlp_hist,
        "u_mlp_hist":    u_mlp_hist,
        "u_def_hist":    u_def_hist,
        "vx_mlp_hist":   vx_mlp_hist,
        "vx_def_hist":   vx_def_hist,
        "errors_mlp":    errors_mlp,
        "errors_def":    errors_def,
    }


# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════

def plot_pcl_result(
    clip_id:      str,
    times_s:      np.ndarray,
    result:       dict,
    front_frames: np.ndarray | None,
    out_dir:      Path,
    v0_arr:       np.ndarray | None = None,
    valid_mask:   np.ndarray | None = None,
) -> Path:
    """PCL 시뮬레이션 결과 시각화 저장 (PNG)."""
    xy_mlp  = result["xy_mlp_global"]
    xy_def  = result["xy_def_global"]
    xy_gt   = result["xy_gt_global"]
    err_mlp = result["errors_mlp"]
    err_def = result["errors_def"]
    w_mlp   = result["w_mlp_hist"]
    u_mlp   = result["u_mlp_hist"]
    u_def   = result["u_def_hist"]

    # ── valid_mask: open-loop 구간 제외 ─────────────────────
    if valid_mask is None:
        valid_mask = np.ones(len(times_s), dtype=bool)
    # 유효 구간은 연속된 앞부분만 사용 (중간 구멍 무시)
    last_valid = int(np.where(valid_mask)[0][-1]) + 1 if valid_mask.any() else len(valid_mask)
    sl = slice(0, last_valid)   # 유효 슬라이스

    ade_mlp  = float(np.mean(err_mlp[1:last_valid]))   # step 0는 항상 0 — 제외
    ade_def  = float(np.mean(err_def[1:last_valid]))
    fde_mlp  = float(err_mlp[last_valid - 1])
    fde_def  = float(err_def[last_valid - 1])
    gain_pct = (ade_def - ade_mlp) / ade_def * 100 if ade_def > 0 else 0.0

    has_frames = front_frames is not None and len(front_frames) > 0

    # ─── 레이아웃 구성 ───────────────────────────────────────
    n_rows      = (5 if has_frames else 4)
    row_heights = ([0.9] if has_frames else []) + [1.3, 1.0, 1.0, 2.8]
    fig = plt.figure(figsize=(22, 4.5 * n_rows))
    fig.suptitle(
        f"PCL Simulation  |  {clip_id}\n"
        f"MLP-PCL  ADE={ade_mlp:.3f}m  FDE={fde_mlp:.3f}m    "
        f"Def-PCL  ADE={ade_def:.3f}m  FDE={fde_def:.3f}m    "
        f"impr={gain_pct:+.1f}%    "
        f"clip duration {times_s[-1] - times_s[0]:.1f}s  ({len(times_s)} frames)",
        fontsize=10, y=0.995,
    )
    outer = gridspec.GridSpec(n_rows, 1, figure=fig,
                              height_ratios=row_heights, hspace=0.55)
    row = 0

    # ── 전방 카메라 그리드 ────────────────────────────────────
    if has_frames:
        N_CAM   = min(10, len(front_frames))
        indices = np.linspace(0, len(front_frames) - 1, N_CAM, dtype=int)
        cmap_t  = plt.cm.plasma(np.linspace(0.1, 0.95, N_CAM))
        gs_cam  = gridspec.GridSpecFromSubplotSpec(1, N_CAM, subplot_spec=outer[row], wspace=0.03)
        for col, (idx, color) in enumerate(zip(indices, cmap_t)):
            ax = fig.add_subplot(gs_cam[col])
            ax.imshow(front_frames[idx])
            for spine in ax.spines.values():
                spine.set_edgecolor(color[:3])
                spine.set_linewidth(3)
            ax.set_title(f"t={times_s[idx]:.1f}s", fontsize=7, pad=2)
            ax.axis("off")
        row += 1

    # ── 오차 타임라인 + ADE 요약 바 ──────────────────────────
    gs_err = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[row], wspace=0.32, width_ratios=[2.8, 1]
    )
    t_valid  = times_s[1:last_valid]
    em_valid = err_mlp[1:last_valid]
    ed_valid = err_def[1:last_valid]

    ax_err = fig.add_subplot(gs_err[0])
    ax_err.plot(t_valid, em_valid, "-",  color="mediumpurple", lw=2.2,
                label=f"MLP-PCL  ADE={ade_mlp:.3f}m  FDE={fde_mlp:.3f}m")
    ax_err.plot(t_valid, ed_valid, "--", color="tomato",       lw=2.0,
                label=f"Def-PCL   ADE={ade_def:.3f}m  FDE={fde_def:.3f}m")
    ax_err.fill_between(
        t_valid, em_valid, ed_valid,
        where=(em_valid < ed_valid),
        alpha=0.15, color="mediumpurple", label="MLP better",
    )
    ax_err.fill_between(
        t_valid, em_valid, ed_valid,
        where=(em_valid > ed_valid),
        alpha=0.12, color="tomato", label="Default better",
    )
    ax_err.set_xlabel("Clip time [s]")
    ax_err.set_ylabel("Position error vs GT [m]")
    ax_err.set_title("PCL Position Error Timeline  (lower = better, flat = no divergence)")
    ax_err.legend(fontsize=8, loc="upper left")
    ax_err.grid(True, ls=":", lw=0.5)
    ax_err.set_ylim(bottom=0)

    ax_bar = fig.add_subplot(gs_err[1])
    bars = ax_bar.bar(
        ["MLP-PCL", "Def-PCL"],
        [ade_mlp, ade_def],
        color=["mediumpurple", "tomato"],
        alpha=0.85, edgecolor="white", width=0.5,
    )
    for bar, val in zip(bars, [ade_mlp, ade_def]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01, f"{val:.3f}m",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax_bar.set_ylabel("Mean ADE [m]  (step 0 excluded)")
    ax_bar.set_title(f"ADE Summary\nimpr={gain_pct:+.1f}%")
    ax_bar.grid(True, axis="y", ls=":", lw=0.5)
    row += 1

    # ── MLP 가중치 진화 ──────────────────────────────────────
    gs_w = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[row], wspace=0.38)
    for k, (wname, w_def_val) in enumerate(zip(_W_NAMES, WEIGHTS_DEFAULT)):
        ax_w = fig.add_subplot(gs_w[k])
        ax_w.plot(times_s, w_mlp[:, k], "-", color="mediumpurple", lw=1.5, label="MLP")
        ax_w.axhline(w_def_val, color="tomato", ls="--", lw=1.2,
                     label=f"def={w_def_val:.2f}")
        ax_w.set_title(wname, fontsize=8)
        ax_w.set_xlabel("t [s]", fontsize=7)
        ax_w.legend(fontsize=6, loc="best")
        ax_w.grid(True, ls=":", lw=0.5)
    row += 1

    # ── 제어 입력 (조향 / 가속) ──────────────────────────────
    gs_u = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[row], wspace=0.35)

    ax_steer = fig.add_subplot(gs_u[0])
    ax_steer.plot(times_s[sl], u_mlp[sl, 0], "-",  color="mediumpurple", lw=1.5, label="MLP")
    ax_steer.plot(times_s[sl], u_def[sl, 0], "--", color="tomato",       lw=1.5, label="Def")
    ax_steer.axhline(0, c="k", lw=0.5)
    ax_steer.set_xlabel("t [s]"); ax_steer.set_ylabel("Steering cmd [rad]")
    ax_steer.set_title("Steering Control Input")
    ax_steer.legend(fontsize=8); ax_steer.grid(True, ls=":", lw=0.5)

    ax_accel = fig.add_subplot(gs_u[1])
    ax_accel.plot(times_s[sl], u_mlp[sl, 1], "-",  color="mediumpurple", lw=1.5, label="MLP")
    ax_accel.plot(times_s[sl], u_def[sl, 1], "--", color="tomato",       lw=1.5, label="Def")
    ax_accel.axhline(0, c="k", lw=0.5)
    ax_accel.set_xlabel("t [s]"); ax_accel.set_ylabel("Accel cmd [m/s²]")
    ax_accel.set_title("Acceleration Control Input")
    ax_accel.legend(fontsize=8); ax_accel.grid(True, ls=":", lw=0.5)

    vx_mlp = result["vx_mlp_hist"]
    vx_def = result["vx_def_hist"]
    ax_speed = fig.add_subplot(gs_u[2])
    if v0_arr is not None:
        ax_speed.plot(times_s[sl], v0_arr[sl], "-", color="gray", lw=1.5,
                      alpha=0.6, label="GT v0", zorder=3)
    ax_speed.plot(times_s[sl], vx_mlp[sl], "-",  color="mediumpurple", lw=1.5, label="MLP")
    ax_speed.plot(times_s[sl], vx_def[sl], "--", color="tomato",       lw=1.5, label="Def")
    ax_speed.set_xlabel("t [s]"); ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_title("Vehicle Speed")
    ax_speed.legend(fontsize=8); ax_speed.grid(True, ls=":", lw=0.5)

    row += 1

    # ── 전역 궤적 (BEV) ──────────────────────────────────────
    ax_traj = fig.add_subplot(outer[row])

    # GT — 얇고 반투명하게, 기준선 역할만
    ax_traj.plot(xy_gt[sl, 1],  xy_gt[sl, 0],  "-",
                 color="#888888", lw=1.2, alpha=0.5, zorder=3,
                 label="GT (ground truth)")

    # MLP-PCL
    ax_traj.plot(xy_mlp[sl, 1], xy_mlp[sl, 0], "-",
                 color="mediumpurple", lw=2.2, zorder=5,
                 label=f"MLP-PCL  ADE={ade_mlp:.3f}m  FDE={fde_mlp:.3f}m")

    # Default-PCL
    ax_traj.plot(xy_def[sl, 1], xy_def[sl, 0], "--",
                 color="tomato", lw=2.0, alpha=0.85, zorder=4,
                 label=f"Def-PCL   ADE={ade_def:.3f}m  FDE={fde_def:.3f}m")

    # 시작 / 끝 마커
    ax_traj.scatter([xy_gt[0, 1]], [xy_gt[0, 0]],
                    c="black", s=160, zorder=10,
                    marker="o", edgecolors="white", linewidths=2.0, label="Start (common)")
    for xy_arr, color, lbl in [
        (xy_gt[sl],   "black",        "GT end"),
        (xy_mlp[sl],  "mediumpurple", "MLP end"),
        (xy_def[sl],  "tomato",       "Def end"),
    ]:
        ax_traj.scatter([xy_arr[-1, 1]], [xy_arr[-1, 0]],
                        c=color, s=150, zorder=10,
                        marker="*", edgecolors="white", linewidths=1.5, label=lbl)

    ax_traj.set_xlabel("Y / lateral [m]   (<- left | right ->)")
    ax_traj.set_ylabel("X / longitudinal [m]")
    ax_traj.set_title(
        "Global Trajectory (BEV)   "
        "dark=GT  |  purple solid=MLP-PCL  |  red dashed=Def-PCL\n"
        f"star=clip end position   impr={gain_pct:+.1f}%"
        + (f"   [!] MLP diverge" if fde_mlp > 10.0 else "   [ok] MLP stable")
        + (f"   [!] Def diverge" if fde_def > 10.0 else "   [ok] Def stable"),
    )
    ax_traj.invert_xaxis()
    ax_traj.legend(fontsize=8, loc="best", ncol=2)
    ax_traj.grid(True, ls=":", lw=0.5)
    ax_traj.set_aspect("equal", adjustable="datalim")

    # ── 저장 ─────────────────────────────────────────────────
    out_path = out_dir / f"{clip_id[:16]}_pcl_sim.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════
# 발산 판정
# ══════════════════════════════════════════════════════════════

def _divergence_report(errors: np.ndarray, name: str,
                       warn_m: float = 5.0, fail_m: float = 10.0) -> str:
    """단순 발산 판정 문자열 반환."""
    fde = float(errors[-1])
    max_e = float(np.max(errors[1:]))
    if fde > fail_m or max_e > fail_m:
        status = "⚠  DIVERGED"
    elif fde > warn_m or max_e > warn_m:
        status = "△  MARGINAL"
    else:
        status = "✓  STABLE  "
    return f"  {name:12s}: {status}  (FDE={fde:.3f}m  max={max_e:.3f}m)"


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="PCL 시뮬레이션 — build_pcl_dataset.py 출력 h5 파일 사용",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--pcl-clip",  required=True,
                   help="PCL 데이터셋 h5 파일 경로")
    p.add_argument("--model-dir", required=True,
                   help="학습된 MLP 모델 디렉토리 (model_full.pt 또는 best_model.pt)")
    p.add_argument("--out",       required=True,
                   help="출력 디렉토리 (PNG + npz 저장)")
    p.add_argument("--device",    default="cpu",
                   help="추론 장치 (cpu / cuda, default: cpu)")
    p.add_argument("--no-frames", action="store_true",
                   help="전방 카메라 패널 제외 (메모리 절약)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, _ = load_mlp(Path(args.model_dir), device)

    # ── h5 로드 ───────────────────────────────────────────────
    pcl_path = Path(args.pcl_clip)
    print(f"\n{'='*60}")
    print(f"PCL 클립 로드: {pcl_path.name}")

    with h5py.File(pcl_path, "r") as f:
        clip_id          = str(f.attrs["clip_id"])
        n_frames         = int(f.attrs["n_frames"])
        step_us          = int(f.attrs["step_us"])
        t_start          = float(f.attrs["t_start_s"])
        t_end            = float(f.attrs["t_end_s"])

        times_s          = f["times_s"][:]
        v0_arr           = f["v0"][:]
        cotend           = f["cotend"][:]
        gt_xyz_local     = f["gt_xyz_local"][:]
        gt_yaw_local     = f["gt_yaw_local"][:]
        global_origin_xy = f["global_origin_xy"][:]
        global_yaw       = f["global_yaw"][:]
        front_frames     = None if args.no_frames else f["front_frames"][:]

    print(f"  clip_id         : {clip_id}")
    print(f"  n_frames        : {n_frames}  ({t_end - t_start:.1f}s @ {step_us // 1000}ms)")
    print(f"  cotend shape    : {cotend.shape}")
    print(f"  gt_xyz_local    : {gt_xyz_local.shape}")
    print(f"  global_origin_xy: {global_origin_xy.shape}")

    # ── 시뮬레이션 ────────────────────────────────────────────
    print(f"\nPCL 시뮬레이션 실행 중  ({n_frames} steps) ...")
    result = run_pcl_simulation(
        cotend=cotend,
        gt_xyz_local=gt_xyz_local,
        gt_yaw_local=gt_yaw_local,
        v0_arr=v0_arr,
        global_origin_xy=global_origin_xy,
        global_yaw=global_yaw,
        model=model,
        device=device,
    )
    print("완료.")

    # ── 결과 요약 ─────────────────────────────────────────────
    err_mlp = result["errors_mlp"]
    err_def = result["errors_def"]
    ade_mlp = float(np.mean(err_mlp[1:]))
    ade_def = float(np.mean(err_def[1:]))
    gain    = (ade_def - ade_mlp) / ade_def * 100 if ade_def > 0 else 0.0

    print(f"\n{'─'*60}")
    print(f"  MLP-PCL ADE : {ade_mlp:.4f} m")
    print(f"  Def-PCL ADE : {ade_def:.4f} m")
    print(f"  개선율      : {gain:+.2f}%")
    print(f"\n발산 판정 (경고>5m / 발산>10m):")
    print(_divergence_report(err_mlp, "MLP-PCL"))
    print(_divergence_report(err_def, "Def-PCL"))
    print(f"{'─'*60}")

    # ── valid_mask 계산 (open-loop 구간 탐지) ────────────────
    valid_mask = compute_valid_mask(gt_xyz_local)
    n_valid = int(valid_mask.sum())
    n_invalid = len(valid_mask) - n_valid
    if n_invalid > 0:
        print(f"  open-loop frames: {n_invalid} (last {n_invalid * step_us // 1000}ms excluded from plot)")

    # ── PNG 저장 ──────────────────────────────────────────────
    png_path = plot_pcl_result(
        clip_id, times_s, result, front_frames, out_dir,
        v0_arr=v0_arr, valid_mask=valid_mask,
    )
    print(f"\n시각화 저장: {png_path}")

    # ── npz 저장 (추가 분석용) ────────────────────────────────
    npz_path = out_dir / f"{clip_id[:16]}_pcl_sim.npz"
    np.savez(
        npz_path,
        times_s          = times_s,
        xy_mlp_global    = result["xy_mlp_global"],
        xy_def_global    = result["xy_def_global"],
        xy_gt_global     = result["xy_gt_global"],
        errors_mlp       = err_mlp,
        errors_def       = err_def,
        w_mlp_hist       = result["w_mlp_hist"],
        u_mlp_hist       = result["u_mlp_hist"],
        u_def_hist       = result["u_def_hist"],
    )
    print(f"수치 결과 저장: {npz_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
