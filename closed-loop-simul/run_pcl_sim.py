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

sys.path.insert(0, str(Path(__file__).parent.parent / "alpamayo_dataset"))
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
    ema_alpha:        float = 1.0,  # EMA 평활화 계수 (1.0=비활성, 0<α<1=평활화)
) -> dict:
    """
    Receding Horizon PCL 시뮬레이션.

    MLP 가중치 / Default 가중치 두 가지를 동시에 시뮬레이션한다.

    ema_alpha: EMA 계수. w_ema = alpha * w_raw + (1-alpha) * w_prev
               1.0이면 평활화 없음 (원래 동작). 낮을수록 강한 평활화.

    Returns:
        xy_mlp_global  (N_frames, 2) : MLP-PCL 시뮬 전역 위치
        xy_def_global  (N_frames, 2) : Def-PCL 시뮬 전역 위치
        xy_gt_global   (N_frames, 2) : GT 전역 위치 (= global_origin_xy)
        w_mlp_hist     (N_frames, 5) : 매 스텝 EMA 평활화된 MLP 가중치 (MPC 실입력)
        w_raw_hist     (N_frames, 5) : 매 스텝 MLP 원본 예측 가중치
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
    w_mlp_hist    = np.zeros((N_frames, 5))   # EMA 평활화된 가중치 (실제 MPC 입력)
    w_raw_hist    = np.zeros((N_frames, 5))   # MLP 원본 예측 가중치
    u_mlp_hist    = np.zeros((N_frames, 2))
    u_def_hist    = np.zeros((N_frames, 2))
    vx_mlp_hist   = np.zeros(N_frames)
    vx_def_hist   = np.zeros(N_frames)

    w_ema: np.ndarray | None = None  # EMA 상태 (첫 프레임은 raw 값으로 초기화)

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

        # ④ MLP 가중치 예측 + EMA 평활화
        w_raw = _predict_weights(model, cotend[i], device)
        if w_ema is None:
            w_ema = w_raw.copy()            # 첫 프레임: EMA 상태 초기화
        else:
            w_ema = ema_alpha * w_raw + (1.0 - ema_alpha) * w_ema
        w_raw_hist[i]  = w_raw
        w_mlp_hist[i]  = w_ema             # MPC에는 평활화된 가중치 사용
        w_mlp = w_ema

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
        "w_mlp_hist":    w_mlp_hist,    # EMA 평활화된 가중치 (실제 MPC 입력)
        "w_raw_hist":    w_raw_hist,    # MLP 원본 예측 가중치
        "u_mlp_hist":    u_mlp_hist,
        "u_def_hist":    u_def_hist,
        "vx_mlp_hist":   vx_mlp_hist,
        "vx_def_hist":   vx_def_hist,
        "errors_mlp":    errors_mlp,
        "errors_def":    errors_def,
    }


# ══════════════════════════════════════════════════════════════
# 시각화 유틸리티
# ══════════════════════════════════════════════════════════════

def _valid_slice(valid_mask: np.ndarray | None, n: int) -> slice:
    """유효 구간 슬라이스 반환 (open-loop 테일 제거)."""
    if valid_mask is None or not valid_mask.any():
        return slice(0, n)
    return slice(0, int(np.where(valid_mask)[0][-1]) + 1)


# ══════════════════════════════════════════════════════════════
# Fig 1 — 오차 타임라인 + BEV 궤적
# ══════════════════════════════════════════════════════════════

def plot_fig1_overview(
    clip_id:    str,
    times_s:    np.ndarray,
    result:     dict,
    out_dir:    Path,
    valid_mask: np.ndarray | None = None,
) -> Path:
    """Fig 1: 오차 타임라인 + ADE/FDE 바 + BEV 궤적."""
    xy_mlp  = result["xy_mlp_global"]
    xy_def  = result["xy_def_global"]
    xy_gt   = result["xy_gt_global"]
    err_mlp = result["errors_mlp"]
    err_def = result["errors_def"]

    sl         = _valid_slice(valid_mask, len(times_s))
    last_valid = sl.stop
    ade_mlp    = float(np.mean(err_mlp[1:last_valid]))
    ade_def    = float(np.mean(err_def[1:last_valid]))
    fde_mlp    = float(err_mlp[last_valid - 1])
    fde_def    = float(err_def[last_valid - 1])
    gain_pct   = (ade_def - ade_mlp) / ade_def * 100 if ade_def > 0 else 0.0

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"[Fig 1] PCL Overview  |  {clip_id}\n"
        f"MLP  ADE={ade_mlp:.3f}m  FDE={fde_mlp:.3f}m    "
        f"Default  ADE={ade_def:.3f}m  FDE={fde_def:.3f}m    "
        f"impr={gain_pct:+.1f}%    "
        f"clip {times_s[-1] - times_s[0]:.1f}s ({len(times_s)} frames)",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           height_ratios=[1, 1.6])

    t_valid  = times_s[1:last_valid]
    em_valid = err_mlp[1:last_valid]
    ed_valid = err_def[1:last_valid]

    # ── 오차 타임라인 ──────────────────────────────────────────
    ax_err = fig.add_subplot(gs[0, 0])
    ax_err.plot(t_valid, em_valid, "-",  color="mediumpurple", lw=2.2,
                label=f"MLP-PCL  ADE={ade_mlp:.3f}m")
    ax_err.plot(t_valid, ed_valid, "--", color="tomato", lw=2.0,
                label=f"Def-PCL  ADE={ade_def:.3f}m")
    ax_err.fill_between(t_valid, em_valid, ed_valid,
                        where=(em_valid < ed_valid), alpha=0.15,
                        color="mediumpurple", label="MLP better")
    ax_err.fill_between(t_valid, em_valid, ed_valid,
                        where=(em_valid > ed_valid), alpha=0.12,
                        color="tomato", label="Default better")
    ax_err.set_xlabel("Clip time [s]", fontsize=11)
    ax_err.set_ylabel("Position error vs GT [m]", fontsize=11)
    ax_err.set_title("Position Error Timeline  (lower = better)", fontsize=12)
    ax_err.legend(fontsize=10)
    ax_err.grid(True, ls=":", lw=0.5)
    ax_err.set_ylim(bottom=0)

    # ── ADE / FDE 요약 바 (그룹형) ────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    bw = 0.32
    b_ade = ax_bar.bar(x - bw / 2, [ade_mlp, ade_def], bw,
                       color=["mediumpurple", "tomato"], alpha=0.85,
                       edgecolor="white", label="ADE")
    b_fde = ax_bar.bar(x + bw / 2, [fde_mlp, fde_def], bw,
                       color=["mediumpurple", "tomato"], alpha=0.5,
                       edgecolor="white", hatch="//", label="FDE")
    for bar, val in zip(list(b_ade) + list(b_fde),
                        [ade_mlp, ade_def, fde_mlp, fde_def]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["MLP-PCL", "Def-PCL"], fontsize=11)
    ax_bar.set_ylabel("Error [m]", fontsize=11)
    ax_bar.set_title(f"ADE / FDE Summary  (impr={gain_pct:+.1f}%)", fontsize=12)
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, axis="y", ls=":", lw=0.5)

    # ── BEV 궤적 ──────────────────────────────────────────────
    ax_traj = fig.add_subplot(gs[1, :])
    ax_traj.plot(xy_gt[sl, 1],  xy_gt[sl, 0],  "-",
                 color="#888888", lw=1.2, alpha=0.6, zorder=3,
                 label="GT (ground truth)")
    ax_traj.plot(xy_mlp[sl, 1], xy_mlp[sl, 0], "-",
                 color="mediumpurple", lw=2.5, zorder=5,
                 label=f"MLP-PCL  ADE={ade_mlp:.3f}m  FDE={fde_mlp:.3f}m")
    ax_traj.plot(xy_def[sl, 1], xy_def[sl, 0], "--",
                 color="tomato", lw=2.2, alpha=0.85, zorder=4,
                 label=f"Def-PCL  ADE={ade_def:.3f}m  FDE={fde_def:.3f}m")
    ax_traj.scatter([xy_gt[0, 1]], [xy_gt[0, 0]],
                    c="black", s=200, zorder=10, marker="o",
                    edgecolors="white", linewidths=2.5, label="Start")
    # GT end — 작은 속빈 다이아몬드 (MLP/Def end 아래 묻히지 않도록 zorder 높임)
    ax_traj.scatter([xy_gt[sl][-1, 1]], [xy_gt[sl][-1, 0]],
                    s=70, zorder=12, marker="D",
                    facecolors="none", edgecolors="black", linewidths=2.0,
                    label="GT end")
    # MLP / Def end — 큰 별 마커
    for xy_arr, clr, lbl in [
        (xy_mlp[sl], "mediumpurple", "MLP end"),
        (xy_def[sl], "tomato",       "Def end"),
    ]:
        ax_traj.scatter([xy_arr[-1, 1]], [xy_arr[-1, 0]],
                        c=clr, s=200, zorder=11, marker="*",
                        edgecolors="white", linewidths=2.0, label=lbl)
    status_mlp = "[!] MLP diverged" if fde_mlp > 10.0 else "[ok] MLP stable"
    status_def = "[!] Def diverged" if fde_def > 10.0 else "[ok] Def stable"
    ax_traj.set_xlabel("Y / lateral [m]  (<- left | right ->)", fontsize=11)
    ax_traj.set_ylabel("X / longitudinal [m]", fontsize=11)
    ax_traj.set_title(
        f"Global Trajectory (BEV)  —  {status_mlp}   {status_def}", fontsize=12)
    ax_traj.invert_xaxis()
    ax_traj.legend(fontsize=10, loc="best", ncol=3)
    ax_traj.grid(True, ls=":", lw=0.5)
    ax_traj.set_aspect("equal", adjustable="datalim")

    out_path = out_dir / f"{clip_id[:16]}_fig1_overview.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════
# Fig 2 — MLP 가중치 진화 (2×3 크게)
# ══════════════════════════════════════════════════════════════

def plot_fig2_weights(
    clip_id: str,
    times_s: np.ndarray,
    result:  dict,
    out_dir: Path,
) -> Path:
    """Fig 2: MLP 가중치 5개 진화 + Default 비교 바."""
    w_mlp = result["w_mlp_hist"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f"[Fig 2] MLP Weight Evolution  |  {clip_id}", fontsize=13)

    for k, (wname, w_def_val) in enumerate(zip(_W_NAMES, WEIGHTS_DEFAULT)):
        row, col = divmod(k, 3)
        ax    = axes[row, col]
        wvals = w_mlp[:, k]
        mu, sig = float(np.mean(wvals)), float(np.std(wvals))

        ax.plot(times_s, wvals, "-", color="mediumpurple", lw=2.0, label="MLP predicted")
        ax.axhline(w_def_val, color="tomato",    ls="--", lw=1.8, label=f"Default = {w_def_val:.2f}")
        ax.axhline(mu,        color="steelblue", ls=":",  lw=1.5, label=f"MLP mean = {mu:.2f}")
        ax.fill_between(times_s, mu - sig, mu + sig, alpha=0.12, color="mediumpurple")
        ax.set_title(wname.replace("\n", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel("t [s]", fontsize=10)
        ax.set_ylabel("Weight value", fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, ls=":", lw=0.5)
        ax.text(
            0.97, 0.04,
            f"μ={mu:.3f}  σ={sig:.3f}\nmin={wvals.min():.3f}  max={wvals.max():.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray",
            bbox=dict(fc="white", ec="lightgray", alpha=0.7, pad=2),
        )

    # 6번째 패널: MLP 평균 vs Default 그룹 바
    ax_sum = axes[1, 2]
    short_names = ["long_pos", "lat_pos*", "heading", "steer_r", "accel_r"]
    mlp_means = [float(np.mean(w_mlp[:, k])) for k in range(5)]
    x = np.arange(5)
    bw = 0.35
    ax_sum.bar(x - bw / 2, mlp_means,          bw, color="mediumpurple",
               alpha=0.85, edgecolor="white", label="MLP mean")
    ax_sum.bar(x + bw / 2, list(WEIGHTS_DEFAULT), bw, color="tomato",
               alpha=0.75, edgecolor="white", label="Default")
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(short_names, fontsize=9, rotation=15)
    ax_sum.set_ylabel("Weight value", fontsize=10)
    ax_sum.set_title("MLP mean vs Default (all weights)", fontsize=12, fontweight="bold")
    ax_sum.legend(fontsize=9)
    ax_sum.grid(True, axis="y", ls=":", lw=0.5)
    ax_sum.text(0.5, -0.20, "* lat_pos is fixed = 1.0 (not predicted)",
                transform=ax_sum.transAxes, ha="center", fontsize=8, color="gray")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / f"{clip_id[:16]}_fig2_weights.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════
# Fig 3 — 제어 입력 분석 (명령값 + 안정성 지표)
# ══════════════════════════════════════════════════════════════

def plot_fig3_control(
    clip_id:    str,
    times_s:    np.ndarray,
    result:     dict,
    out_dir:    Path,
    v0_arr:     np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> Path:
    """Fig 3: 제어 명령 + Steering Rate / Jerk / 누적 에너지."""
    u_mlp  = result["u_mlp_hist"]
    u_def  = result["u_def_hist"]
    vx_mlp = result["vx_mlp_hist"]
    vx_def = result["vx_def_hist"]

    sl = _valid_slice(valid_mask, len(times_s))
    t  = times_s[sl]
    um = u_mlp[sl]
    ud = u_def[sl]

    steer_rate_mlp = np.diff(um[:, 0], prepend=um[0, 0])
    steer_rate_def = np.diff(ud[:, 0], prepend=ud[0, 0])
    jerk_mlp       = np.diff(um[:, 1], prepend=um[0, 1])
    jerk_def       = np.diff(ud[:, 1], prepend=ud[0, 1])
    effort_mlp     = np.cumsum(um[:, 0] ** 2 + um[:, 1] ** 2)
    effort_def     = np.cumsum(ud[:, 0] ** 2 + ud[:, 1] ** 2)

    sr_rms_mlp = float(np.sqrt(np.mean(steer_rate_mlp[1:] ** 2)))
    sr_rms_def = float(np.sqrt(np.mean(steer_rate_def[1:] ** 2)))
    jk_rms_mlp = float(np.sqrt(np.mean(jerk_mlp[1:] ** 2)))
    jk_rms_def = float(np.sqrt(np.mean(jerk_def[1:] ** 2)))

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"[Fig 3] Control Input Analysis  |  {clip_id}\n"
        f"Steering Rate RMS  MLP={sr_rms_mlp:.4f}  Def={sr_rms_def:.4f}  |  "
        f"Jerk RMS  MLP={jk_rms_mlp:.4f}  Def={jk_rms_def:.4f}",
        fontsize=12,
    )

    # [0,0] 조향 명령
    ax = axes[0, 0]
    ax.plot(t, um[:, 0], "-",  color="mediumpurple", lw=2.0, label="MLP")
    ax.plot(t, ud[:, 0], "--", color="tomato",       lw=1.8, label="Default")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("t [s]", fontsize=10); ax.set_ylabel("[rad]", fontsize=10)
    ax.set_title("Steering Command", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, ls=":", lw=0.5)

    # [0,1] 가속 명령
    ax = axes[0, 1]
    ax.plot(t, um[:, 1], "-",  color="mediumpurple", lw=2.0, label="MLP")
    ax.plot(t, ud[:, 1], "--", color="tomato",       lw=1.8, label="Default")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("t [s]", fontsize=10); ax.set_ylabel("[m/s²]", fontsize=10)
    ax.set_title("Acceleration Command", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, ls=":", lw=0.5)

    # [0,2] 속도
    ax = axes[0, 2]
    if v0_arr is not None:
        ax.plot(t, v0_arr[sl], "-", color="gray", lw=1.8,
                alpha=0.7, label="GT v0", zorder=3)
    ax.plot(t, vx_mlp[sl], "-",  color="mediumpurple", lw=2.0, label="MLP")
    ax.plot(t, vx_def[sl], "--", color="tomato",       lw=1.8, label="Default")
    ax.set_xlabel("t [s]", fontsize=10); ax.set_ylabel("[m/s]", fontsize=10)
    ax.set_title("Vehicle Speed", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, ls=":", lw=0.5)

    # [1,0] 조향 Rate — Jitter 지표
    ax = axes[1, 0]
    ax.plot(t, steer_rate_mlp, "-",  color="mediumpurple", lw=1.8,
            label=f"MLP  RMS={sr_rms_mlp:.4f}")
    ax.plot(t, steer_rate_def, "--", color="tomato",       lw=1.6,
            label=f"Def  RMS={sr_rms_def:.4f}")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("t [s]", fontsize=10); ax.set_ylabel("[rad/step]", fontsize=10)
    ax.set_title("Steering Rate  (jitter indicator, lower = smoother)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, ls=":", lw=0.5)

    # [1,1] Jerk — 승차감 지표
    ax = axes[1, 1]
    ax.plot(t, jerk_mlp, "-",  color="mediumpurple", lw=1.8,
            label=f"MLP  RMS={jk_rms_mlp:.4f}")
    ax.plot(t, jerk_def, "--", color="tomato",       lw=1.6,
            label=f"Def  RMS={jk_rms_def:.4f}")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("t [s]", fontsize=10); ax.set_ylabel("[(m/s²)/step]", fontsize=10)
    ax.set_title("Jerk  (ride comfort indicator, lower = smoother)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, ls=":", lw=0.5)

    # [1,2] 누적 제어 에너지
    ax = axes[1, 2]
    ax.plot(t, effort_mlp, "-",  color="mediumpurple", lw=2.0,
            label=f"MLP  total={effort_mlp[-1]:.2f}")
    ax.plot(t, effort_def, "--", color="tomato",       lw=1.8,
            label=f"Def  total={effort_def[-1]:.2f}")
    ax.set_xlabel("t [s]", fontsize=10); ax.set_ylabel("Cumulative Σu²", fontsize=10)
    ax.set_title("Cumulative Control Effort  (lower = efficient)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, ls=":", lw=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / f"{clip_id[:16]}_fig3_control.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════
# Fig 4 — 전방 카메라 스트립 (옵션)
# ══════════════════════════════════════════════════════════════

def plot_fig4_camera(
    clip_id:      str,
    times_s:      np.ndarray,
    front_frames: np.ndarray,
    out_dir:      Path,
) -> Path:
    """Fig 4: 전방 카메라 타임라인 스트립."""
    N_CAM   = min(12, len(front_frames))
    indices = np.linspace(0, len(front_frames) - 1, N_CAM, dtype=int)
    cmap_t  = plt.cm.plasma(np.linspace(0.1, 0.95, N_CAM))

    fig, axes = plt.subplots(1, N_CAM, figsize=(N_CAM * 2.6, 3.8))
    fig.suptitle(f"[Fig 4] Front Camera  |  {clip_id}", fontsize=12)
    if N_CAM == 1:
        axes = [axes]
    for ax, idx, color in zip(axes, indices, cmap_t):
        ax.imshow(front_frames[idx])
        for spine in ax.spines.values():
            spine.set_edgecolor(color[:3])
            spine.set_linewidth(3)
        ax.set_title(f"t={times_s[idx]:.1f}s", fontsize=9, pad=3)
        ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / f"{clip_id[:16]}_fig4_camera.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
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
# 클립 단위 처리
# ══════════════════════════════════════════════════════════════

def _run_one_clip(
    pcl_path:  Path,
    model:     "CotendMLP",
    device:    "torch.device",
    out_dir:   Path,
    no_frames: bool,
    ema_alpha: float = 1.0,
) -> dict:
    """
    h5 파일 한 개를 로드 → 시뮬 → PNG/npz 저장.

    Returns:
        dict with keys: clip_id, ade_mlp, ade_def, gain,
                        fde_mlp, fde_def, status_mlp, status_def
    """
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
        front_frames     = None if no_frames else f["front_frames"][:]

    print(f"  clip_id         : {clip_id}")
    print(f"  n_frames        : {n_frames}  ({t_end - t_start:.1f}s @ {step_us // 1000}ms)")
    print(f"  cotend shape    : {cotend.shape}")
    print(f"  gt_xyz_local    : {gt_xyz_local.shape}")
    print(f"  global_origin_xy: {global_origin_xy.shape}")

    # ── 시뮬레이션 ────────────────────────────────────────────
    ema_str = f"EMA α={ema_alpha:.2f}" if ema_alpha < 1.0 else "EMA off"
    print(f"\nPCL 시뮬레이션 실행 중  ({n_frames} steps, {ema_str}) ...")
    result = run_pcl_simulation(
        cotend=cotend,
        gt_xyz_local=gt_xyz_local,
        gt_yaw_local=gt_yaw_local,
        v0_arr=v0_arr,
        global_origin_xy=global_origin_xy,
        global_yaw=global_yaw,
        model=model,
        device=device,
        ema_alpha=ema_alpha,
    )
    print("완료.")

    # ── 결과 요약 ─────────────────────────────────────────────
    err_mlp = result["errors_mlp"]
    err_def = result["errors_def"]
    ade_mlp = float(np.mean(err_mlp[1:]))
    ade_def = float(np.mean(err_def[1:]))
    fde_mlp = float(err_mlp[-1])
    fde_def = float(err_def[-1])
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
    n_invalid  = int((~valid_mask).sum())
    if n_invalid > 0:
        print(f"  open-loop frames: {n_invalid} (last {n_invalid * step_us // 1000}ms excluded from plot)")

    # ── PNG 저장 (4개 분리 figure) ────────────────────────────
    print("\n시각화 저장 중...")
    saved = []
    saved.append(plot_fig1_overview(clip_id, times_s, result, out_dir,
                                    valid_mask=valid_mask))
    saved.append(plot_fig2_weights(clip_id, times_s, result, out_dir))
    saved.append(plot_fig3_control(clip_id, times_s, result, out_dir,
                                   v0_arr=v0_arr, valid_mask=valid_mask))
    if front_frames is not None and len(front_frames) > 0:
        saved.append(plot_fig4_camera(clip_id, times_s, front_frames, out_dir))
    for sp in saved:
        print(f"  → {sp.name}")

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

    return dict(
        clip_id    = clip_id,
        ade_mlp    = ade_mlp,
        ade_def    = ade_def,
        fde_mlp    = fde_mlp,
        fde_def    = fde_def,
        gain       = gain,
        max_e_mlp  = float(np.max(err_mlp[1:])),
        max_e_def  = float(np.max(err_def[1:])),
    )


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="PCL 시뮬레이션 — build_pcl_dataset.py 출력 h5 파일 사용",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcl-clip", metavar="H5",
                     help="PCL h5 파일 경로 (단일 클립)")
    src.add_argument("--pcl-dir",  metavar="DIR",
                     help="PCL h5 파일이 들어있는 디렉토리 (전체 일괄 실행)")

    p.add_argument("--model-dir", required=True,
                   help="학습된 MLP 모델 디렉토리 (model_full.pt 또는 best_model.pt)")
    p.add_argument("--out",       required=True,
                   help="출력 디렉토리 (PNG + npz 저장)")
    p.add_argument("--device",    default="cuda",
                   help="추론 장치 (cpu / cuda, default: cuda)")
    p.add_argument("--no-frames", action="store_true",
                   help="전방 카메라 패널 제외 (메모리 절약)")
    p.add_argument("--ema-alpha", type=float, default=1.0, metavar="ALPHA",
                   help="MLP 가중치 EMA 평활화 계수 0<α≤1 (1.0=비활성, 권장 0.3~0.5, default: 1.0)")
    args = p.parse_args()

    if not (0.0 < args.ema_alpha <= 1.0):
        p.error("--ema-alpha 는 (0, 1] 범위여야 합니다.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, _ = load_mlp(Path(args.model_dir), device)

    # ── 처리할 h5 파일 목록 결정 ─────────────────────────────
    if args.pcl_clip:
        h5_files = [Path(args.pcl_clip)]
    else:
        h5_files = sorted(Path(args.pcl_dir).glob("*.h5"))
        if not h5_files:
            print(f"[ERROR] {args.pcl_dir} 에서 .h5 파일을 찾을 수 없습니다.")
            return
        print(f"총 {len(h5_files)}개 클립 발견: {args.pcl_dir}")

    # ── 클립별 순차 실행 ─────────────────────────────────────
    summaries = []
    errors_list = []
    for idx, h5_path in enumerate(h5_files, 1):
        print(f"\n[{idx}/{len(h5_files)}] {h5_path.name}")
        try:
            stats = _run_one_clip(h5_path, model, device, out_dir, args.no_frames,
                                  ema_alpha=args.ema_alpha)
            summaries.append(stats)
        except Exception as exc:
            print(f"  [SKIP] 처리 실패: {exc}")
            errors_list.append((h5_path.name, str(exc)))

    # ── 전체 요약 (다중 클립 시에만 출력) ────────────────────
    if len(summaries) > 1:
        print(f"\n{'#'*70}")
        print(f"  BATCH SUMMARY  ({len(summaries)} clips processed, {len(errors_list)} failed)")
        print(f"{'#'*70}")
        print(f"  {'Clip':38s}  {'ADE_MLP':>8}  {'ADE_Def':>8}  {'Gain':>7}  {'MaxE_MLP':>9}")
        print(f"  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*9}")
        ade_mlp_all = []
        ade_def_all = []
        for s in summaries:
            print(
                f"  {s['clip_id'][:38]:38s}  "
                f"{s['ade_mlp']:8.4f}  "
                f"{s['ade_def']:8.4f}  "
                f"{s['gain']:+7.2f}%  "
                f"{s['max_e_mlp']:9.4f}"
            )
            ade_mlp_all.append(s["ade_mlp"])
            ade_def_all.append(s["ade_def"])
        mean_mlp  = float(np.mean(ade_mlp_all))
        mean_def  = float(np.mean(ade_def_all))
        mean_gain = (mean_def - mean_mlp) / mean_def * 100 if mean_def > 0 else 0.0
        print(f"  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*9}")
        print(f"  {'MEAN':38s}  {mean_mlp:8.4f}  {mean_def:8.4f}  {mean_gain:+7.2f}%")
        if errors_list:
            print(f"\n  Failed clips:")
            for name, err in errors_list:
                print(f"    {name}: {err}")
        print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
