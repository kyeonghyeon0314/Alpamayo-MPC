#!/usr/bin/env python3
"""
MPC 레이블 시각화

저장된 최적 가중치(labels/mpc_weights)로 MPC 궤적을 재현하고
GT 궤적 및 기본 가중치 궤적과 비교합니다.

사용법:
    # 라벨 있는 파일 전부 페이지네이션 격자로 저장
    python alpamayo_dataset/viz_mpc_label.py \
        --data-dir alpamayo_dataset/data/collected \
        --all --out viz_out

    # 페이지당 셀 수 조정 (기본 20)
    python alpamayo_dataset/viz_mpc_label.py \
        --data-dir alpamayo_dataset/data/collected \
        --all --page-size 30 --out viz_out

    # 상세 플롯 N개 (궤적+제어+가중치)
    python alpamayo_dataset/viz_mpc_label.py \
        --data-dir alpamayo_dataset/data/collected \
        --n-samples 12 --out viz_out

    # 특정 파일 단일 상세 플롯
    python alpamayo_dataset/viz_mpc_label.py \
        --file alpamayo_dataset/data/collected/xxx.h5 \
        --out viz_out

결과: viz_out/ 디렉토리에 PNG 저장
  --all 모드: mpc_all_p001.png, mpc_all_p002.png, ...
  --n-samples: mpc_grid.png + <name>_mpc_viz.png
  --file: <name>_mpc_viz.png
"""

import argparse
import itertools
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

# ── 전처리: matplotlib 백엔드를 Agg로 고정 (X11 없어도 작동) ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════
# 공유 MPC 모듈 (mpc.py) — 파라미터 / 함수 단일 정의
# ══════════════════════════════════════════════════════
sys.path.insert(0, str(Path(__file__).parent))
from mpc import (
    DT, N, NX, NU,
    IX, IY, IYAW, IVX, IVY, IYR, ISTEER, IACCEL,
    V_MIN_LIN, QP_REG, W_LAT_FIXED, WEIGHTS_DEFAULT,
    U_MIN, U_MAX,
    _make_QR, _linearize, _build_prediction_matrices, _solve_qp, run_mpc,
    compute_x0,
)


# ══════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════

_INDEX_FILE = "labeled_index.txt"  # label_mpc_weights.py 와 동일


def _load_index(data_dir: Path) -> set[str]:
    """라벨 완료 파일명(basename) 집합 로드. 파일 없으면 빈 집합 반환."""
    idx = data_dir / _INDEX_FILE
    if not idx.exists():
        return set()
    return set(idx.read_text(encoding="utf-8").splitlines())


def _has_label(path: Path) -> bool:
    """h5 파일에 labels/mpc_weights 가 존재하는지 확인 (fallback용)."""
    try:
        with h5py.File(path, "r") as f:
            return "labels/mpc_weights" in f
    except Exception:
        return False


def _has_label_str(path_str: str) -> bool:
    """ProcessPoolExecutor용 — str 경로로 받아 labels/mpc_weights 존재 여부 반환."""
    try:
        with h5py.File(path_str, "r") as f:
            return "labels/mpc_weights" in f
    except Exception:
        return False


def _scan_labeled(h5_files: list, workers: int = 4) -> list:
    """
    h5_files 중 labels/mpc_weights 가 있는 파일 목록을 병렬로 스캔.
    인덱스가 없을 때의 fallback. workers=1 이면 순차 실행.
    """
    path_strs = [str(p) for p in h5_files]
    if workers <= 1:
        mask = [_has_label_str(s) for s in path_strs]
    else:
        chunksize = max(1, len(path_strs) // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            mask = list(ex.map(_has_label_str, path_strs, chunksize=chunksize))
    return [p for p, ok in zip(h5_files, mask) if ok]


def _get_labeled(data_dir: Path, workers: int) -> list:
    """인덱스가 있으면 glob 없이 경로 직접 구성, 없으면 h5 스캔 fallback."""
    idx_set = _load_index(data_dir)
    if idx_set:
        labeled = [data_dir / name for name in sorted(idx_set)]
        print(f"인덱스: {len(labeled)}개 라벨 파일 (glob 스킵)")
        return labeled
    # fallback: 전체 glob 후 h5 열어서 확인
    print("라벨 있는 파일 스캔 중...", end=" ", flush=True)
    all_h5 = sorted(data_dir.glob("*.h5"))
    labeled = _scan_labeled(all_h5, workers)
    print(f"{len(labeled)}개 발견")
    return labeled


def _set_traj_limits(ax, *xy_arrays):
    """여러 (N,2) 배열 + 원점(0,0)을 포함하는 equal-aspect 축 범위 설정."""
    all_lat = np.concatenate([a[:, 1] for a in xy_arrays] + [np.array([0.])])
    all_lon = np.concatenate([a[:, 0] for a in xy_arrays] + [np.array([0.])])
    half = max(all_lat.max() - all_lat.min(), all_lon.max() - all_lon.min()) / 2 + 1.5
    cx = (all_lat.max() + all_lat.min()) / 2
    cy = (all_lon.max() + all_lon.min()) / 2
    ax.set_xlim(cx + half, cx - half)   # x축 반전 (← left turn | right turn →)
    ax.set_ylim(cy - half, cy + half)


def _comfort_metrics(U: np.ndarray) -> tuple[float, float]:
    """U (N, 2) → (steer_rate_rms [rad/step], jerk_rms [(m/s²)/step])."""
    sr_rms   = float(np.sqrt(np.mean(np.diff(U[:, 0]) ** 2)))
    jerk_rms = float(np.sqrt(np.mean(np.diff(U[:, 1]) ** 2)))
    return sr_rms, jerk_rms


# ══════════════════════════════════════════════════════
# 경량 셀 플롯 (--all 격자용)
#   GT + Opt 궤적만 그림. Default MPC 계산 생략 → 2× 빠름.
#   저장된 ADE 값(labels/ade)을 제목에 표시.
# ══════════════════════════════════════════════════════

def plot_cell(ax, h5_path: Path):
    """격자 한 칸: GT 궤적(녹) + Opt MPC 궤적(청) + 저장된 ADE."""
    name = h5_path.name

    with h5py.File(h5_path, "r") as f:
        gt_xyz    = f["gt/future_xyz"][:]
        gt_ego    = f["gt/future_ego_states"][:]
        hist      = f["input/ego_history_ego_states"][:]
        hist_vel  = f["input/ego_history_vel"][:]
        hist_curv = f["input/ego_history_curv"][:]
        hist_quat = f["input/ego_history_quat_global"][:]
        w_opt      = f["labels/mpc_weights"][:]
        ade_stored = float(f["labels/ade"][()])
        valid      = bool(f["labels/valid"][()])
        sr_stored  = float(f["labels/steer_rate_rms"][()]) if "labels/steer_rate_rms" in f else None
        jr_stored  = float(f["labels/jerk_rms"][()])       if "labels/jerk_rms"       in f else None

    speed    = float(hist[-1, 2])
    x0       = compute_x0(speed, hist[-1, 4], hist_vel[-1], hist_curv[-1, 0], hist_quat[-1])
    gt_yaw   = gt_ego[:, 3]
    gt_xy    = gt_xyz[:N, :2]

    xy_opt, _, _ = run_mpc(speed, gt_xyz, gt_yaw, w_opt, x0_full=x0)

    ax.plot(gt_xy[:, 1],  gt_xy[:, 0],  "g-",  lw=1.5, label="GT")
    ax.plot(xy_opt[:, 1], xy_opt[:, 0], "b--", lw=1.2, label="Opt")
    ax.scatter([0], [0], c="k", s=20, zorder=5)
    _T_MARKS = [4, 9, 14, 19]
    ax.scatter(gt_xy[_T_MARKS, 1], gt_xy[_T_MARKS, 0],
               c="green", s=18, zorder=6, linewidths=0)
    ax.scatter(xy_opt[_T_MARKS, 1], xy_opt[_T_MARKS, 0],
               c="steelblue", s=18, zorder=6, linewidths=0)

    _set_traj_limits(ax, gt_xy, xy_opt)
    ax.grid(True, ls=":", lw=0.4)

    valid_mark   = "✓" if valid else "✗"
    short_name   = name[:22] + "…" if len(name) > 22 else name
    comfort_line = (f"\nSR={sr_stored:.4f} J={jr_stored:.4f}"
                    if sr_stored is not None else "")
    ax.set_title(
        f"{short_name}\n"
        f"v0={speed:.1f} ADE={ade_stored:.3f}m {valid_mark}\n"
        f"w=[{w_opt[0]:.1f},{w_opt[2]:.2f},{w_opt[3]:.3f},{w_opt[4]:.2f}]"
        f"{comfort_line}",
        fontsize=5.5,
    )
    ax.tick_params(labelsize=5)


# ══════════════════════════════════════════════════════
# 상세 셀 플롯 (--n-samples 격자용)
#   GT + Opt + Default 궤적 + 제어 입력
# ══════════════════════════════════════════════════════

def plot_sample(ax_traj, ax_ctrl, h5_path):
    """ax_traj: 궤적 subplot, ax_ctrl: 제어입력 subplot"""
    name = Path(h5_path).name

    with h5py.File(h5_path, "r") as f:
        gt_xyz    = f["gt/future_xyz"][:]
        gt_ego    = f["gt/future_ego_states"][:]
        hist      = f["input/ego_history_ego_states"][:]
        hist_vel  = f["input/ego_history_vel"][:]
        hist_curv = f["input/ego_history_curv"][:]
        hist_quat = f["input/ego_history_quat_global"][:]
        if "labels/mpc_weights" not in f:
            ax_traj.set_title(f"{name[:24]}\n[NO LABEL]", fontsize=7)
            return
        w_opt = f["labels/mpc_weights"][:]

    speed    = float(hist[-1, 2])
    x0       = compute_x0(speed, hist[-1, 4], hist_vel[-1], hist_curv[-1, 0], hist_quat[-1])
    gt_yaw   = gt_ego[:, 3]
    gt_xy    = gt_xyz[:N, :2]
    t_axis   = np.arange(1, N+1) * DT

    xy_opt, U_opt, ade_opt = run_mpc(speed, gt_xyz, gt_yaw, w_opt,          x0_full=x0)
    xy_def, U_def, ade_def = run_mpc(speed, gt_xyz, gt_yaw, WEIGHTS_DEFAULT, x0_full=x0)

    sr_opt, jr_opt = _comfort_metrics(U_opt)
    sr_def, jr_def = _comfort_metrics(U_def)

    ax_traj.plot(gt_xy[:, 1],  gt_xy[:, 0],  "g-",  lw=2.0, label="GT")
    ax_traj.plot(xy_opt[:, 1], xy_opt[:, 0], "b--", lw=1.5, label=f"Opt ADE={ade_opt:.3f}m")
    ax_traj.plot(xy_def[:, 1], xy_def[:, 0], "r:",  lw=1.2, label=f"Default ADE={ade_def:.3f}m")
    ax_traj.scatter([0], [0], c="k", s=30, zorder=5)
    _T_MARKS = [4, 9, 14, 19]
    ax_traj.scatter(gt_xy[_T_MARKS, 1], gt_xy[_T_MARKS, 0],
                    c="green", s=20, zorder=6, linewidths=0)
    ax_traj.scatter(xy_opt[_T_MARKS, 1], xy_opt[_T_MARKS, 0],
                    c="steelblue", s=20, zorder=6, linewidths=0)
    ax_traj.scatter(xy_def[_T_MARKS, 1], xy_def[_T_MARKS, 0],
                    c="tomato", s=20, zorder=6, linewidths=0)

    _set_traj_limits(ax_traj, gt_xy, xy_opt, xy_def)
    ax_traj.set_xlabel("Y (lateral) [m]", fontsize=7)
    ax_traj.set_ylabel("X (long.) [m]",   fontsize=7)
    ax_traj.set_title(
        f"{name[:30]}\nv0={speed:.1f}m/s  w=[{w_opt[0]:.1f},{w_opt[2]:.2f},{w_opt[3]:.3f},{w_opt[4]:.2f}]",
        fontsize=7,
    )
    ax_traj.legend(fontsize=6, loc="best")
    ax_traj.grid(True, ls=":", lw=0.5)

    ax_ctrl.plot(t_axis, U_opt[:, 0], "b--", lw=1.5, label="steer (opt)")
    ax_ctrl.plot(t_axis, U_opt[:, 1], "b-",  lw=1.5, label="accel (opt)")
    ax_ctrl.plot(t_axis, U_def[:, 0], "r:",  lw=1.2, label="steer (def)")
    ax_ctrl.plot(t_axis, U_def[:, 1], "r-.", lw=1.2, label="accel (def)")
    ax_ctrl.axhline(0, c="k", lw=0.5, ls="--")
    ax_ctrl.set_xlabel("Time [s]", fontsize=7)
    ax_ctrl.set_ylabel("Cmd",      fontsize=7)
    ax_ctrl.set_title(
        f"Control  |  opt: SR={sr_opt:.4f}  Jerk={jr_opt:.4f}"
        f"   def: SR={sr_def:.4f}  Jerk={jr_def:.4f}",
        fontsize=6,
    )
    ax_ctrl.legend(fontsize=6, ncol=2)
    ax_ctrl.grid(True, ls=":", lw=0.5)


# ══════════════════════════════════════════════════════
# 단일 파일 상세 플롯 (넓은 레이아웃)
# ══════════════════════════════════════════════════════

def plot_single(h5_path: Path, out_dir: Path):
    name = h5_path.stem

    with h5py.File(h5_path, "r") as f:
        gt_xyz    = f["gt/future_xyz"][:]
        gt_ego    = f["gt/future_ego_states"][:]
        hist      = f["input/ego_history_ego_states"][:]
        hist_vel  = f["input/ego_history_vel"][:]
        hist_curv = f["input/ego_history_curv"][:]
        hist_quat = f["input/ego_history_quat_global"][:]
        if "labels/mpc_weights" not in f:
            print(f"[WARN] {h5_path.name} has no labels/mpc_weights")
            return
        w_opt      = f["labels/mpc_weights"][:]
        ade_stored = float(f["labels/ade"][()])

    speed    = float(hist[-1, 2])
    x0       = compute_x0(speed, hist[-1, 4], hist_vel[-1], hist_curv[-1, 0], hist_quat[-1])
    gt_yaw   = gt_ego[:, 3]
    gt_xy    = gt_xyz[:N, :2]
    t_axis   = np.arange(1, N + 1) * DT
    t_diff   = t_axis[:-1]          # N-1 midpoints for diff-based plots

    xy_opt, U_opt, ade_opt = run_mpc(speed, gt_xyz, gt_yaw, w_opt,          x0_full=x0)
    xy_def, U_def, ade_def = run_mpc(speed, gt_xyz, gt_yaw, WEIGHTS_DEFAULT, x0_full=x0)

    sr_opt, jr_opt = _comfort_metrics(U_opt)
    sr_def, jr_def = _comfort_metrics(U_def)

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f"{h5_path.name}   v0={speed:.1f} m/s   stored ADE={ade_stored:.3f}m\n"
        f"Opt  ADE={ade_opt:.3f}m  SR={sr_opt:.4f}  Jerk={jr_opt:.4f}   |   "
        f"Def  ADE={ade_def:.3f}m  SR={sr_def:.4f}  Jerk={jr_def:.4f}",
        fontsize=9,
    )

    # Layout: 4 rows × 3 cols
    #   col 0 (all rows) : trajectory
    #   row 0, col 1-2   : ADE per-step error  (wide)
    #   row 1, col 1     : steer command
    #   row 1, col 2     : accel command
    #   row 2, col 1     : steer rate  |Δsteer|
    #   row 2, col 2     : jerk        |Δaccel|
    #   row 3, col 1-2   : MPC weight bar chart (wide)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.60, wspace=0.35)

    # ── trajectory (col 0) ──────────────────────────────
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(gt_xy[:, 1],  gt_xy[:, 0],  "g-",  lw=2.5, label="GT")
    ax_traj.plot(xy_opt[:, 1], xy_opt[:, 0], "b--", lw=2.0, label=f"Opt  ADE={ade_opt:.3f}m")
    ax_traj.plot(xy_def[:, 1], xy_def[:, 0], "r:",  lw=1.8, label=f"Def  ADE={ade_def:.3f}m")
    ax_traj.scatter([0], [0], c="k", s=50, zorder=5, label="t0")
    
    _T_MARKS = [4, 9, 14, 19]
    for k in _T_MARKS:
        ax_traj.annotate(f"{(k+1)*DT:.1f}s",
                         xy=(gt_xy[k, 1], gt_xy[k, 0]), fontsize=6, color="green",
                         xytext=(3, 3), textcoords="offset points")
    ax_traj.scatter(gt_xy[_T_MARKS, 1], gt_xy[_T_MARKS, 0],
                    c="green", s=25, zorder=6, linewidths=0)
    ax_traj.scatter(xy_opt[_T_MARKS, 1], xy_opt[_T_MARKS, 0],
                    c="steelblue", s=25, zorder=6, linewidths=0)
    ax_traj.scatter(xy_def[_T_MARKS, 1], xy_def[_T_MARKS, 0],
                    c="tomato", s=25, zorder=6, linewidths=0)
    all_lat = np.concatenate([gt_xy[:, 1], xy_opt[:, 1], xy_def[:, 1], [0.]])
    all_lon = np.concatenate([gt_xy[:, 0], xy_opt[:, 0], xy_def[:, 0], [0.]])
    half = max(all_lat.max() - all_lat.min(), all_lon.max() - all_lon.min()) / 2 + 1.5
    cx = (all_lat.max() + all_lat.min()) / 2
    cy = (all_lon.max() + all_lon.min()) / 2
    ax_traj.set_xlim(cx + half, cx - half)
    ax_traj.set_ylim(cy - half, cy + half)
    ax_traj.set_xlabel("Y / lateral [m]")
    ax_traj.set_ylabel("X / longitudinal [m]")
    ax_traj.set_title("Trajectory (vehicle frame)\n(← left turn | right turn →)")
    ax_traj.legend(loc="best", fontsize=8)
    ax_traj.grid(True, ls=":", lw=0.5)

    # ── ADE per-step error (row 0, cols 1-2) ────────────
    ax_err = fig.add_subplot(gs[0, 1:3])
    err_opt = np.linalg.norm(xy_opt - gt_xy, axis=1)
    err_def = np.linalg.norm(xy_def - gt_xy, axis=1)
    ax_err.plot(t_axis, err_opt, "b--", lw=1.8, label=f"Opt  ADE={err_opt.mean():.3f}m")
    ax_err.plot(t_axis, err_def, "r:",  lw=1.5, label=f"Def  ADE={err_def.mean():.3f}m")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("||pred - GT|| [m]")
    ax_err.set_title("Per-step displacement error  (ADE = mean)")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, ls=":", lw=0.5)
    ax_err.set_ylim(bottom=0)

    # ── Steer command (row 1, col 1) ────────────────────
    ax_steer = fig.add_subplot(gs[1, 1])
    ax_steer.plot(t_axis, U_opt[:, 0], "b--", lw=1.8, label="opt")
    ax_steer.plot(t_axis, U_def[:, 0], "r:",  lw=1.5, label="def")
    ax_steer.axhline(0, c="k", lw=0.5, ls="--")
    ax_steer.set_xlabel("Time [s]")
    ax_steer.set_ylabel("Steering [rad]")
    ax_steer.set_title("Steering command")
    ax_steer.legend(fontsize=7)
    ax_steer.grid(True, ls=":", lw=0.5)

    # ── Accel command (row 1, col 2) ────────────────────
    ax_accel = fig.add_subplot(gs[1, 2])
    ax_accel.plot(t_axis, U_opt[:, 1], "b--", lw=1.8, label="opt")
    ax_accel.plot(t_axis, U_def[:, 1], "r:",  lw=1.5, label="def")
    ax_accel.axhline(0, c="k", lw=0.5, ls="--")
    ax_accel.set_xlabel("Time [s]")
    ax_accel.set_ylabel("Accel [m/s²]")
    ax_accel.set_title("Acceleration command")
    ax_accel.legend(fontsize=7)
    ax_accel.grid(True, ls=":", lw=0.5)

    # ── Steer rate |Δsteer| (row 2, col 1) ──────────────
    ax_sr = fig.add_subplot(gs[2, 1])
    ax_sr.plot(t_diff, np.abs(np.diff(U_opt[:, 0])), "b--", lw=1.8,
               label=f"opt  rms={sr_opt:.4f}")
    ax_sr.plot(t_diff, np.abs(np.diff(U_def[:, 0])), "r:",  lw=1.5,
               label=f"def  rms={sr_def:.4f}")
    ax_sr.set_xlabel("Time [s]")
    ax_sr.set_ylabel("|Δsteer| [rad/step]")
    ax_sr.set_title("Steering rate  |Δsteer|")
    ax_sr.legend(fontsize=7)
    ax_sr.grid(True, ls=":", lw=0.5)
    ax_sr.set_ylim(bottom=0)

    # ── Jerk |Δaccel| (row 2, col 2) ────────────────────
    ax_jerk = fig.add_subplot(gs[2, 2])
    ax_jerk.plot(t_diff, np.abs(np.diff(U_opt[:, 1])), "b--", lw=1.8,
                 label=f"opt  rms={jr_opt:.4f}")
    ax_jerk.plot(t_diff, np.abs(np.diff(U_def[:, 1])), "r:",  lw=1.5,
                 label=f"def  rms={jr_def:.4f}")
    ax_jerk.set_xlabel("Time [s]")
    ax_jerk.set_ylabel("|Δaccel| [(m/s²)/step]")
    ax_jerk.set_title("Jerk  |Δaccel|")
    ax_jerk.legend(fontsize=7)
    ax_jerk.grid(True, ls=":", lw=0.5)
    ax_jerk.set_ylim(bottom=0)

    # ── MPC weight bar (row 3, cols 1-2) ────────────────
    ax_w = fig.add_subplot(gs[3, 1:3])
    labels_w = ["long pos", "lat pos\n(fixed)", "heading", "steer rate", "accel rate"]
    x = np.arange(len(labels_w))
    width = 0.35
    ax_w.bar(x - width / 2, w_opt,           width, label="Optimal", color="steelblue", alpha=0.8)
    ax_w.bar(x + width / 2, WEIGHTS_DEFAULT, width, label="Default",  color="tomato",    alpha=0.6)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(labels_w, fontsize=8)
    ax_w.set_ylabel("Weight value")
    ax_w.set_title("MPC cost weights")
    ax_w.legend()
    ax_w.grid(True, axis="y", ls=":", lw=0.5)
    for i, (ov, dv) in enumerate(zip(w_opt, WEIGHTS_DEFAULT)):
        ax_w.text(i - width / 2, ov + 0.05, f"{ov:.2f}", ha="center", fontsize=7, color="steelblue")
        ax_w.text(i + width / 2, dv + 0.05, f"{dv:.2f}", ha="center", fontsize=7, color="tomato")

    out_path = out_dir / f"{name}_mpc_viz.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════
# 상세 격자 플롯 (--n-samples 용)
#   GT + Opt + Default 궤적 + 제어 입력 (2행 per 샘플)
# ══════════════════════════════════════════════════════

def plot_detail_grid(h5_files: list, out_dir: Path, n_samples: int, workers: int = 4):
    """라벨 있는 파일에서 최대 n_samples개를 상세 격자(궤적+제어)로 저장."""
    print("라벨 있는 파일 스캔 중...", end=" ", flush=True)
    labeled = _scan_labeled(h5_files, workers)[:n_samples]
    print(f"{len(labeled)}개 사용")
    if not labeled:
        print("[WARN] 라벨 있는 파일이 없습니다.")
        return

    ncols = 4
    nrows = math.ceil(len(labeled) / ncols)
    fig, axes = plt.subplots(nrows * 2, ncols,
                             figsize=(ncols * 4, nrows * 5),
                             squeeze=False)
    fig.suptitle(f"MPC Label Detail — {len(labeled)} samples", fontsize=12)

    for idx, path in enumerate(labeled):
        r, c = divmod(idx, ncols)
        try:
            plot_sample(axes[r * 2, c], axes[r * 2 + 1, c], path)
        except Exception as e:
            axes[r * 2, c].set_title(f"ERROR\n{e}", fontsize=7)

    for idx in range(len(labeled), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r * 2,     c].set_visible(False)
        axes[r * 2 + 1, c].set_visible(False)

    out_path = out_dir / "mpc_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"상세 격자 저장: {out_path}")


# ══════════════════════════════════════════════════════
# 전체 라벨 파일 개별 PNG 저장 (--all 용)
#   라벨 있는 파일마다 plot_single 호출 → <name>_mpc_viz.png
# ══════════════════════════════════════════════════════

def _plot_one(args: tuple) -> str:
    """멀티프로세싱 worker: (h5_path, out_dir) → 저장 경로 또는 에러 메시지."""
    h5_path, out_dir = args
    # matplotlib은 fork 후 재초기화 필요
    import matplotlib
    matplotlib.use("Agg")
    try:
        result = plot_single(Path(h5_path), Path(out_dir))
        return f"OK {result}"
    except Exception as e:
        return f"ERR {h5_path}: {e}"


def plot_all_labeled(labeled: list, out_dir: Path, workers: int = 4):
    """
    라벨 있는 파일 전부를 개별 PNG로 저장 (pre-filtered list 입력).

    출력: viz_out/<uuid>_mpc_viz.png  (파일당 1개)
    이미 PNG가 존재하면 스킵.
    """
    total = len(labeled)

    if total == 0:
        print("[WARN] 라벨 있는 파일이 없습니다.")
        return

    # 이미 생성된 PNG 스킵
    tasks = []
    skipped = 0
    for p in labeled:
        out_path = out_dir / f"{p.stem}_mpc_viz.png"
        if out_path.exists():
            skipped += 1
        else:
            tasks.append((str(p), str(out_dir)))

    if skipped:
        print(f"  스킵 (이미 존재): {skipped}개 / 처리 대상: {len(tasks)}개")
    if not tasks:
        print("모두 이미 생성됨. (--overwrite 없이 재실행 시 스킵)")
        return

    done = 0
    errors = []

    if workers == 1:
        for args in tasks:
            msg = _plot_one(args)
            done += 1
            if msg.startswith("ERR"):
                errors.append(msg)
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done:4d}/{len(tasks)}] 완료 (오류: {len(errors)}개)")
    else:
        with mp.Pool(processes=workers) as pool:
            for msg in pool.imap_unordered(_plot_one, tasks, chunksize=1):
                done += 1
                if msg.startswith("ERR"):
                    errors.append(msg)
                if done % 50 == 0 or done == len(tasks):
                    print(f"  [{done:4d}/{len(tasks)}] 완료 (오류: {len(errors)}개)")

    print(f"\n전체 완료: {done}개 저장, {len(errors)}개 오류 → {out_dir.resolve()}")
    for e in errors[:10]:
        print(f"  {e}")


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MPC 레이블 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file",     help="단일 h5 파일 경로")
    src.add_argument("--data-dir", help="h5 파일 디렉토리")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all",       action="store_true",
                      help="라벨 있는 파일 전부 개별 PNG로 저장")
    mode.add_argument("--n-samples", type=int, default=0,
                      help="상세 격자 플롯 샘플 수 (0=단일 상세 플롯)")

    parser.add_argument("--workers",   type=int, default=4,
                        help="--all 모드: 병렬 프로세스 수 (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="--all 모드: 이미 존재하는 PNG도 덮어쓰기")
    parser.add_argument("--out",       default="viz_out",
                        help="출력 디렉토리 (default: viz_out)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="처음 5개 파일만 즉시 시각화 (glob/인덱스 스킵)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 라벨 파일 목록 수집 ──────────────────────────────
    if args.file:
        labeled = [Path(args.file)]
    elif args.dry_run:
        # glob/인덱스 없이 첫 5개 파일에서 라벨 있는 것만 즉시 수집
        data_dir_path = Path(args.data_dir)
        labeled = [
            p for p in itertools.islice(data_dir_path.glob("*.h5"), 20)
            if _has_label(p)
        ][:5]
        print(f"dry-run: {len(labeled)}개 파일 사용")
    else:
        data_dir_path = Path(args.data_dir)
        labeled = _get_labeled(data_dir_path, args.workers)

    if not labeled:
        print("[WARN] 라벨 있는 파일이 없습니다.")
        sys.exit(0)

    # ── 모드 분기 ──────────────────────────────────────
    if args.all:
        if args.overwrite:
            for p in out_dir.glob("*_mpc_viz.png"):
                p.unlink()
        plot_all_labeled(labeled, out_dir, workers=args.workers)

    elif args.n_samples > 0:
        for p in labeled[:args.n_samples]:
            plot_single(p, out_dir)

    else:
        plot_single(labeled[0], out_dir)

    print(f"\n완료. 결과: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
