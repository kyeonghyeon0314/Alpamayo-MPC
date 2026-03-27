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


# ══════════════════════════════════════════════════════
# 경량 셀 플롯 (--all 격자용)
#   GT + Opt 궤적만 그림. Default MPC 계산 생략 → 2× 빠름.
#   저장된 ADE 값(labels/ade)을 제목에 표시.
# ══════════════════════════════════════════════════════

def plot_cell(ax, h5_path: Path):
    """격자 한 칸: GT 궤적(녹) + Opt MPC 궤적(청) + 저장된 ADE."""
    name = h5_path.name

    with h5py.File(h5_path, "r") as f:
        gt_xyz     = f["gt/future_xyz"][:]
        gt_ego     = f["gt/future_ego_states"][:]
        hist       = f["input/ego_history_ego_states"][:]
        w_opt      = f["labels/mpc_weights"][:]
        ade_stored = float(f["labels/ade"][()])
        valid      = bool(f["labels/valid"][()])

    v0       = float(hist[-1, 2])
    gt_yaw   = gt_ego[:, 3]
    gt_xy    = gt_xyz[:N, :2]

    xy_opt, _, _ = run_mpc(v0, gt_xyz, gt_yaw, w_opt)

    ax.plot(gt_xy[:, 1],  gt_xy[:, 0],  "g-",  lw=1.5, label="GT")
    ax.plot(xy_opt[:, 1], xy_opt[:, 0], "b--", lw=1.2, label=f"Opt")
    ax.scatter([0], [0], c="k", s=20, zorder=5)

    _set_traj_limits(ax, gt_xy, xy_opt)
    ax.grid(True, ls=":", lw=0.4)

    valid_mark = "✓" if valid else "✗"
    short_name = name[:22] + "…" if len(name) > 22 else name
    ax.set_title(
        f"{short_name}\n"
        f"v0={v0:.1f} ADE={ade_stored:.3f}m {valid_mark}\n"
        f"w=[{w_opt[0]:.1f},{w_opt[2]:.2f},{w_opt[3]:.3f},{w_opt[4]:.2f}]",
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
        gt_xyz   = f["gt/future_xyz"][:]
        gt_ego   = f["gt/future_ego_states"][:]
        hist     = f["input/ego_history_ego_states"][:]
        if "labels/mpc_weights" not in f:
            ax_traj.set_title(f"{name[:24]}\n[NO LABEL]", fontsize=7)
            return
        w_opt = f["labels/mpc_weights"][:]

    v0       = float(hist[-1, 2])
    gt_yaw   = gt_ego[:, 3]
    gt_xy    = gt_xyz[:N, :2]
    t_axis   = np.arange(1, N+1) * DT

    xy_opt, U_opt, ade_opt = run_mpc(v0, gt_xyz, gt_yaw, w_opt)
    xy_def, U_def, ade_def = run_mpc(v0, gt_xyz, gt_yaw, WEIGHTS_DEFAULT)

    ax_traj.plot(gt_xy[:, 1],  gt_xy[:, 0],  "g-",  lw=2.0, label="GT")
    ax_traj.plot(xy_opt[:, 1], xy_opt[:, 0], "b--", lw=1.5, label=f"Opt ADE={ade_opt:.3f}m")
    ax_traj.plot(xy_def[:, 1], xy_def[:, 0], "r:",  lw=1.2, label=f"Default ADE={ade_def:.3f}m")
    ax_traj.scatter([0], [0], c="k", s=30, zorder=5)

    _set_traj_limits(ax_traj, gt_xy, xy_opt, xy_def)
    ax_traj.set_xlabel("Y (lateral) [m]", fontsize=7)
    ax_traj.set_ylabel("X (long.) [m]",   fontsize=7)
    ax_traj.set_title(
        f"{name[:30]}\nv0={v0:.1f}m/s  w=[{w_opt[0]:.1f},{w_opt[2]:.2f},{w_opt[3]:.3f},{w_opt[4]:.2f}]",
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
    ax_ctrl.legend(fontsize=6, ncol=2)
    ax_ctrl.grid(True, ls=":", lw=0.5)


# ══════════════════════════════════════════════════════
# 단일 파일 상세 플롯 (넓은 레이아웃)
# ══════════════════════════════════════════════════════

def plot_single(h5_path: Path, out_dir: Path):
    name = h5_path.stem

    with h5py.File(h5_path, "r") as f:
        gt_xyz   = f["gt/future_xyz"][:]
        gt_ego   = f["gt/future_ego_states"][:]
        hist     = f["input/ego_history_ego_states"][:]
        if "labels/mpc_weights" not in f:
            print(f"[WARN] {h5_path.name} 에 labels/mpc_weights 없음")
            return
        w_opt = f["labels/mpc_weights"][:]
        ade_stored = float(f["labels/ade"][()])

    v0       = float(hist[-1, 2])
    gt_yaw   = gt_ego[:, 3]
    gt_xy    = gt_xyz[:N, :2]
    t_axis   = np.arange(1, N+1) * DT

    xy_opt, U_opt, ade_opt = run_mpc(v0, gt_xyz, gt_yaw, w_opt)
    xy_def, U_def, ade_def = run_mpc(v0, gt_xyz, gt_yaw, WEIGHTS_DEFAULT)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        f"{h5_path.name}\nv0={v0:.1f} m/s | Opt ADE={ade_opt:.3f}m (stored={ade_stored:.3f}m)"
        f" | Default ADE={ade_def:.3f}m",
        fontsize=10
    )
    # 3행 3열: 좌열=궤적(3행), 중열=오차+steer+accel, 우열=가중치(3행)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── (좌) 궤적 ──
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(gt_xy[:, 1],  gt_xy[:, 0],  "g-",  lw=2.5,  label="GT")
    ax_traj.plot(xy_opt[:, 1], xy_opt[:, 0], "b--", lw=2.0,  label=f"Opt   ADE={ade_opt:.3f}m")
    ax_traj.plot(xy_def[:, 1], xy_def[:, 0], "r:",  lw=1.8,  label=f"Def   ADE={ade_def:.3f}m")
    ax_traj.scatter([0],[0], c="k", s=50, zorder=5, label="t0")
    for k in [4, 9, 14, 19]:
        ax_traj.annotate(f"{(k+1)*DT:.1f}s",
                         xy=(gt_xy[k,1], gt_xy[k,0]), fontsize=6, color="green",
                         xytext=(3, 3), textcoords="offset points")

    # 축 범위를 수동으로 계산해 equal-aspect + 여백 보장
    all_lat = np.concatenate([gt_xy[:,1], xy_opt[:,1], xy_def[:,1], [0.]])
    all_lon = np.concatenate([gt_xy[:,0], xy_opt[:,0], xy_def[:,0], [0.]])
    half = max(all_lat.max()-all_lat.min(), all_lon.max()-all_lon.min()) / 2 + 1.5
    cx = (all_lat.max() + all_lat.min()) / 2
    cy = (all_lon.max() + all_lon.min()) / 2
    ax_traj.set_xlim(cx + half, cx - half)   # x축 반전 (운전자 시점: 좌=+Y)
    ax_traj.set_ylim(cy - half, cy + half)

    ax_traj.set_xlabel("Y / lateral [m]"); ax_traj.set_ylabel("X / longitudinal [m]")
    ax_traj.set_title("Trajectory (vehicle frame)\n(← left turn | right turn →)")
    ax_traj.legend(loc="best", fontsize=8); ax_traj.grid(True, ls=":", lw=0.5)

    # ── (중상) 스텝별 거리 오차 ──
    # 각 타임스텝 k에서: ||pred_xy_k - gt_xy_k|| [m]
    # ADE = 이 값들의 평균 (Average Displacement Error)
    ax_err = fig.add_subplot(gs[0, 1])
    err_opt = np.linalg.norm(xy_opt - gt_xy, axis=1)
    err_def = np.linalg.norm(xy_def - gt_xy, axis=1)
    ax_err.plot(t_axis, err_opt, "b--", lw=1.8, label=f"Opt  ADE={err_opt.mean():.3f}m")
    ax_err.plot(t_axis, err_def, "r:",  lw=1.5, label=f"Def  ADE={err_def.mean():.3f}m")
    ax_err.set_xlabel("Time [s]"); ax_err.set_ylabel("||pred - GT|| [m]")
    ax_err.set_title("Per-step displacement error\n(ADE = mean of this curve)")
    ax_err.legend(fontsize=8); ax_err.grid(True, ls=":", lw=0.5)
    ax_err.set_ylim(bottom=0)

    # ── (중중) Steering 제어 입력 ──
    ax_steer = fig.add_subplot(gs[1, 1])
    ax_steer.plot(t_axis, U_opt[:,0], "b--", lw=1.8, label="steer_cmd (opt)")
    ax_steer.plot(t_axis, U_def[:,0], "r:",  lw=1.5, label="steer_cmd (def)")
    ax_steer.axhline(0, c="k", lw=0.5, ls="--")
    ax_steer.set_xlabel("Time [s]"); ax_steer.set_ylabel("Steering [rad]")
    ax_steer.set_title("Steering command")
    ax_steer.legend(fontsize=7); ax_steer.grid(True, ls=":", lw=0.5)

    # ── (중하) Acceleration 제어 입력 ──
    ax_accel = fig.add_subplot(gs[2, 1])
    ax_accel.plot(t_axis, U_opt[:,1], "b--", lw=1.8, label="accel_cmd (opt)")
    ax_accel.plot(t_axis, U_def[:,1], "r:",  lw=1.5, label="accel_cmd (def)")
    ax_accel.axhline(0, c="k", lw=0.5, ls="--")
    ax_accel.set_xlabel("Time [s]"); ax_accel.set_ylabel("Accel [m/s²]")
    ax_accel.set_title("Acceleration command")
    ax_accel.legend(fontsize=7); ax_accel.grid(True, ls=":", lw=0.5)

    # ── (우) 가중치 막대 비교 ──
    ax_w = fig.add_subplot(gs[:, 2])
    labels_w = ["long\npos", "lat\npos\n(fixed)", "heading", "steer\nrate", "accel\nrate"]
    x = np.arange(len(labels_w)); width=0.35
    ax_w.bar(x - width/2, w_opt,           width, label="Optimal", color="steelblue", alpha=0.8)
    ax_w.bar(x + width/2, WEIGHTS_DEFAULT, width, label="Default",  color="tomato",    alpha=0.6)
    ax_w.set_xticks(x); ax_w.set_xticklabels(labels_w, fontsize=8)
    ax_w.set_ylabel("Weight value"); ax_w.set_title("MPC cost weights")
    ax_w.legend(); ax_w.grid(True, axis="y", ls=":", lw=0.5)
    for i,(ov,dv) in enumerate(zip(w_opt, WEIGHTS_DEFAULT)):
        ax_w.text(i-width/2, ov+0.05, f"{ov:.2f}", ha="center", fontsize=7, color="steelblue")
        ax_w.text(i+width/2, dv+0.05, f"{dv:.2f}", ha="center", fontsize=7, color="tomato")

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
