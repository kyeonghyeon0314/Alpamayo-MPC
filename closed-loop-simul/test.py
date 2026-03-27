#!/usr/bin/env python3
"""
MLP 예측 가중치 시각화

학습된 MLP 모델로 테스트셋 cotend를 추론하고,
예측 MPC 가중치로 궤적을 생성해 GT / Opt(IOC) / MLP / Default 와 비교합니다.

비교 대상:
  GT      : 실제 주행 궤적 (녹색 실선)
  Opt     : IOC 최적 가중치 MPC 궤적 (파랑 점선)
  MLP     : MLP 예측 가중치 MPC 궤적 (보라 일점쇄선)
  Default : alpasim 기본 가중치 MPC 궤적 (빨강 점선)

실행:
    # 테스트셋 전체 (split/test/*.h5)
    python alpamayo_dataset/test.py \
        --model-dir alpamayo_dataset/mlp_out \
        --split-dir alpamayo_dataset/data/split \
        --out       alpamayo_dataset/mlp_viz_out

     최대 N개만
    python alpamayo_dataset/test.py \
        --model-dir alpamayo_dataset/mlp_out \
        --split-dir alpamayo_dataset/data/split \
        --n-samples 20 \
        --out       alpamayo_dataset/mlp_viz_out

    # 특정 파일 하나
    python alpamayo_dataset/test.py \
        --model-dir alpamayo_dataset/mlp_out \
        --file      alpamayo_dataset/data/split/test/xxx.h5 \
        --out       alpamayo_dataset/mlp_viz_out
"""

import argparse
import random
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# 공유 MPC 모듈 / 시각화 유틸 / MLP 모델 임포트 (같은 디렉토리)
sys.path.insert(0, str(Path(__file__).parent))
from mpc import run_mpc, WEIGHTS_DEFAULT, W_LAT_FIXED, N, DT, NX, IX, IY
from viz_mpc_label import _set_traj_limits
from model import CotendMLP, PREDICT_IDX, load_mlp


# ══════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════

def _to_weights5(log_theta: np.ndarray) -> np.ndarray:
    """MLP log-scale 출력 (4,) → MPC용 5-weight 벡터.

    PREDICT_IDX = [0, 2, 3, 4] 이므로 index 1(lat_pos)에 W_LAT_FIXED 삽입.
    weights5 순서: [long, lat(fixed), heading, steer_r, accel_r]
    """
    w4 = np.exp(log_theta.astype(np.float64))
    return np.array([w4[0], W_LAT_FIXED, w4[1], w4[2], w4[3]])


@torch.no_grad()
def predict_weights(model: CotendMLP, cotend: np.ndarray, device: torch.device) -> np.ndarray:
    """cotend (4096,) → weights5 (5,)."""
    x = torch.from_numpy(cotend.astype(np.float32)).unsqueeze(0).to(device)
    log_theta = model(x).squeeze(0).cpu().numpy()  # (4,) log-scale
    return _to_weights5(log_theta)


# ══════════════════════════════════════════════════════
# 단일 파일 상세 플롯
# ══════════════════════════════════════════════════════

def plot_mlp_single(h5_path: Path, model: CotendMLP, device: torch.device, out_dir: Path):
    """h5 파일 1개에 대해 GT/Opt/MLP/Default 비교 플롯 저장.

    레이아웃 (3행 × 3열):
        좌열  (3행) : 궤적 (4개 비교)
        중상        : 스텝별 거리 오차
        중중        : Steering 명령
        중하        : Acceleration 명령
        우열  (3행) : 가중치 막대 비교 (Opt / MLP / Default)
    """
    name = h5_path.stem

    # ── 데이터 로드 ───────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        gt_xyz   = f["gt/future_xyz"][:]
        gt_ego   = f["gt/future_ego_states"][:]
        hist     = f["input/ego_history_ego_states"][:]
        cotend   = f["output/cotend_hidden_state"][:]
        has_opt  = "labels/mpc_weights" in f
        w_opt    = f["labels/mpc_weights"][:] if has_opt else None
        ade_stored = float(f["labels/ade"][()]) if has_opt else float("nan")

    v0       = float(hist[-1, 2])
    gt_yaw   = gt_ego[:, 3]
    gt_xy    = gt_xyz[:N, :2]
    t_axis   = np.arange(1, N + 1) * DT

    # ── MPC 실행 ──────────────────────────────────────
    w_mlp = predict_weights(model, cotend, device)
    xy_mlp, U_mlp, ade_mlp = run_mpc(v0, gt_xyz, gt_yaw, w_mlp)
    xy_def, U_def, ade_def = run_mpc(v0, gt_xyz, gt_yaw, WEIGHTS_DEFAULT)

    if has_opt:
        xy_opt, U_opt, ade_opt = run_mpc(v0, gt_xyz, gt_yaw, w_opt)
    else:
        xy_opt = U_opt = None
        ade_opt = float("nan")

    # ── Figure 구성 ───────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    title_parts = [
        f"{h5_path.name}",
        f"v0={v0:.1f} m/s",
        f"MLP ADE={ade_mlp:.3f}m",
    ]
    if has_opt:
        title_parts.append(f"Opt ADE={ade_opt:.3f}m (stored={ade_stored:.3f}m)")
    title_parts.append(f"Default ADE={ade_def:.3f}m")
    fig.suptitle("  |  ".join(title_parts), fontsize=9)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── 궤적 (좌열) ──────────────────────────────────
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(gt_xy[:, 1],   gt_xy[:, 0],   "g-",   lw=2.5, label="GT")
    if has_opt:
        ax_traj.plot(xy_opt[:, 1], xy_opt[:, 0], "b--",  lw=1.8,
                     label=f"Opt  ADE={ade_opt:.3f}m")
    ax_traj.plot(xy_mlp[:, 1],  xy_mlp[:, 0],  "m-.",  lw=1.8,
                 label=f"MLP  ADE={ade_mlp:.3f}m")
    ax_traj.plot(xy_def[:, 1],  xy_def[:, 0],  "r:",   lw=1.5,
                 label=f"Def  ADE={ade_def:.3f}m")
    ax_traj.scatter([0], [0], c="k", s=50, zorder=5, label="t0")

    all_xy = [gt_xy, xy_mlp, xy_def] + ([xy_opt] if has_opt else [])
    _set_traj_limits(ax_traj, *all_xy)
    ax_traj.set_xlabel("Y / lateral [m]")
    ax_traj.set_ylabel("X / longitudinal [m]")
    ax_traj.set_title("Trajectory (vehicle frame)\n(← left | right →)")
    ax_traj.legend(loc="best", fontsize=7)
    ax_traj.grid(True, ls=":", lw=0.5)

    # ── 스텝별 거리 오차 (중상) ──────────────────────
    ax_err = fig.add_subplot(gs[0, 1])
    if has_opt:
        err_opt = np.linalg.norm(xy_opt - gt_xy, axis=1)
        ax_err.plot(t_axis, err_opt, "b--", lw=1.5, label=f"Opt  {err_opt.mean():.3f}m")
    err_mlp = np.linalg.norm(xy_mlp - gt_xy, axis=1)
    err_def = np.linalg.norm(xy_def - gt_xy, axis=1)
    ax_err.plot(t_axis, err_mlp, "m-.",  lw=1.5, label=f"MLP  {err_mlp.mean():.3f}m")
    ax_err.plot(t_axis, err_def, "r:",   lw=1.3, label=f"Def  {err_def.mean():.3f}m")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("||pred - GT|| [m]")
    ax_err.set_title("Per-step displacement error")
    ax_err.legend(fontsize=7)
    ax_err.grid(True, ls=":", lw=0.5)
    ax_err.set_ylim(bottom=0)

    # ── Steering 제어 입력 (중중) ────────────────────
    ax_steer = fig.add_subplot(gs[1, 1])
    if has_opt:
        ax_steer.plot(t_axis, U_opt[:, 0], "b--",  lw=1.5, label="steer (opt)")
    ax_steer.plot(t_axis, U_mlp[:, 0], "m-.", lw=1.5, label="steer (mlp)")
    ax_steer.plot(t_axis, U_def[:, 0], "r:",  lw=1.3, label="steer (def)")
    ax_steer.axhline(0, c="k", lw=0.5, ls="--")
    ax_steer.set_xlabel("Time [s]")
    ax_steer.set_ylabel("Steering [rad]")
    ax_steer.set_title("Steering command")
    ax_steer.legend(fontsize=7)
    ax_steer.grid(True, ls=":", lw=0.5)

    # ── Acceleration 제어 입력 (중하) ────────────────
    ax_accel = fig.add_subplot(gs[2, 1])
    if has_opt:
        ax_accel.plot(t_axis, U_opt[:, 1], "b--",  lw=1.5, label="accel (opt)")
    ax_accel.plot(t_axis, U_mlp[:, 1], "m-.", lw=1.5, label="accel (mlp)")
    ax_accel.plot(t_axis, U_def[:, 1], "r:",  lw=1.3, label="accel (def)")
    ax_accel.axhline(0, c="k", lw=0.5, ls="--")
    ax_accel.set_xlabel("Time [s]")
    ax_accel.set_ylabel("Accel [m/s²]")
    ax_accel.set_title("Acceleration command")
    ax_accel.legend(fontsize=7)
    ax_accel.grid(True, ls=":", lw=0.5)

    # ── 가중치 막대 비교 (우열) ──────────────────────
    ax_w = fig.add_subplot(gs[:, 2])
    labels_w = ["long\npos", "lat\npos\n(fixed)", "heading", "steer\nrate", "accel\nrate"]
    x = np.arange(len(labels_w))

    n_bars  = 3 if has_opt else 2
    width   = 0.25 if has_opt else 0.35
    offsets = np.linspace(-width, width, n_bars)

    bar_items = []
    if has_opt:
        bar_items.append((w_opt,           "Opt",     "steelblue",    offsets[0]))
    bar_items.append(    (w_mlp,           "MLP",     "mediumpurple", offsets[-2 if has_opt else -1]))
    bar_items.append(    (WEIGHTS_DEFAULT, "Default", "tomato",       offsets[-1]))

    for w_vals, lbl, color, offset in bar_items:
        bars = ax_w.bar(x + offset, w_vals, width, label=lbl, color=color, alpha=0.8)
        for bar, v in zip(bars, w_vals):
            ax_w.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                      f"{v:.2f}", ha="center", fontsize=6, color=color)

    ax_w.set_xticks(x)
    ax_w.set_xticklabels(labels_w, fontsize=8)
    ax_w.set_ylabel("Weight value")
    ax_w.set_title("MPC cost weights")
    ax_w.legend(fontsize=8)
    ax_w.grid(True, axis="y", ls=":", lw=0.5)

    # ── 저장 ─────────────────────────────────────────
    out_path = out_dir / f"{name}_mlp_viz.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MLP 예측 가중치로 MPC 궤적 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file",      help="단일 h5 파일 경로")
    src.add_argument("--split-dir", help="split_dataset.py 결과 디렉토리 (test/ 서브폴더 사용)")

    parser.add_argument("--model-dir",  default="mlp_out",
                        help="best_model.pt / model_full.pt 가 있는 디렉토리 (default: mlp_out)")
    parser.add_argument("--n-samples",  type=int, default=0,
                        help="처리할 파일 수 — 랜덤 샘플링 (0=전체, default: 0)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="랜덤 샘플링 시드 (default: 42)")
    parser.add_argument("--out",        default="mlp_viz_out",
                        help="출력 디렉토리 (default: mlp_viz_out)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="이미 존재하는 PNG 덮어쓰기")
    args = parser.parse_args()

    out_dir   = Path(args.out)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 모델 로드 ─────────────────────────────────────
    model, _ = load_mlp(model_dir, device)

    # ── 파일 목록 수집 ────────────────────────────────
    if args.file:
        h5_files = [Path(args.file)]
    else:
        test_dir = Path(args.split_dir) / "test"
        if not test_dir.exists():
            print(f"[ERROR] {test_dir} 없음. split_dataset.py 를 먼저 실행하세요.")
            sys.exit(1)
        h5_files = sorted(test_dir.glob("*.h5"))
        if not h5_files:
            print(f"[ERROR] {test_dir} 에 .h5 파일 없음")
            sys.exit(1)

    if args.n_samples > 0 and args.n_samples < len(h5_files):
        total = len(h5_files)
        random.seed(args.seed)
        h5_files = random.sample(h5_files, args.n_samples)
        print(f"  랜덤 샘플링: {args.n_samples}개 / 전체 {total}개 (seed={args.seed})")

    print(f"\n처리 대상: {len(h5_files)}개")

    # ── 개별 PNG 저장 ─────────────────────────────────
    done = errors = 0
    for path in h5_files:
        out_path = out_dir / f"{path.stem}_mlp_viz.png"
        if out_path.exists() and not args.overwrite:
            print(f"  스킵 (이미 존재): {out_path.name}")
            done += 1
            continue
        try:
            saved = plot_mlp_single(path, model, device, out_dir)
            print(f"  저장: {saved.name}")
            done += 1
        except Exception as e:
            print(f"  [ERR] {path.name}: {e}")
            errors += 1

    print(f"\n완료: {done}개 저장, {errors}개 오류 → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
