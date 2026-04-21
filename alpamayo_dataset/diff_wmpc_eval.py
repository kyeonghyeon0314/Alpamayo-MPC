#!/usr/bin/env python3
"""
diff_wmpc_eval.py — Diff-WMPC 학습 결과 시각화

학습된 MLP(diff_wmpc_train.py 출력)로 테스트셋을 평가합니다.

비교 대상:
  GT       : 실제 주행 궤적 (녹색 실선)
  DiffWMPC : MLP 예측 가중치 MPC 궤적 (보라 일점쇄선)
  Default  : alpasim 기본 가중치 MPC 궤적 (빨강 점선)

출력:
  <out>/sample_<N>.png    — 샘플별 궤적 상세 플롯
  <out>/summary.png       — ADE 히스토그램 + 예측 가중치 분포

실행:
  py alpamayo_dataset/diff_wmpc_eval.py \\
      --model-dir result/diff_mpc/out \\
      --data-dir  alpamayo_dataset/data/prepare \\
      --out       result/diff_mpc/eval \\
      --n-samples 20
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

sys.path.insert(0, str(Path(__file__).parent))
from mpc import run_mpc, WEIGHTS_DEFAULT, W_LAT_FIXED, N, DT, NX, compute_x0
from viz_mpc_label import _set_traj_limits, _comfort_metrics
from model import load_mlp, LOG_MIN, LOG_MAX, W_NAMES, PREDICT_IDX


# ══════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════

def _to_weights5(log_theta: np.ndarray) -> np.ndarray:
    """log-scale (4,) → MPC용 5-weight (5,)."""
    log_theta = np.clip(log_theta.astype(np.float64), LOG_MIN, LOG_MAX)
    w4 = np.exp(log_theta)
    return np.array([w4[0], W_LAT_FIXED, w4[1], w4[2], w4[3]])


@torch.no_grad()
def predict_weights(model, cotend: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(cotend.astype(np.float32)).unsqueeze(0).to(device)
    return _to_weights5(model(x).squeeze(0).cpu().numpy())


# ══════════════════════════════════════════════════════
# 단일 샘플 플롯
# ══════════════════════════════════════════════════════

def plot_single(h5_path: Path, model, device, out_dir: Path, idx: int) -> dict | None:
    """h5 1개 평가 → PNG 저장 후 ADE/weights 딕셔너리 반환."""
    try:
        with h5py.File(h5_path, "r") as f:
            gt_ego    = f["gt/future_ego_states"][:]
            hist      = f["input/ego_history_ego_states"][:]
            hist_vel  = f["input/ego_history_vel"][:]
            hist_curv = f["input/ego_history_curv"][:]
            hist_quat = f["input/ego_history_quat_global"][:]
            cotend    = f["output/cotend_hidden_state"][:]

            if "gt_smooth" in f:
                gt_xy  = f["gt_smooth/future_xy"][:].astype(np.float64)
                gt_yaw = f["gt_smooth/future_yaw"][:].astype(np.float64)
            else:
                gt_xyz = f["gt/future_xyz"][:]
                gt_xy  = gt_xyz[:, :2].astype(np.float64)
                gt_yaw = gt_ego[:, 3].astype(np.float64)

    except Exception as e:
        print(f"  [로드 실패] {h5_path.name}: {e}")
        return None

    x0    = compute_x0(float(hist[-1, 2]), float(hist[-1, 4]),
                       hist_vel[-1], float(hist_curv[-1, 0]), hist_quat[-1])
    speed = float(hist[-1, 2])
    t_ax  = np.arange(1, N + 1) * DT

    w_pred = predict_weights(model, cotend, device)
    gt_xyz_run = np.column_stack([gt_xy[:N], np.zeros(N)])   # run_mpc 용 (N, 3)

    try:
        xy_dw,  _, _, _, U_dw,  ade_dw  = run_mpc(speed, gt_xyz_run, gt_yaw[:N], w_pred,        x0_full=x0)
        xy_def, _, _, _, U_def, ade_def = run_mpc(speed, gt_xyz_run, gt_yaw[:N], WEIGHTS_DEFAULT, x0_full=x0)
    except Exception as e:
        print(f"  [MPC 실패] {h5_path.name}: {e}")
        return None

    sr_dw, jr_dw   = _comfort_metrics(U_dw)
    sr_def, jr_def = _comfort_metrics(U_def)
    t_diff = t_ax[:-1]

    # ── Figure ───────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"{h5_path.name}   v0={speed:.1f} m/s\n"
        f"DiffWMPC  ADE={ade_dw:.4f}m  SR={sr_dw:.4f}  Jerk={jr_dw:.4f}   |   "
        f"Default   ADE={ade_def:.4f}m  SR={sr_def:.4f}  Jerk={jr_def:.4f}",
        fontsize=9,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.60, wspace=0.35)

    # trajectory
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(gt_xy[:N, 1], gt_xy[:N, 0], "g-",  lw=2.5, label="GT")
    ax_traj.plot(xy_dw[:,  1], xy_dw[:,  0], "m-.", lw=1.8, label=f"DiffWMPC  {ade_dw:.4f}m")
    ax_traj.plot(xy_def[:, 1], xy_def[:, 0], "r:",  lw=1.5, label=f"Default   {ade_def:.4f}m")
    ax_traj.scatter([0], [0], c="k", s=50, zorder=5, label="t0")
    _set_traj_limits(ax_traj, gt_xy[:N], xy_dw, xy_def)
    ax_traj.set_xlabel("Y / lateral [m]")
    ax_traj.set_ylabel("X / longitudinal [m]")
    ax_traj.set_title("Trajectory (vehicle frame)")
    ax_traj.legend(loc="best", fontsize=7)
    ax_traj.grid(True, ls=":", lw=0.5)

    # ADE per step
    ax_err = fig.add_subplot(gs[0, 1:3])
    err_dw  = np.linalg.norm(xy_dw  - gt_xy[:N], axis=1)
    err_def = np.linalg.norm(xy_def - gt_xy[:N], axis=1)
    ax_err.plot(t_ax, err_dw,  "m-.", lw=1.5, label=f"DiffWMPC  {err_dw.mean():.4f}m")
    ax_err.plot(t_ax, err_def, "r:",  lw=1.3, label=f"Default   {err_def.mean():.4f}m")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("||pred - GT|| [m]")
    ax_err.set_title("Per-step displacement error")
    ax_err.legend(fontsize=7); ax_err.grid(True, ls=":", lw=0.5); ax_err.set_ylim(bottom=0)

    # Steering
    ax_s = fig.add_subplot(gs[1, 1])
    ax_s.plot(t_ax, U_dw[:, 0],  "m-.", lw=1.5, label="DiffWMPC")
    ax_s.plot(t_ax, U_def[:, 0], "r:",  lw=1.3, label="Default")
    ax_s.axhline(0, c="k", lw=0.5, ls="--")
    ax_s.set_xlabel("Time [s]"); ax_s.set_ylabel("Steer [rad]")
    ax_s.set_title("Steering command"); ax_s.legend(fontsize=7); ax_s.grid(True, ls=":", lw=0.5)

    # Accel
    ax_a = fig.add_subplot(gs[1, 2])
    ax_a.plot(t_ax, U_dw[:, 1],  "m-.", lw=1.5, label="DiffWMPC")
    ax_a.plot(t_ax, U_def[:, 1], "r:",  lw=1.3, label="Default")
    ax_a.axhline(0, c="k", lw=0.5, ls="--")
    ax_a.set_xlabel("Time [s]"); ax_a.set_ylabel("Accel [m/s²]")
    ax_a.set_title("Acceleration command"); ax_a.legend(fontsize=7); ax_a.grid(True, ls=":", lw=0.5)

    # Steer rate
    ax_sr = fig.add_subplot(gs[2, 1])
    ax_sr.plot(t_diff, np.abs(np.diff(U_dw[:, 0])),  "m-.", lw=1.5, label=f"DiffWMPC  rms={sr_dw:.4f}")
    ax_sr.plot(t_diff, np.abs(np.diff(U_def[:, 0])), "r:",  lw=1.3, label=f"Default   rms={sr_def:.4f}")
    ax_sr.set_xlabel("Time [s]"); ax_sr.set_ylabel("|Δsteer| [rad/step]")
    ax_sr.set_title("Steering rate"); ax_sr.legend(fontsize=7); ax_sr.grid(True, ls=":", lw=0.5)
    ax_sr.set_ylim(bottom=0)

    # Jerk
    ax_j = fig.add_subplot(gs[2, 2])
    ax_j.plot(t_diff, np.abs(np.diff(U_dw[:, 1])),  "m-.", lw=1.5, label=f"DiffWMPC  rms={jr_dw:.4f}")
    ax_j.plot(t_diff, np.abs(np.diff(U_def[:, 1])), "r:",  lw=1.3, label=f"Default   rms={jr_def:.4f}")
    ax_j.set_xlabel("Time [s]"); ax_j.set_ylabel("|Δaccel| [(m/s²)/step]")
    ax_j.set_title("Jerk"); ax_j.legend(fontsize=7); ax_j.grid(True, ls=":", lw=0.5)
    ax_j.set_ylim(bottom=0)

    # Weight bars
    ax_w = fig.add_subplot(gs[3, 1:3])
    labels_w = ["long pos", "lat pos\n(fixed)", "heading", "steer rate", "accel rate"]
    x = np.arange(5)
    width = 0.35
    ax_w.bar(x - width/2, w_pred,        width, label="DiffWMPC", color="mediumpurple", alpha=0.85)
    ax_w.bar(x + width/2, WEIGHTS_DEFAULT, width, label="Default",  color="tomato",       alpha=0.85)
    for xi, v in zip(x, w_pred):
        ax_w.text(xi - width/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=6, color="purple")
    for xi, v in zip(x, WEIGHTS_DEFAULT):
        ax_w.text(xi + width/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=6, color="tomato")
    ax_w.set_xticks(x); ax_w.set_xticklabels(labels_w, fontsize=8)
    ax_w.set_ylabel("Weight"); ax_w.set_title("Predicted MPC weights")
    ax_w.legend(fontsize=8); ax_w.grid(True, axis="y", ls=":", lw=0.5)

    out_path = out_dir / f"sample_{idx:04d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{idx:04d}] ADE={ade_dw:.4f}m  (def={ade_def:.4f}m)  → {out_path.name}")

    return {"ade_dw": ade_dw, "ade_def": ade_def, "weights": w_pred}


# ══════════════════════════════════════════════════════
# 요약 플롯
# ══════════════════════════════════════════════════════

def save_summary(results: list[dict], out_dir: Path) -> None:
    ade_dw  = np.array([r["ade_dw"]  for r in results])
    ade_def = np.array([r["ade_def"] for r in results])
    weights = np.stack([r["weights"] for r in results])   # (N, 5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Diff-WMPC 테스트 요약  (n={len(results)})\n"
        f"DiffWMPC  ADE: mean={ade_dw.mean():.4f}m  median={np.median(ade_dw):.4f}m   |   "
        f"Default   ADE: mean={ade_def.mean():.4f}m  median={np.median(ade_def):.4f}m",
        fontsize=10,
    )

    # ADE 히스토그램
    ax = axes[0]
    bins = np.linspace(0, max(ade_dw.max(), ade_def.max()) * 1.05, 40)
    ax.hist(ade_dw,  bins=bins, alpha=0.7, color="mediumpurple", label="DiffWMPC")
    ax.hist(ade_def, bins=bins, alpha=0.5, color="tomato",       label="Default")
    ax.axvline(ade_dw.mean(),  color="purple", ls="--", lw=1.5, label=f"DW mean {ade_dw.mean():.4f}")
    ax.axvline(ade_def.mean(), color="red",    ls="--", lw=1.5, label=f"Def mean {ade_def.mean():.4f}")
    ax.set_xlabel("ADE [m]"); ax.set_ylabel("Count")
    ax.set_title("ADE 분포"); ax.legend(fontsize=8); ax.grid(True, ls=":", lw=0.5)

    # DiffWMPC vs Default scatter
    ax = axes[1]
    lim = max(ade_dw.max(), ade_def.max()) * 1.05
    ax.scatter(ade_def, ade_dw, s=10, alpha=0.4, color="mediumpurple")
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    improve = (ade_dw < ade_def).mean() * 100
    ax.set_xlabel("Default ADE [m]"); ax.set_ylabel("DiffWMPC ADE [m]")
    ax.set_title(f"DiffWMPC < Default: {improve:.1f}%")
    ax.legend(fontsize=8); ax.grid(True, ls=":", lw=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # 예측 가중치 박스플롯
    ax = axes[2]
    labels_w = ["long", "lat\n(fixed)", "heading", "steer_r", "accel_r"]
    ax.boxplot(weights, labels=labels_w, patch_artist=True,
               boxprops=dict(facecolor="mediumpurple", alpha=0.6))
    for xi, dv in enumerate(WEIGHTS_DEFAULT):
        ax.scatter(xi + 1, dv, marker="D", s=60, color="tomato",
                   zorder=5, label="Default" if xi == 0 else "")
    ax.set_ylabel("Weight value")
    ax.set_title("예측 가중치 분포 (◆=Default)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", ls=":", lw=0.5)

    fig.tight_layout()
    path = out_dir / "summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n요약 저장: {path}")
    print(f"  DiffWMPC  ADE: mean={ade_dw.mean():.4f}  median={np.median(ade_dw):.4f}  "
          f"p90={np.percentile(ade_dw,90):.4f}")
    print(f"  Default   ADE: mean={ade_def.mean():.4f}  median={np.median(ade_def):.4f}  "
          f"p90={np.percentile(ade_def,90):.4f}")
    print(f"  DiffWMPC < Default: {improve:.1f}%  ({int(improve/100*len(results))}/{len(results)} 샘플)")


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diff-WMPC 학습 결과 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-dir", required=True,
                        help="diff_wmpc_train.py 출력 디렉토리 (model_full.pt 위치)")
    parser.add_argument("--data-dir",  required=True,
                        help="prepare/ 베이스 디렉토리 (test/ 서브폴더 사용)")
    parser.add_argument("--out",       default="diff_wmpc_eval_out")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="궤적 플롯할 샘플 수 (0=전체, default: 20)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--all-summary", action="store_true",
                        help="요약 통계는 전체 테스트셋으로 계산 (--n-samples와 무관)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_mlp(Path(args.model_dir), device)
    model.eval()

    test_dir = Path(args.data_dir) / "test"
    h5_files = sorted(test_dir.glob("*.h5"))
    if not h5_files:
        print(f"[ERROR] {test_dir} 에 .h5 파일 없음"); sys.exit(1)
    print(f"테스트 파일: {len(h5_files)}개")

    # 샘플링 (궤적 플롯용)
    plot_files = h5_files
    if args.n_samples > 0 and args.n_samples < len(h5_files):
        random.seed(args.seed)
        plot_files = random.sample(h5_files, args.n_samples)
        print(f"궤적 플롯: {args.n_samples}개 랜덤 샘플 (seed={args.seed})")

    # 궤적 플롯
    print("\n--- 샘플 궤적 플롯 ---")
    plot_results = []
    for i, p in enumerate(sorted(plot_files)):
        r = plot_single(p, model, device, out_dir, i + 1)
        if r:
            plot_results.append(r)

    # 요약: --all-summary면 전체, 아니면 플롯한 샘플
    summary_files = h5_files if args.all_summary else plot_files
    if args.all_summary and len(summary_files) != len(plot_files):
        print(f"\n--- 전체 테스트셋 통계 계산 ({len(summary_files)}개) ---")
        all_results = []
        for p in summary_files:
            with torch.no_grad():
                try:
                    with h5py.File(p, "r") as f:
                        cotend    = f["output/cotend_hidden_state"][:]
                        gt_ego    = f["gt/future_ego_states"][:]
                        hist      = f["input/ego_history_ego_states"][:]
                        hist_vel  = f["input/ego_history_vel"][:]
                        hist_curv = f["input/ego_history_curv"][:]
                        hist_quat = f["input/ego_history_quat_global"][:]
                        gt_xy  = (f["gt_smooth/future_xy"][:] if "gt_smooth" in f
                                  else f["gt/future_xyz"][:, :2]).astype(np.float64)
                        gt_yaw = (f["gt_smooth/future_yaw"][:] if "gt_smooth" in f
                                  else gt_ego[:, 3]).astype(np.float64)
                    x0    = compute_x0(float(hist[-1,2]), float(hist[-1,4]),
                                       hist_vel[-1], float(hist_curv[-1,0]), hist_quat[-1])
                    speed = float(hist[-1, 2])
                    gt_xyz_run = np.column_stack([gt_xy[:N], np.zeros(N)])
                    w_pred = predict_weights(model, cotend, device)
                    _, _, _, _, _, ade_dw  = run_mpc(speed, gt_xyz_run, gt_yaw[:N], w_pred,         x0_full=x0)
                    _, _, _, _, _, ade_def = run_mpc(speed, gt_xyz_run, gt_yaw[:N], WEIGHTS_DEFAULT, x0_full=x0)
                    all_results.append({"ade_dw": ade_dw, "ade_def": ade_def, "weights": w_pred})
                except Exception:
                    pass
        save_summary(all_results, out_dir)
    else:
        if plot_results:
            save_summary(plot_results, out_dir)

    print(f"\n완료 → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
