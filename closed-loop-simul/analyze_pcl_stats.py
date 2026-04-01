#!/usr/bin/env python3
"""
PCL 시뮬레이션 결과 통계 분석기

run_pcl_sim.py 가 생성한 .npz 파일들을 폴더 단위로 읽어
논문 수준의 통계 지표를 산출하고 시각화합니다.

────────────────────────────────────────────────────────────────
산출 지표 (클립당):
  [궤적 추종 정확도]
    ADE, FDE, RMSE, P95 (95th percentile error)
  [제어 안정성 / 승차감]
    Steering Rate RMS, Jerk RMS, Total Control Effort
  [강건성]
    Diverged (FDE > 10m 또는 Max error > 10m)

출력 파일 (--out-dir 지정 폴더):
  stats_summary.csv           — 전체 클립 DataFrame
  fig_accuracy_boxplot.png    — ADE / FDE / RMSE / P95 Boxplot (2×2)
  fig_stability_boxplot.png   — Steering Rate RMS / Jerk RMS / Effort Boxplot
  fig_scatter_ade.png         — Default ADE vs MLP ADE 산점도
  fig_survival.png            — 발산율 바 차트 (클립 수 / %)
  (터미널)                    — Wilcoxon signed-rank 검정 결과

실행 예시:
  python closed-loop-simul/analyze_pcl_stats.py \\
      --npz-dir  closed-loop-simul/pcl_sim_out \\
      --out-dir  closed-loop-simul/pcl_stats \\
      --div-thr  10.0
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


# ══════════════════════════════════════════════════════════════
# 클립 단위 지표 계산
# ══════════════════════════════════════════════════════════════

def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def compute_clip_stats(npz_path: Path, div_thr: float = 10.0) -> dict:
    """
    단일 .npz 파일에서 통계 지표를 계산한다.

    Parameters
    ----------
    npz_path : Path
    div_thr  : float
        발산 판정 임계값 [m]. FDE 또는 Max error 초과 시 diverged=True.

    Returns
    -------
    dict — clip_id 포함 모든 지표
    """
    d = np.load(npz_path)

    errors_mlp = d["errors_mlp"]   # (N,)
    errors_def = d["errors_def"]   # (N,)
    u_mlp      = d["u_mlp_hist"]   # (N, 2) — [steer, accel]
    u_def      = d["u_def_hist"]   # (N, 2)

    # step 0 제외 (항상 0)
    e_mlp = errors_mlp[1:]
    e_def = errors_def[1:]

    # ── 궤적 추종 정확도 ──────────────────────────────────────
    ade_mlp  = float(np.mean(e_mlp))
    ade_def  = float(np.mean(e_def))
    fde_mlp  = float(errors_mlp[-1])
    fde_def  = float(errors_def[-1])
    rmse_mlp = _rms(e_mlp)
    rmse_def = _rms(e_def)
    p95_mlp  = float(np.percentile(e_mlp, 95))
    p95_def  = float(np.percentile(e_def, 95))

    # ── 제어 안정성 ───────────────────────────────────────────
    steer_rate_mlp = np.diff(u_mlp[:, 0])
    steer_rate_def = np.diff(u_def[:, 0])
    jerk_mlp       = np.diff(u_mlp[:, 1])
    jerk_def       = np.diff(u_def[:, 1])

    sr_rms_mlp    = _rms(steer_rate_mlp)
    sr_rms_def    = _rms(steer_rate_def)
    jerk_rms_mlp  = _rms(jerk_mlp)
    jerk_rms_def  = _rms(jerk_def)
    effort_mlp    = float(np.sum(u_mlp ** 2))
    effort_def    = float(np.sum(u_def ** 2))

    # ── 강건성 ────────────────────────────────────────────────
    diverged_mlp = bool(fde_mlp > div_thr or float(np.max(e_mlp)) > div_thr)
    diverged_def = bool(fde_def > div_thr or float(np.max(e_def)) > div_thr)

    # ── 개선율 ────────────────────────────────────────────────
    gain_ade  = (ade_def  - ade_mlp)  / ade_def  * 100 if ade_def  > 0 else 0.0
    gain_rmse = (rmse_def - rmse_mlp) / rmse_def * 100 if rmse_def > 0 else 0.0

    return {
        "clip_id":       npz_path.stem,
        # 정확도
        "ade_mlp":       ade_mlp,   "ade_def":       ade_def,
        "fde_mlp":       fde_mlp,   "fde_def":       fde_def,
        "rmse_mlp":      rmse_mlp,  "rmse_def":      rmse_def,
        "p95_mlp":       p95_mlp,   "p95_def":       p95_def,
        # 안정성
        "sr_rms_mlp":    sr_rms_mlp,   "sr_rms_def":    sr_rms_def,
        "jerk_rms_mlp":  jerk_rms_mlp, "jerk_rms_def":  jerk_rms_def,
        "effort_mlp":    effort_mlp,   "effort_def":    effort_def,
        # 강건성
        "diverged_mlp":  diverged_mlp,
        "diverged_def":  diverged_def,
        # 요약
        "gain_ade_pct":  gain_ade,
        "gain_rmse_pct": gain_rmse,
        "n_frames":      len(errors_mlp),
    }


# ══════════════════════════════════════════════════════════════
# 통계 검정
# ══════════════════════════════════════════════════════════════

def print_statistical_tests(df: pd.DataFrame) -> None:
    """Wilcoxon signed-rank 검정 결과를 터미널에 출력."""
    metrics = [
        ("ADE",          "ade_mlp",      "ade_def"),
        ("FDE",          "fde_mlp",      "fde_def"),
        ("RMSE",         "rmse_mlp",     "rmse_def"),
        ("P95 error",    "p95_mlp",      "p95_def"),
        ("Steer Rate RMS","sr_rms_mlp",  "sr_rms_def"),
        ("Jerk RMS",     "jerk_rms_mlp", "jerk_rms_def"),
        ("Control Effort","effort_mlp",  "effort_def"),
    ]

    print("\n" + "=" * 68)
    print(f"  {'Metric':<20}  {'MLP mean':>10}  {'Def mean':>10}  "
          f"{'Δ%':>8}  {'p-value':>10}  {'sig':>5}")
    print("=" * 68)
    for name, col_m, col_d in metrics:
        vals_m = df[col_m].values
        vals_d = df[col_d].values
        diff   = vals_d - vals_m        # 양수 = MLP가 더 나음
        try:
            stat, pval = scipy.stats.wilcoxon(diff)
        except ValueError:
            pval = float("nan")
        delta_pct = float(np.mean(diff) / np.mean(vals_d) * 100) if np.mean(vals_d) > 0 else 0.0
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
        print(f"  {name:<20}  {np.mean(vals_m):>10.4f}  {np.mean(vals_d):>10.4f}  "
              f"{delta_pct:>+7.1f}%  {pval:>10.4f}  {sig:>5}")
    print("=" * 68)
    print("  * p<0.05   ** p<0.01   *** p<0.001   (Wilcoxon signed-rank)\n")


# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════

_C_MLP = "mediumpurple"
_C_DEF = "tomato"


def _boxplot_pair(ax: plt.Axes, data_mlp: np.ndarray, data_def: np.ndarray,
                  ylabel: str, title: str, lower_better: bool = True) -> None:
    """MLP vs Default Boxplot + 개별 점 오버레이."""
    bp = ax.boxplot(
        [data_mlp, data_def],
        labels=["MLP-PCL", "Def-PCL"],
        patch_artist=True,
        medianprops=dict(color="white", lw=2.5),
        widths=0.45,
    )
    colors = [_C_MLP, _C_DEF]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for whisker in bp["whiskers"]:
        whisker.set(color="gray", lw=1.2, ls="--")
    for cap in bp["caps"]:
        cap.set(color="gray", lw=1.5)
    for flier in bp["fliers"]:
        flier.set(marker="o", color="gray", alpha=0.4, markersize=4)

    # 개별 점 (jitter)
    for i, (data, c) in enumerate(zip([data_mlp, data_def], colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(data))
        ax.scatter(np.full(len(data), i) + jitter, data,
                   alpha=0.35, s=18, color=c, zorder=3)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", ls=":", lw=0.5)
    note = "(lower = better)" if lower_better else "(higher = better)"
    ax.text(0.98, 0.97, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray")


def plot_accuracy_boxplot(df: pd.DataFrame, out_dir: Path) -> Path:
    """Fig A: ADE / FDE / RMSE / P95 Boxplot 2×2."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    n = len(df)
    fig.suptitle(
        f"Trajectory Tracking Accuracy  (N={n} clips)\n"
        "MLP-PCL  vs  Default-PCL",
        fontsize=13,
    )
    pairs = [
        ("ade_mlp",  "ade_def",  "ADE [m]",  "Average Displacement Error"),
        ("fde_mlp",  "fde_def",  "FDE [m]",  "Final Displacement Error"),
        ("rmse_mlp", "rmse_def", "RMSE [m]", "Root Mean Square Error"),
        ("p95_mlp",  "p95_def",  "P95 [m]",  "95th-Percentile Error"),
    ]
    for ax, (cm, cd, ylabel, title) in zip(axes.flat, pairs):
        _boxplot_pair(ax, df[cm].values, df[cd].values, ylabel, title)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "fig_accuracy_boxplot.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_stability_boxplot(df: pd.DataFrame, out_dir: Path) -> Path:
    """Fig B: Steering Rate RMS / Jerk RMS / Control Effort Boxplot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    n = len(df)
    fig.suptitle(
        f"Control Stability & Comfort  (N={n} clips)\n"
        "MLP-PCL  vs  Default-PCL",
        fontsize=13,
    )
    pairs = [
        ("sr_rms_mlp",   "sr_rms_def",   "[rad/step]",  "Steering Rate RMS\n(jitter indicator)"),
        ("jerk_rms_mlp", "jerk_rms_def", "[(m/s²)/step]","Jerk RMS\n(ride comfort indicator)"),
        ("effort_mlp",   "effort_def",   "Σu² [arb.]",   "Total Control Effort"),
    ]
    for ax, (cm, cd, ylabel, title) in zip(axes, pairs):
        _boxplot_pair(ax, df[cm].values, df[cd].values, ylabel, title)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / "fig_stability_boxplot.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scatter_ade(df: pd.DataFrame, out_dir: Path) -> Path:
    """Fig C: Default ADE vs MLP ADE 산점도 (대각선 기준)."""
    fig, ax = plt.subplots(figsize=(9, 9))

    mlp_v = df["ade_mlp"].values
    def_v = df["ade_def"].values

    n_mlp_wins = int((mlp_v < def_v).sum())
    n_def_wins = int((mlp_v > def_v).sum())
    n_tied     = len(df) - n_mlp_wins - n_def_wins

    ax.scatter(def_v, mlp_v, alpha=0.55, s=40,
               c=np.where(mlp_v < def_v, _C_MLP, _C_DEF),
               edgecolors="none")

    # 대각선 (y=x)
    lim = max(def_v.max(), mlp_v.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.5, label="y = x  (equal)")

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Default ADE [m]", fontsize=12)
    ax.set_ylabel("MLP ADE [m]",     fontsize=12)
    ax.set_title(
        f"Per-Clip ADE: Default vs MLP  (N={len(df)})\n"
        f"Purple = MLP wins ({n_mlp_wins})  |  "
        f"Red = Default wins ({n_def_wins})  |  "
        f"Tied = {n_tied}",
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", lw=0.5)

    # 텍스트 요약
    ax.text(0.03, 0.97,
            f"MLP wins: {n_mlp_wins / len(df) * 100:.1f}%\n"
            f"Def wins: {n_def_wins / len(df) * 100:.1f}%",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(fc="white", ec="lightgray", alpha=0.85, pad=4))

    out_path = out_dir / "fig_scatter_ade.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_survival(df: pd.DataFrame, out_dir: Path, div_thr: float) -> Path:
    """Fig D: 발산율 바 차트."""
    n = len(df)
    n_div_mlp = int(df["diverged_mlp"].sum())
    n_div_def = int(df["diverged_def"].sum())
    n_ok_mlp  = n - n_div_mlp
    n_ok_def  = n - n_div_def

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"Robustness & Survival Rate  (N={n} clips,  divergence threshold={div_thr}m)",
        fontsize=13,
    )

    # ── 바 차트 (클립 수) ──────────────────────────────────────
    ax = axes[0]
    x = np.array([0, 1])
    bw = 0.35
    ax.bar(x - bw / 2, [n_ok_mlp,  n_ok_def],  bw, color=[_C_MLP, _C_DEF],
           alpha=0.85, edgecolor="white", label="Stable")
    ax.bar(x + bw / 2, [n_div_mlp, n_div_def], bw, color=[_C_MLP, _C_DEF],
           alpha=0.45, edgecolor="white", hatch="//", label="Diverged")
    ax.set_xticks(x)
    ax.set_xticklabels(["MLP-PCL", "Def-PCL"], fontsize=11)
    ax.set_ylabel("Number of clips", fontsize=11)
    ax.set_title("Clip Count", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", ls=":", lw=0.5)
    for xi, (ok, div) in zip(x, [(n_ok_mlp, n_div_mlp), (n_ok_def, n_div_def)]):
        ax.text(xi - bw / 2, ok  + 0.3, str(ok),  ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + bw / 2, div + 0.3, str(div), ha="center", fontsize=9, fontweight="bold")

    # ── 생존율 바 차트 (%) ─────────────────────────────────────
    ax = axes[1]
    surv_mlp = n_ok_mlp  / n * 100
    surv_def = n_ok_def  / n * 100
    bars = ax.bar(["MLP-PCL", "Def-PCL"], [surv_mlp, surv_def],
                  color=[_C_MLP, _C_DEF], alpha=0.85, edgecolor="white", width=0.45)
    for bar, val in zip(bars, [surv_mlp, surv_def]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Survival rate [%]", fontsize=11)
    ax.set_title("Survival Rate  (higher = better)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", ls=":", lw=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / "fig_survival.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gain_histogram(df: pd.DataFrame, out_dir: Path) -> Path:
    """Fig E: 클립별 ADE 개선율 히스토그램."""
    gains = df["gain_ade_pct"].values
    n_pos = int((gains > 0).sum())
    n_neg = int((gains < 0).sum())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gains[gains >= 0], bins=30, color=_C_MLP, alpha=0.75,
            label=f"MLP better  (n={n_pos})")
    ax.hist(gains[gains < 0],  bins=30, color=_C_DEF, alpha=0.75,
            label=f"Default better  (n={n_neg})")
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.axvline(float(np.mean(gains)), color="steelblue", lw=2.0, ls="-",
               label=f"Mean = {np.mean(gains):+.1f}%")
    ax.set_xlabel("ADE Improvement (%)  [positive = MLP better]", fontsize=11)
    ax.set_ylabel("Clip count", fontsize=11)
    ax.set_title(
        f"Per-Clip ADE Gain Distribution  (N={len(df)})\n"
        f"Mean gain = {np.mean(gains):+.1f}%  |  Median = {np.median(gains):+.1f}%",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", lw=0.5)

    out_path = out_dir / "fig_gain_histogram.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="PCL 시뮬레이션 .npz 배치 통계 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--npz-dir",  required=True,
                   help=".npz 파일들이 저장된 디렉토리")
    p.add_argument("--out-dir",  default=None,
                   help="통계 결과 저장 디렉토리 (기본: --npz-dir)")
    p.add_argument("--div-thr",  type=float, default=10.0,
                   help="발산 판정 임계값 [m] (기본: 10.0)")
    args = p.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir) if args.out_dir else npz_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("*_pcl_sim.npz"))
    if not npz_files:
        print(f"[ERROR] .npz 파일을 찾을 수 없습니다: {npz_dir}/*_pcl_sim.npz")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  NPZ 파일 수  : {len(npz_files)}")
    print(f"  발산 임계값  : {args.div_thr} m")
    print(f"  출력 디렉토리: {out_dir}")
    print(f"{'='*60}")

    # ── 클립별 지표 계산 ──────────────────────────────────────
    rows = []
    for i, fp in enumerate(npz_files, 1):
        try:
            row = compute_clip_stats(fp, div_thr=args.div_thr)
            rows.append(row)
        except Exception as exc:
            print(f"  [SKIP] {fp.name}: {exc}")
        if i % 50 == 0 or i == len(npz_files):
            print(f"  처리 중 {i}/{len(npz_files)} ...")

    if not rows:
        print("[ERROR] 유효한 클립이 없습니다.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\n  유효 클립 수: {n}")

    # ── 통계 검정 ─────────────────────────────────────────────
    print_statistical_tests(df)

    # ── 요약 출력 ─────────────────────────────────────────────
    print("  [발산율]")
    print(f"    MLP-PCL  : {df['diverged_mlp'].sum():3d} / {n}  "
          f"({df['diverged_mlp'].mean() * 100:.1f}%)")
    print(f"    Def-PCL  : {df['diverged_def'].sum():3d} / {n}  "
          f"({df['diverged_def'].mean() * 100:.1f}%)")
    print(f"\n  [ADE 개선율]  mean={df['gain_ade_pct'].mean():+.2f}%  "
          f"median={df['gain_ade_pct'].median():+.2f}%")

    # ── CSV 저장 ──────────────────────────────────────────────
    csv_path = out_dir / "stats_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  CSV 저장: {csv_path}")

    # ── 시각화 저장 ───────────────────────────────────────────
    print("\n  시각화 생성 중...")
    figs = [
        plot_accuracy_boxplot(df, out_dir),
        plot_stability_boxplot(df, out_dir),
        plot_scatter_ade(df, out_dir),
        plot_survival(df, out_dir, args.div_thr),
        plot_gain_histogram(df, out_dir),
    ]
    for fp in figs:
        print(f"  → {fp.name}")

    print(f"\n{'='*60}")
    print(f"  완료  ({n} clips)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
