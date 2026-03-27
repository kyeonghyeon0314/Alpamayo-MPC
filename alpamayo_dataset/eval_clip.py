#!/usr/bin/env python3
"""
Clip-level MLP+MPC continuous validation (Plan A)

Selects clips with multiple timestamps from the test set and compares
MLP vs Default vs Opt(IOC) trajectory quality across the full clip timeline.

Goal:
  "Is MPC with MLP-predicted weights consistently better than Default across the clip?"
  Track how MLP weight predictions adapt over time within a clip.

Output (one PNG per clip):
  Top   : ADE timeline (MLP / Default / Opt) + improvement bars vs Default
  Middle: 5 weight evolution plots (MLP prediction vs Default baseline vs Opt)
  Bottom: ONE stitched global-frame trajectory
           - All MPC predictions transformed into a shared coordinate frame
           - Each t0 segment color-coded; GT shown as single continuous path

Run:
  python alpamayo_dataset/eval_clip.py \\
      --model-dir alpamayo_dataset/mlp_out10 \\
      --split-dir alpamayo_dataset/data/split \\
      --out       alpamayo_dataset/eval_clip_out \\
      --n-clips   10
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mpc import (run_mpc, step_dynamics, WEIGHTS_DEFAULT, W_LAT_FIXED,
                 N, DT, IX, IY, IYAW, V_MIN_LIN)
from model import load_mlp, CotendMLP

_PRED_NAMES = ["long_pos", "heading", "steer_rate", "accel_rate"]
_PRED_WIDX  = [0, 2, 3, 4]   # indices in weights5 (lat=1 fixed)

# colors per timestamp slot — distinct, no green (avoids clash with GT)
_T_COLORS = ["#D62728", "#1F77B4", "#FF7F00", "#9467BD", "#8C564B"]


# ══════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════

def _to_weights5(log_theta: np.ndarray) -> np.ndarray:
    w4 = np.exp(log_theta.astype(np.float64))
    return np.array([w4[0], W_LAT_FIXED, w4[1], w4[2], w4[3]])


@torch.no_grad()
def _predict_weights(model: CotendMLP, cotend: np.ndarray,
                     device: torch.device) -> np.ndarray:
    x = torch.from_numpy(cotend.astype(np.float32)).unsqueeze(0).to(device)
    return _to_weights5(model(x).squeeze(0).cpu().numpy())


def pseudo_closed_loop(
    v0: float,
    gt_xyz: np.ndarray,
    gt_yaw: np.ndarray,
    weights5: np.ndarray,
    n: int = N,
) -> tuple[np.ndarray, float, float]:
    """
    N-step Receding Horizon Control pseudo-closed-loop 시뮬레이션.

    매 스텝 MPC를 새로 풀되, 차량 출발점을 GT 위치가 아닌
    직전 스텝 동역학 적분 결과로 설정한다. 오차가 누적되어도
    MPC가 이를 보정하는지(안정성)를 검증하기 위한 지표.

    Args:
        v0:       초기 속도 [m/s]
        gt_xyz:   GT 미래 궤적 (≥N, 3) — t0 local frame
        gt_yaw:   GT yaw (≥N,) — t0 local frame
        weights5: MPC 가중치 [long, lat, hdg, steer_r, accel_r]
        n:        Receding horizon 스텝 수 (기본값 N=20)

    Returns:
        xy_pcl (n, 2): 시뮬레이션 궤적 (t0 local frame)
        ade_pcl:       GT 대비 평균 위치 오차 [m]
        fde_pcl:       GT 대비 최종 위치 오차 [m]  ← 발산 지표
    """
    x_sim = np.array([0., 0., 0., max(v0, V_MIN_LIN), 0., 0., 0., 0.])
    n_gt  = len(gt_xyz)
    xy_pcl: list[np.ndarray] = []

    for i in range(n):
        # ① GT reference 슬라이싱 (t0 local frame)
        ref_end    = min(i + n, n_gt)
        gt_ref_xy  = gt_xyz[i:ref_end, :2].copy()
        gt_ref_yaw = gt_yaw[i:ref_end].copy()
        if len(gt_ref_xy) < n:                        # 끝 부분 패딩
            pad        = n - len(gt_ref_xy)
            gt_ref_xy  = np.vstack([gt_ref_xy,  np.tile(gt_ref_xy[-1],  (pad, 1))])
            gt_ref_yaw = np.concatenate([gt_ref_yaw, np.tile(gt_ref_yaw[-1], pad)])

        # ② GT reference → x_sim 기준 local frame 변환
        sx, sy, syaw = x_sim[IX], x_sim[IY], x_sim[IYAW]
        c, s         = np.cos(syaw), np.sin(syaw)
        R_inv        = np.array([[c, s], [-s, c]])
        gt_local_xy  = (gt_ref_xy - np.array([sx, sy])) @ R_inv.T
        gt_local_yaw = gt_ref_yaw - syaw
        gt_local     = np.column_stack([gt_local_xy, np.zeros(n)])  # z=0

        # ③ velocity/actuator 상태 보존, position만 0으로 정규화
        x0_local      = x_sim.copy()
        x0_local[IX]  = x0_local[IY] = x0_local[IYAW] = 0.0

        # ④ MPC 풀기 — 첫 번째 제어 입력만 적용 (Receding Horizon 원리)
        _, U_opt, _ = run_mpc(v0, gt_local, gt_local_yaw, weights5,
                               n=n, x0_full=x0_local)
        u0 = U_opt[0]

        # ⑤ 단일 스텝 동역학 전진 (x_sim은 t0 local frame 절대 좌표 유지)
        x_sim = step_dynamics(x_sim, u0)
        xy_pcl.append(x_sim[[IX, IY]])

    xy_pcl_arr = np.array(xy_pcl)                       # (n, 2), t0 local frame
    errors     = np.linalg.norm(xy_pcl_arr - gt_xyz[:n, :2], axis=1)
    ade_pcl    = float(np.mean(errors))
    fde_pcl    = float(errors[-1])
    return xy_pcl_arr, ade_pcl, fde_pcl


def _group_clips(test_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """clip_id -> [(t0_us, path), ...] sorted by t0_us."""
    clips: dict[str, list] = defaultdict(list)
    for h5_path in sorted(test_dir.glob("*.h5")):
        parts = h5_path.stem.split("__")
        if len(parts) == 2:
            try:
                clips[parts[0]].append((int(parts[1]), h5_path))
            except ValueError:
                pass
    for v in clips.values():
        v.sort()
    return dict(clips)


# ── Global-frame coordinate transforms ────────────────

def _local_to_global(xy_local: np.ndarray,
                     origin: np.ndarray, yaw: float) -> np.ndarray:
    """
    Transform trajectory from local vehicle frame to global frame.

    Local frame: X = longitudinal (forward), Y = lateral (left).
    Rotation by yaw, then translate by origin.

      p_global = R(yaw) @ p_local + origin
    """
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (R @ xy_local.T).T + origin


def _build_global_poses(rows: list[dict]) -> list[tuple[np.ndarray, float]]:
    """
    Compute (origin_xy, yaw) for each sequence in the shared global frame.

    Row 0 defines the global origin (0, 0) with yaw=0.
    For each subsequent row k, find the most recent earlier row j whose
    full GT covers t0_k, read the local pose there, and convert to global.
    """
    poses: list[tuple[np.ndarray, float]] = [(np.zeros(2), 0.0)]

    for k in range(1, len(rows)):
        found = False
        for j in range(k - 1, -1, -1):
            dt = rows[k]["t0_s"] - rows[j]["t0_s"]
            idx = round(dt / DT) - 1
            gt_full = rows[j]["gt_xyz_full"]
            if 0 <= idx < len(gt_full):
                # Local pose of rows[k] in rows[j]'s frame
                x_loc = gt_full[idx, 0]
                y_loc = gt_full[idx, 1]
                yaw_loc = rows[j]["gt_yaw_full"][idx]

                # Convert to global via rows[j]'s global pose
                origin_j, yaw_j = poses[j]
                p_global = _local_to_global(
                    np.array([[x_loc, y_loc]]), origin_j, yaw_j
                )[0]
                yaw_global = yaw_j + yaw_loc

                poses.append((p_global, yaw_global))
                found = True
                break

        if not found:
            poses.append(poses[-1])   # fallback: reuse last pose

    return poses


def _build_stitched_gt(rows: list[dict],
                       poses: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Build a single continuous GT path in global frame by chaining sequences.

    Each sequence contributes the portion of its GT that hasn't been covered
    by an earlier sequence (non-overlapping extension).
    Returns (M, 2) array [X_long, Y_lat] in global frame.
    """
    segments: list[np.ndarray] = []
    covered_until_s: float = rows[0]["t0_s"] - 100.0

    for k, (row, (origin, yaw)) in enumerate(zip(rows, poses)):
        t0 = row["t0_s"]
        gt_full = row["gt_xyz_full"]  # (64, 3)
        n_steps = len(gt_full)

        # time stamps of this GT in global time
        t_start = t0 + DT          # first GT step
        t_end   = t0 + n_steps * DT

        # skip steps already covered
        skip = max(0, round((covered_until_s - t_start) / DT) + 1)
        if skip >= n_steps:
            continue

        xy_local = gt_full[skip:, :2]
        xy_global = _local_to_global(xy_local, origin, yaw)

        # adjust origin so the new segment starts exactly where the last ended
        if segments and skip > 0:
            # recompute first point explicitly for continuity
            first_local = gt_full[skip:skip+1, :2]
            xy_global = _local_to_global(first_local, origin, yaw)
            xy_global = np.vstack([xy_global,
                                   _local_to_global(gt_full[skip+1:, :2],
                                                    origin, yaw)])

        segments.append(xy_global)
        covered_until_s = t_end

    return np.vstack(segments) if segments else np.zeros((0, 2))


# ══════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════

def eval_clip(
    clip_id: str,
    sequences: list[tuple[int, Path]],
    model: CotendMLP,
    device: torch.device,
) -> dict | None:
    """Evaluate all sequences in one clip. Returns None on failure."""
    rows = []
    for t0_us, h5_path in sequences:
        try:
            with h5py.File(h5_path, "r") as f:
                gt_xyz     = f["gt/future_xyz"][:]
                gt_ego     = f["gt/future_ego_states"][:]
                hist       = f["input/ego_history_ego_states"][:]
                cotend     = f["output/cotend_hidden_state"][:]
                has_opt    = "labels/mpc_weights" in f
                w_opt      = f["labels/mpc_weights"][:].astype(np.float64) if has_opt else None
                ade_stored = float(f["labels/ade"][()]) if has_opt else float("nan")
        except Exception as e:
            print(f"  WARN: {h5_path.name} load failed - {e}")
            continue

        v0       = float(hist[-1, 2])
        gt_yaw   = gt_ego[:, 3]

        w_mlp = _predict_weights(model, cotend, device)
        xy_mlp, _, ade_mlp = run_mpc(v0, gt_xyz, gt_yaw, w_mlp)
        xy_def, _, ade_def = run_mpc(v0, gt_xyz, gt_yaw, WEIGHTS_DEFAULT)

        if has_opt and w_opt is not None:
            xy_opt, _, ade_opt = run_mpc(v0, gt_xyz, gt_yaw, w_opt)
        else:
            xy_opt, ade_opt = None, float("nan")

        xy_pcl_arr, ade_pcl, fde_pcl = pseudo_closed_loop(v0, gt_xyz, gt_yaw, w_mlp)

        rows.append({
            "t0_s":        t0_us / 1e6,
            "v0":          v0,
            "w_mlp":       w_mlp,
            "w_opt":       w_opt,
            "ade_mlp":     ade_mlp,
            "ade_def":     ade_def,
            "ade_opt":     ade_opt,
            "ade_pcl":     ade_pcl,
            "fde_pcl":     fde_pcl,
            "ade_stored":  ade_stored,
            "gt_xy":       gt_xyz[:N, :2],       # first N steps (for ADE)
            "gt_xyz_full": gt_xyz,               # full 64 steps (for stitching)
            "gt_yaw_full": gt_yaw,               # full 64 steps yaw
            "xy_mlp":      xy_mlp,
            "xy_def":      xy_def,
            "xy_opt":      xy_opt,
            "xy_pcl":      xy_pcl_arr,
        })

    return {"clip_id": clip_id, "rows": rows} if rows else None


# ══════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════

def plot_clip(clip_data: dict, out_dir: Path) -> Path:
    clip_id = clip_data["clip_id"]
    rows    = clip_data["rows"]

    t0s     = [r["t0_s"]    for r in rows]
    ade_mlp = [r["ade_mlp"] for r in rows]
    ade_def = [r["ade_def"] for r in rows]
    ade_opt = [r["ade_opt"] for r in rows]
    v0s     = [r["v0"]      for r in rows]
    has_opt = not all(np.isnan(ade_opt))

    w_mlp_arr = np.array([r["w_mlp"] for r in rows])
    if has_opt:
        w_opt_arr = np.array([
            r["w_opt"] if r["w_opt"] is not None else [np.nan] * 5
            for r in rows
        ])

    gain_pct = [(d - m) / d * 100 if d > 0 else 0.0
                for m, d in zip(ade_mlp, ade_def)]

    ade_pcl  = [r["ade_pcl"] for r in rows]
    fde_pcl  = [r["fde_pcl"] for r in rows]
    has_pcl  = not all(np.isnan(ade_pcl))
    gain_pcl = [(d - p) / d * 100 if d > 0 else 0.0
                for p, d in zip(ade_pcl, ade_def)]

    # ── Global-frame transforms ──────────────────────
    poses   = _build_global_poses(rows)
    gt_path = _build_stitched_gt(rows, poses)   # (M, 2)

    # ─── Figure ─────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Clip: {clip_id}  |  "
        f"MLP={np.nanmean(ade_mlp):.3f}m  "
        f"Def={np.nanmean(ade_def):.3f}m"
        + (f"  Opt={np.nanmean(ade_opt):.3f}m" if has_opt else "")
        + (f"  PCL={np.nanmean(ade_pcl):.3f}m(FDE={np.nanmean(fde_pcl):.3f}m)" if has_pcl else "")
        + f"  MLP_impr={np.mean(gain_pct):+.1f}%",
        fontsize=9,
    )

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 1.0, 2.0],
        hspace=0.50,
    )

    # ── Row 0: ADE timeline + improvement bars ───────
    gs0 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.38, width_ratios=[2.2, 1]
    )

    ax_ade = fig.add_subplot(gs0[0])
    ax_ade.plot(t0s, ade_mlp, "o-",  color="mediumpurple", lw=2, ms=7, label="MLP")
    ax_ade.plot(t0s, ade_def, "s--", color="tomato",       lw=2, ms=7, label="Default")
    if has_opt:
        ax_ade.plot(t0s, ade_opt, "^:", color="steelblue",   lw=2, ms=7, label="Opt(IOC)")
    if has_pcl:
        ax_ade.plot(t0s, ade_pcl, "D-", color="forestgreen", lw=2, ms=7, label="PCL(RHC)")
    ax_ade.set_xlabel("Clip time t0 [s]")
    ax_ade.set_ylabel("ADE [m]")
    ax_ade.set_title("ADE Timeline  (lower is better)")
    ax_ade.set_xticks(t0s)
    ax_ade.grid(True, ls=":", lw=0.5)

    ax_v = ax_ade.twinx()
    ax_v.plot(t0s, v0s, "D-.", color="gray", lw=1, ms=4, alpha=0.55, label="v0 [m/s]")
    ax_v.set_ylabel("v0 [m/s]", color="gray", fontsize=8)
    ax_v.tick_params(axis="y", labelcolor="gray", labelsize=7)

    lines1, labels1 = ax_ade.get_legend_handles_labels()
    lines2, labels2 = ax_v.get_legend_handles_labels()
    ax_ade.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax_gain = fig.add_subplot(gs0[1])
    x_pos       = np.arange(len(t0s))
    tick_labels = [f"{t:.0f}s" for t in t0s]
    if has_pcl:
        width  = 0.38
        c_mlp  = ["seagreen" if g > 0 else "tomato"   for g in gain_pct]
        c_pcl  = ["steelblue" if g > 0 else "salmon"  for g in gain_pcl]
        bars_m = ax_gain.bar(x_pos - width / 2, gain_pct, width,
                              color=c_mlp, alpha=0.85, edgecolor="white", label="MLP")
        bars_p = ax_gain.bar(x_pos + width / 2, gain_pcl, width,
                              color=c_pcl, alpha=0.85, edgecolor="white", label="PCL")
        for bar, g in zip(bars_m, gain_pct):
            ax_gain.text(bar.get_x() + bar.get_width() / 2,
                         g + (0.3 if g >= 0 else -1.0),
                         f"{g:+.1f}%", ha="center", va="bottom", fontsize=7)
        for bar, g in zip(bars_p, gain_pcl):
            ax_gain.text(bar.get_x() + bar.get_width() / 2,
                         g + (0.3 if g >= 0 else -1.0),
                         f"{g:+.1f}%", ha="center", va="bottom", fontsize=7)
        ax_gain.legend(fontsize=7)
    else:
        c_mlp  = ["seagreen" if g > 0 else "tomato" for g in gain_pct]
        bars_m = ax_gain.bar(x_pos, gain_pct, color=c_mlp, alpha=0.85, edgecolor="white")
        for bar, g in zip(bars_m, gain_pct):
            ax_gain.text(bar.get_x() + bar.get_width() / 2,
                         g + (0.4 if g >= 0 else -1.2),
                         f"{g:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax_gain.axhline(0, c="k", lw=0.8)
    ax_gain.set_xticks(x_pos)
    ax_gain.set_xticklabels(tick_labels)
    ax_gain.set_xlabel("t0")
    ax_gain.set_ylabel("ADE improvement [%]\n(+: better than Default)")
    ax_gain.set_title("Improvement vs Default")
    ax_gain.grid(True, axis="y", ls=":", lw=0.5)

    # ── Row 1: Weight evolution ───────────────────────
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], wspace=0.35
    )
    for k, (wname, widx) in enumerate(zip(_PRED_NAMES, _PRED_WIDX)):
        ax_w = fig.add_subplot(gs1[k])
        ax_w.plot(t0s, w_mlp_arr[:, widx], "o-", color="mediumpurple",
                  lw=1.8, ms=6, label="MLP")
        ax_w.axhline(WEIGHTS_DEFAULT[widx], color="tomato", ls="--",
                     lw=1.2, label=f"def={WEIGHTS_DEFAULT[widx]:.2f}")
        if has_opt:
            ax_w.plot(t0s, w_opt_arr[:, widx], "^:", color="steelblue",
                      lw=1.2, ms=5, label="Opt")
        ax_w.set_title(wname, fontsize=8)
        ax_w.set_xticks(t0s)
        ax_w.set_xticklabels([f"{t:.0f}s" for t in t0s], fontsize=7)
        ax_w.legend(fontsize=6, loc="best")
        ax_w.grid(True, ls=":", lw=0.5)
        ax_w.set_xlabel("t0 [s]", fontsize=7)

    # ── Row 2: Stitched global-frame trajectory ───────
    ax_traj = fig.add_subplot(outer[2])

    # GT continuous path — thin dark reference line
    if len(gt_path) > 0:
        ax_traj.plot(gt_path[:, 1], gt_path[:, 0],
                     "-", color="#333333", lw=1.2, alpha=0.6,
                     zorder=3, label="GT (full clip)")

    # MPC predictions per timestamp, all in global frame
    for k, (row, (origin, yaw), color) in enumerate(
        zip(rows, poses, _T_COLORS[:len(rows)])
    ):
        label_t = f"t0={row['t0_s']:.0f}s"

        # MLP — solid, prominent
        xy_mlp_g = _local_to_global(row["xy_mlp"], origin, yaw)
        ax_traj.plot(xy_mlp_g[:, 1], xy_mlp_g[:, 0],
                     "-", color=color, lw=2.2, zorder=5,
                     label=f"MLP {label_t} ({row['ade_mlp']:.3f}m)")

        # Default — same color, dashed, faded
        xy_def_g = _local_to_global(row["xy_def"], origin, yaw)
        ax_traj.plot(xy_def_g[:, 1], xy_def_g[:, 0],
                     "--", color=color, lw=1.8, alpha=0.75, zorder=4,
                     label=f"Def {label_t} ({row['ade_def']:.3f}m)")

        # Opt — same color, dash-dot, faded (if available)
        if row["xy_opt"] is not None:
            xy_opt_g = _local_to_global(row["xy_opt"], origin, yaw)
            ax_traj.plot(xy_opt_g[:, 1], xy_opt_g[:, 0],
                         "-.", color=color, lw=1.0, alpha=0.35, zorder=4)

        # PCL — same color, dotted, thick (Receding Horizon 시뮬레이션)
        if has_pcl and row["xy_pcl"] is not None:
            xy_pcl_g = _local_to_global(row["xy_pcl"], origin, yaw)
            ax_traj.plot(xy_pcl_g[:, 1], xy_pcl_g[:, 0],
                         ":", color=color, lw=2.5, alpha=0.90, zorder=5,
                         label=f"PCL {label_t} (ADE={row['ade_pcl']:.3f}m FDE={row['fde_pcl']:.3f}m)")

        # t0 marker on path
        ax_traj.plot(origin[1], origin[0],
                     "o", color=color, ms=10, zorder=6,
                     markeredgecolor="white", markeredgewidth=1.5)
        ax_traj.text(origin[1], origin[0] + 0.35, label_t,
                     fontsize=7, color=color, ha="center", fontweight="bold")

    ax_traj.set_xlabel("Y / lateral [m]  (<- left  |  right ->)")
    ax_traj.set_ylabel("X / longitudinal [m]")
    ax_traj.set_title(
        "Stitched global trajectory  "
        "(dark=GT  |  solid=MLP, dashed=Default, same color per t0)"
    )
    ax_traj.invert_xaxis()
    ax_traj.legend(fontsize=7, loc="best", ncol=3)
    ax_traj.grid(True, ls=":", lw=0.5)
    ax_traj.set_aspect("equal", adjustable="datalim")

    out_path = out_dir / f"{clip_id[:8]}_clip_eval.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Clip-level MLP+MPC continuous validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, help="Trained MLP model directory")
    p.add_argument("--split-dir", required=True, help="Split directory (contains test/)")
    p.add_argument("--out",       required=True, help="Output directory")
    p.add_argument("--n-clips",   type=int, default=10, help="Number of clips to evaluate")
    p.add_argument("--min-seq",   type=int, default=3,
                   help="Min sequences per clip (default: 3)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, _ = load_mlp(Path(args.model_dir), device)

    test_dir = Path(args.split_dir) / "test"
    if not test_dir.exists():
        test_dir = Path(args.split_dir)
    clips = _group_clips(test_dir)

    valid = {cid: seqs for cid, seqs in clips.items() if len(seqs) >= args.min_seq}
    print(f"test set: {len(clips):,} clips -> {len(valid):,} clips (>={args.min_seq} sequences)")

    random.seed(args.seed)
    chosen = random.sample(sorted(valid.keys()), min(args.n_clips, len(valid)))
    print(f"evaluating: {len(chosen)} clips (seed={args.seed})\n")

    all_gains:    list[float] = []
    all_pcl_ades: list[float] = []
    all_pcl_gains: list[float] = []
    all_fdes:     list[float] = []
    mlp_better  = 0
    pcl_better  = 0

    for i, clip_id in enumerate(chosen, 1):
        print(f"[{i:2d}/{len(chosen)}] {clip_id} ... ", end="", flush=True)
        data = eval_clip(clip_id, valid[clip_id], model, device)
        if data is None:
            print("SKIP (load failed)")
            continue

        out_path = plot_clip(data, out_dir)

        rows     = data["rows"]
        mean_mlp = float(np.nanmean([r["ade_mlp"] for r in rows]))
        mean_def = float(np.nanmean([r["ade_def"] for r in rows]))
        mean_opt = float(np.nanmean([r["ade_opt"] for r in rows]))
        mean_pcl = float(np.nanmean([r["ade_pcl"] for r in rows]))
        mean_fde = float(np.nanmean([r["fde_pcl"] for r in rows]))
        gain     = (mean_def - mean_mlp) / mean_def * 100 if mean_def > 0 else 0.0
        gain_pcl = (mean_def - mean_pcl) / mean_def * 100 if mean_def > 0 else 0.0

        win_steps = sum(1 for r in rows if r["ade_mlp"] < r["ade_def"])
        all_gains.append(gain)
        all_pcl_ades.append(mean_pcl)
        all_pcl_gains.append(gain_pcl)
        all_fdes.append(mean_fde)
        if gain > 0:
            mlp_better += 1
        if gain_pcl > 0:
            pcl_better += 1

        print(
            f"MLP={mean_mlp:.3f}m  PCL={mean_pcl:.3f}m  Def={mean_def:.3f}m"
            + (f"  Opt={mean_opt:.3f}m" if not np.isnan(mean_opt) else "")
            + f"  impr_mlp={gain:+.1f}%  impr_pcl={gain_pcl:+.1f}%"
            + f"  FDE={mean_fde:.3f}m"
            + f"  ({win_steps}/{len(rows)} steps)"
            + f"  -> {out_path.name}"
        )

    if all_gains:
        print()
        print("-" * 65)
        print(f"  clips evaluated   : {len(all_gains)}")
        print(f"  MLP mean impr     : {np.mean(all_gains):+.2f}%")
        print(f"  MLP median impr   : {np.median(all_gains):+.2f}%")
        print(f"  MLP wins (vs Def) : {mlp_better} / {len(all_gains)}")
        print(f"  PCL mean ADE      : {np.mean(all_pcl_ades):.3f}m")
        print(f"  PCL mean FDE      : {np.mean(all_fdes):.3f}m")
        print(f"  PCL mean impr     : {np.mean(all_pcl_gains):+.2f}%")
        print(f"  PCL wins (vs Def) : {pcl_better} / {len(all_gains)}")
        print("-" * 65)


if __name__ == "__main__":
    main()
