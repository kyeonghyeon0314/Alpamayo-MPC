#!/usr/bin/env python3
"""
smooth_gt.py — GT trajectory smoothing pre-process (STEP 1.7)

Applies Tikhonov 2nd-order penalty smoothing to GT future x, y, yaw
and writes results into the same h5 file under gt_smooth/ group.

  Objective:  min_s  ||s - x||²  +  λ ||D₂ s||²
  Solution:   s = (I + λ D₂ᵀ D₂)⁻¹ x

Output (h5 gt_smooth/ group):
  future_xy  (64, 2)  float32   smoothed x, y
  future_yaw (64,)    float32   smoothed yaw [rad]
  attrs:  lambda_xy, lambda_yaw

Usage:
  python smooth_gt.py \\
      --data-dir /workspace/alpamayo_dataset/data/collected \\
      --lambda-xy 1.0 --lambda-yaw 1.0 --workers 8 \\
      --viz-n 6 --overwrite
"""

import argparse
import logging
import multiprocessing as mp
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ══════════════════════════════════════════════════════
# Paths  ← edit as needed
# ══════════════════════════════════════════════════════
DATA_DIR = "./alpamayo_dataset/data/collected"
VIZ_DIR  = "./alpamayo_dataset/data/smooth_viz"

# ══════════════════════════════════════════════════════
# Smoothing defaults
# ══════════════════════════════════════════════════════
DEFAULT_LAMBDA_XY  = 1.0   # 0 = no smoothing, larger = smoother
DEFAULT_LAMBDA_YAW = 1.0
N_FUTURE = 64
DT       = 0.1             # GT sample interval [s]

# ── Worker globals (set once per process via initializer) ──
_G_INV_A_XY:  np.ndarray | None = None
_G_INV_A_YAW: np.ndarray | None = None
_G_OVERWRITE: bool = False


# ══════════════════════════════════════════════════════
# Smoothing core
# ══════════════════════════════════════════════════════

def _make_inv_A(n: int, lam: float) -> np.ndarray | None:
    """Pre-compute (I + λ D₂ᵀ D₂)⁻¹ once for fixed n and λ.

    Returns None when lam == 0 (identity → no-op).
    """
    if lam == 0.0:
        return None
    d2 = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        d2[i, i]     =  1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] =  1.0
    A = np.eye(n) + lam * (d2.T @ d2)
    return np.linalg.inv(A)


def _apply_smooth(inv_A: np.ndarray | None, signal: np.ndarray) -> np.ndarray:
    """Apply pre-computed inverse: O(N²) matrix-vector multiply."""
    if inv_A is None:
        return signal.copy()
    return inv_A @ signal


def smooth_sample(
    gt_xyz:   np.ndarray,
    gt_yaw:   np.ndarray,
    inv_A_xy: np.ndarray | None,
    inv_A_yaw: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed (xy, yaw) using pre-computed inverses."""
    x_sm = _apply_smooth(inv_A_xy, gt_xyz[:, 0].astype(np.float64))
    y_sm = _apply_smooth(inv_A_xy, gt_xyz[:, 1].astype(np.float64))
    smooth_xy = np.stack([x_sm, y_sm], axis=-1).astype(np.float32)

    yaw_uw  = np.unwrap(gt_yaw.astype(np.float64))
    yaw_sm  = _apply_smooth(inv_A_yaw, yaw_uw)
    yaw_sm  = ((yaw_sm + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)

    return smooth_xy, yaw_sm


# ══════════════════════════════════════════════════════
# File processing
# ══════════════════════════════════════════════════════

def process_file(
    h5_path:   Path,
    inv_A_xy:  np.ndarray | None,
    inv_A_yaw: np.ndarray | None,
    overwrite: bool,
    lam_xy:    float,
    lam_yaw:   float,
) -> str:
    """Process a single h5 file. Returns 'ok' | 'skip' | 'error'."""
    try:
        with h5py.File(h5_path, "a") as f:
            if "gt_smooth" in f and not overwrite:
                return "skip"
            gt_xyz = f["gt/future_xyz"][:]         # (64, 3)
            gt_ego = f["gt/future_ego_states"][:]  # (64, 5)

            gt_yaw = gt_ego[:, 3]
            smooth_xy, smooth_yaw = smooth_sample(gt_xyz, gt_yaw, inv_A_xy, inv_A_yaw)

            grp = f.require_group("gt_smooth")
            for key in ("future_xy", "future_yaw"):
                if key in grp:
                    del grp[key]
            grp.create_dataset("future_xy",  data=smooth_xy)
            grp.create_dataset("future_yaw", data=smooth_yaw)
            grp.attrs["lambda_xy"]  = lam_xy
            grp.attrs["lambda_yaw"] = lam_yaw

        return "ok"

    except Exception as e:
        logging.warning("ERROR %s — %s", h5_path.name, e)
        return "error"


# ── Multiprocessing worker ──────────────────────────

def _init_worker(
    inv_A_xy:  np.ndarray | None,
    inv_A_yaw: np.ndarray | None,
    overwrite: bool,
) -> None:
    global _G_INV_A_XY, _G_INV_A_YAW, _G_OVERWRITE
    _G_INV_A_XY  = inv_A_xy
    _G_INV_A_YAW = inv_A_yaw
    _G_OVERWRITE = overwrite


def _worker_task(args: tuple) -> str:
    h5_path, lam_xy, lam_yaw = args
    return process_file(
        h5_path, _G_INV_A_XY, _G_INV_A_YAW, _G_OVERWRITE, lam_xy, lam_yaw
    )


# ══════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════

def _plot_sample(h5_path: Path, out_path: Path, dpi: int = 200) -> None:
    """One PNG per sample: 2×3 grid (BEV | x | y / yaw | yaw-rate | empty)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as f:
        gt_xyz = f["gt/future_xyz"][:]
        gt_ego = f["gt/future_ego_states"][:]
        sm_xy  = f["gt_smooth/future_xy"][:]
        sm_yaw = f["gt_smooth/future_yaw"][:]

    gt_yaw = gt_ego[:, 3]
    t = np.arange(len(gt_xyz)) * DT

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(h5_path.stem, fontsize=8)
    axes[1, 2].set_visible(False)

    # BEV — minimum ±5 m lateral window to avoid squish on straight segments
    ax = axes[0, 0]
    ax.plot(gt_xyz[:, 1], gt_xyz[:, 0], color="tomato",    lw=1.3, label="raw")
    ax.plot(sm_xy[:, 1],  sm_xy[:, 0],  color="steelblue", lw=1.3, label="smooth")
    ax.scatter([0], [0], c="k", s=20, zorder=5)
    ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
    ax.set_title("BEV")
    ax.legend(fontsize=7)
    all_y  = np.concatenate([gt_xyz[:, 1], sm_xy[:, 1]])
    y_c    = all_y.mean()
    y_half = max(np.ptp(all_y) / 2, 5.0)
    ax.set_xlim(y_c + y_half, y_c - y_half)   # reversed → invert x

    # x(t)
    ax = axes[0, 1]
    ax.plot(t, gt_xyz[:, 0], color="tomato",    lw=1.0, label="raw")
    ax.plot(t, sm_xy[:, 0],  color="steelblue", lw=1.0, label="smooth")
    ax.set_xlabel("t [s]"); ax.set_ylabel("x [m]"); ax.set_title("x(t)")
    ax.legend(fontsize=7)

    # y(t)
    ax = axes[0, 2]
    ax.plot(t, gt_xyz[:, 1], color="tomato",    lw=1.0, label="raw")
    ax.plot(t, sm_xy[:, 1],  color="steelblue", lw=1.0, label="smooth")
    ax.set_xlabel("t [s]"); ax.set_ylabel("y [m]"); ax.set_title("y(t)")
    ax.legend(fontsize=7)

    # yaw(t)
    ax = axes[1, 0]
    ax.plot(t, gt_yaw, color="tomato",    lw=1.0, label="raw")
    ax.plot(t, sm_yaw, color="steelblue", lw=1.0, label="smooth")
    ax.set_xlabel("t [s]"); ax.set_ylabel("yaw [rad]"); ax.set_title("yaw(t)")
    ax.legend(fontsize=7)

    # yaw rate proxy
    ax = axes[1, 1]
    dyaw_raw = np.diff(np.unwrap(gt_yaw.astype(np.float64))) / DT
    dyaw_sm  = np.diff(np.unwrap(sm_yaw.astype(np.float64))) / DT
    ax.plot(t[:-1], dyaw_raw, color="tomato",    lw=1.0, label="raw")
    ax.plot(t[:-1], dyaw_sm,  color="steelblue", lw=1.0, label="smooth")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("t [s]"); ax.set_ylabel("dyaw/dt [rad/s]"); ax.set_title("yaw rate (proxy)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _has_smooth(h5_path: Path) -> bool:
    try:
        with h5py.File(h5_path, "r") as f:
            return "gt_smooth" in f
    except Exception:
        return False


def save_viz(
    h5_files: list[Path],
    viz_dir:  Path,
    n:        int,
    seed:     int,
    dpi:      int = 200,
) -> None:
    """Save one PNG per randomly chosen sample."""
    rng     = random.Random(seed)
    samples = rng.sample(h5_files, min(n, len(h5_files)))
    samples = [p for p in samples if _has_smooth(p)]
    if not samples:
        logging.warning("No files with gt_smooth/ found for visualization.")
        return

    viz_dir.mkdir(parents=True, exist_ok=True)
    for h5_path in samples:
        out_path = viz_dir / f"{h5_path.stem}.png"
        _plot_sample(h5_path, out_path, dpi=dpi)
        logging.info("Saved: %s", out_path)


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GT trajectory Tikhonov smoothing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir",   default=DATA_DIR,
                        help="Directory containing .h5 files")
    parser.add_argument("--viz-dir",    default=VIZ_DIR,
                        help="Output directory for visualization PNGs")
    parser.add_argument("--lambda-xy",  type=float, default=DEFAULT_LAMBDA_XY,
                        help=f"xy smoothing strength (default: {DEFAULT_LAMBDA_XY})")
    parser.add_argument("--lambda-yaw", type=float, default=DEFAULT_LAMBDA_YAW,
                        help=f"yaw smoothing strength (default: {DEFAULT_LAMBDA_YAW})")
    parser.add_argument("--workers",    type=int, default=mp.cpu_count(),
                        help="Parallel worker processes (default: cpu_count)")
    parser.add_argument("--viz-n",      type=int, default=6,
                        help="Number of random samples to visualize (default: 6)")
    parser.add_argument("--viz-seed",   type=int, default=42,
                        help="Random seed for visualization sample selection")
    parser.add_argument("--dpi",        type=int, default=200,
                        help="PNG resolution in DPI (default: 200)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Overwrite existing gt_smooth/ group")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Process only 10 random files for quick check")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    viz_dir  = Path(args.viz_dir)

    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        logging.error("No .h5 files found in: %s", data_dir)
        return

    if args.dry_run:
        h5_files = random.sample(h5_files, min(10, len(h5_files)))
        logging.info("dry-run: processing %d files", len(h5_files))

    # Pre-compute inverse matrices once — reused across all files / workers
    inv_A_xy  = _make_inv_A(N_FUTURE, args.lambda_xy)
    inv_A_yaw = _make_inv_A(N_FUTURE, args.lambda_yaw)

    logging.info(
        "Files: %d | λ_xy=%.3f | λ_yaw=%.3f | workers=%d | overwrite=%s",
        len(h5_files), args.lambda_xy, args.lambda_yaw, args.workers, args.overwrite,
    )

    tasks = [(p, args.lambda_xy, args.lambda_yaw) for p in h5_files]

    ok = skip = error = 0

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(inv_A_xy, inv_A_yaw, args.overwrite),
    ) as pool:
        with tqdm(total=len(tasks), unit="file",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                  ) as pbar:
            for status in pool.imap_unordered(_worker_task, tasks, chunksize=32):
                if   status == "ok":    ok    += 1
                elif status == "skip":  skip  += 1
                else:                   error += 1
                pbar.set_postfix(ok=ok, skip=skip, err=error, refresh=False)
                pbar.update(1)

    logging.info("Done — ok=%d  skip=%d  error=%d", ok, skip, error)

    if args.viz_n > 0:
        save_viz(h5_files, viz_dir, n=args.viz_n, seed=args.viz_seed, dpi=args.dpi)


if __name__ == "__main__":
    main()
