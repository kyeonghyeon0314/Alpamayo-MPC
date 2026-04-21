#!/usr/bin/env python3
"""
diff_wmpc_train.py — Differentiable Weights-Varying MPC 학습 (End-to-End)

IOC 라벨링 없이 ADE 손실로 직접 MLP를 학습.

PCL (Pseudo-Closed-Loop) 구조:
  클립을 K=5 프레임(0.5s) 단위로 롤링. K 프레임마다 cotend → θ 갱신.
  한 클립 내에서 θ가 동적으로 변화 → 씬 변화에 적응하는 가중치 학습.

  [clip: 30 frames = 3s]
  0.0s: cotend[0]  → θ₀  → MPC k=0..4   ADE loss
  0.5s: cotend[5]  → θ₅  → MPC k=5..9   ADE loss
  1.0s: cotend[10] → θ₁₀ → MPC k=10..14 ADE loss
  ...
  clip loss = mean(all segment ADE)

기울기 흐름:
  cotend → MLP → log_θ → exp → θ → MpcDiffLayer → U*
                                       ↑ IFT adjoint (mpc_grad.py)
  → x_free + S_u @ U* → xy_pred → ADE → backward

논문 대응:
  Jahncke et al., "Differentiable Weights-Varying Nonlinear MPC
  via Gradient-Based Policy Learning", IEEE RA-L 2026
  - Gradient clipping ±0.1  (논문 Section III)
  - exp 활성화 (논문 softplus와 동일 역할: 양수 가중치 보장)
  - Fallback: MPC 실패 시 해당 프레임 skip

실행 예시:
  py alpamayo_dataset/diff_wmpc_train.py \\
      --data-dir alpamayo_dataset/data/prepare \\
      --out      alpamayo_dataset/diff_wmpc_out \\
      --K 5 --lr 1e-4 --epochs 200
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import CotendMLP, PREDICT_IDX, W_NAMES, N_OUT, LOG_MIN, LOG_MAX
from mpc import (
    compute_x0, N_EVAL, NX, WEIGHTS_DEFAULT, W_LAT_FIXED,
)
from mpc_grad import (
    precompute_scene, MpcDiffLayer, BatchMpcDiffLayer, PERF,
)


# ══════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════

_GRAD_CLIP_VAL = 0.1   # 논문 Section III: gradient clipping ±0.1


# ══════════════════════════════════════════════════════
# 데이터 로딩
# ══════════════════════════════════════════════════════

def _clip_sort_key(p: Path) -> int:
    """파일명 {uuid}__{timestamp}.h5 → 타임스탬프로 정렬."""
    try:
        return int(p.stem.split("__")[1])
    except (IndexError, ValueError):
        return 0


def _load_frame_from_h5(h5_path: str) -> dict | None:
    """h5 하나에서 cotend + scene 계산. 캐시 없음, _load_clip 내부 전용."""
    try:
        with h5py.File(h5_path, "r") as f:
            if not bool(f["labels/valid"][()]):
                return None

            cotend    = f["output/cotend_hidden_state"][:].astype(np.float32)
            hist      = f["input/ego_history_ego_states"][:]
            hist_vel  = f["input/ego_history_vel"][:]
            hist_curv = f["input/ego_history_curv"][:]
            hist_quat = f["input/ego_history_quat_global"][:]
            gt_ego    = f["gt/future_ego_states"][:]

            if "gt_smooth" in f:
                gt_xy  = f["gt_smooth/future_xy"][:].astype(np.float64)
                gt_yaw = f["gt_smooth/future_yaw"][:].astype(np.float64)
            else:
                gt_xyz = f["gt/future_xyz"][:]
                gt_xy  = gt_xyz[:, :2].astype(np.float64)
                gt_yaw = gt_ego[:, 3].astype(np.float64)

            gt_speed     = gt_ego[:, 2].astype(np.float64)
            gt_lon_accel = gt_ego[:, 4].astype(np.float64)

        x0    = compute_x0(
            float(hist[-1, 2]), float(hist[-1, 4]),
            hist_vel[-1], float(hist_curv[-1, 0]), hist_quat[-1],
        )
        scene = precompute_scene(x0, gt_xy, gt_yaw, gt_speed, gt_lon_accel)
        return {"cotend": cotend, "scene": scene}

    except Exception:
        return None


def _is_clip_continuous(sorted_paths: list[Path]) -> bool:
    """
    클립 내 타임스탬프가 연속인지 검사.

    정렬된 파일의 인접 타임스탬프 차이를 구한 뒤, 어떤 간격이라도
    최소 간격의 1.5배를 초과하면 중간에 빈 구간이 있는 것으로 판단 → False.
    길이가 1인 클립(단일 프레임)은 연속으로 간주 → True.
    """
    if len(sorted_paths) <= 1:
        return True
    ts = [_clip_sort_key(p) for p in sorted_paths]
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    min_diff = min(diffs)
    if min_diff <= 0:
        return False
    return all(d <= min_diff * 1.5 for d in diffs)


def _build_clip_paths(split_dir: Path) -> list[list[Path]]:
    """
    split_dir 내 h5 파일을 클립 UUID 기준으로 그룹화 후 타임스탬프 정렬.

    연속성 검사:
      - 중간에 빈 구간(인접 간격 > 최소 간격 × 1.5)이 있는 클립은 제거.
      - 짧지만 연속적인 클립은 유지.
    """
    clip_map: dict[str, list[Path]] = defaultdict(list)
    for p in split_dir.glob("*.h5"):
        clip_id = p.stem.split("__")[0]
        clip_map[clip_id].append(p)

    result = []
    for files in clip_map.values():
        sorted_files = sorted(files, key=_clip_sort_key)
        if _is_clip_continuous(sorted_files):
            result.append(sorted_files)
    return result


def _load_clip(paths: list[Path]) -> list[dict]:
    """h5 파일 목록 → 유효 프레임 list. 캐시 없음 (split 캐시에서 처리)."""
    frames = []
    for p in sorted(paths, key=_clip_sort_key):
        f = _load_frame_from_h5(str(p))
        if f is not None:
            frames.append(f)
    return frames


def preload_all_clips(
    clip_paths_list: list[list[Path]],
    cache_path:      Path,
    rebuild_cache:   bool = False,
    desc:            str  = "",
) -> list[list[dict]]:
    """
    split 전체를 파일 1개(cache_path)로 캐싱.

    저장 구조:
        clip_lengths  (n_clips,) int   — 클립별 프레임 수
        cotend        (total_frames, 4096) float32
        <scene_key>   (total_frames, ...) float64   — scene 배열 각각

    캐시 히트 시: 파일 1개 로드 → ~수 초
    캐시 미스 시: h5 전체 순회 + precompute_scene + 저장
    """
    # ── 캐시 히트 ────────────────────────────────────────────────────
    if cache_path.exists() and not rebuild_cache:
        try:
            raw          = np.load(cache_path, allow_pickle=False)
            clip_lengths = raw["clip_lengths"].tolist()
            scene_keys   = [k for k in raw.files
                            if k not in ("clip_lengths", "cotend")]

            # 배열 전체를 RAM에 올린 뒤 인덱싱
            # (NpzFile 은 캐시 없어서 data[k][i] 호출마다 zip 재읽기 → 수백 초)
            cotend_all    = np.asarray(raw["cotend"])
            scene_arrays  = {k: np.asarray(raw[k]) for k in scene_keys}

            loaded  = []
            skipped = 0
            start   = 0
            for length in clip_lengths:
                end    = start + length
                frames = [
                    {"cotend": cotend_all[i],
                     "scene":  {k: scene_arrays[k][i] for k in scene_keys}}
                    for i in range(start, end)
                ]
                if len(frames) >= 2:
                    loaded.append(frames)
                else:
                    skipped += 1
                start = end
            if desc:
                print(f"  {desc}: {len(loaded)} 클립 로드 완료"
                      + (f" ({skipped}개 제외)" if skipped else "")
                      + " [캐시]")
            return loaded
        except Exception as e:
            print(f"  [{desc}] 캐시 로드 실패({e}), h5에서 재생성")
            pass   # 손상 → 재생성

    # ── 캐시 미스: h5 전체 순회 ─────────────────────────────────────
    all_clips:     list[list[dict]] = []
    clip_lengths:  list[int]        = []
    skipped = 0

    for paths in clip_paths_list:
        frames = _load_clip(paths)
        clip_lengths.append(len(frames))
        if len(frames) >= 2:
            all_clips.append(frames)
        else:
            skipped += 1

    # ── split 전체를 단일 npz로 저장 ─────────────────────────────────
    if all_clips:
        try:
            all_frames = [f for clip in all_clips for f in clip]
            scene_keys = list(all_frames[0]["scene"].keys())
            save_dict  = {
                "clip_lengths": np.array(
                    [len(c) for c in all_clips], dtype=np.int32),
                "cotend": np.stack([f["cotend"] for f in all_frames]),
            }
            for k in scene_keys:
                # float32로 저장: 파일 크기 절반, 로드 속도 2배
                # QP는 내부에서 float64로 캐스팅하므로 정밀도 영향 없음
                save_dict[k] = np.stack(
                    [f["scene"][k] for f in all_frames]
                ).astype(np.float32)
            np.savez(cache_path, **save_dict)
        except Exception as e:
            pass   # 저장 실패 무시

    if desc:
        print(f"  {desc}: {len(all_clips)} 클립 로드 완료"
              + (f" ({skipped}개 유효 프레임 부족으로 제외)" if skipped else ""))
    return all_clips


# ══════════════════════════════════════════════════════
# PCL Rolling-Window 학습 단위
# ══════════════════════════════════════════════════════

def _forward_batch(
    model:      CotendMLP,
    clip_batch: list[list[dict]],
    K:          int,
    device:     torch.device,
) -> torch.Tensor:
    """
    B개 클립을 한 번의 MLP forward로 처리 (GPU 활용률 향상).

    구조:
      1. 모든 클립의 모든 세그먼트 시작 cotend를 모아 (N_segs, 4096) 배치 구성
      2. MLP 1회 forward → log_θ (N_segs, 4)
      3. 각 세그먼트 θ → 프레임별 MpcDiffLayer → ADE
      4. 전체 ADE 평균 반환

    Returns:
        batch_loss (scalar tensor, requires_grad=True)
    """
    _log_min = torch.tensor(LOG_MIN, dtype=torch.float32, device=device)
    _log_max = torch.tensor(LOG_MAX, dtype=torch.float32, device=device)
    n_cmp    = min(K, N_EVAL)

    # ── Step 1: 모든 세그먼트 인덱싱 ──────────────────────────────────
    seg_cotends: list[np.ndarray]         = []
    seg_frames:  list[list[dict]]         = []

    for frames in clip_batch:
        for start in range(0, len(frames), K):
            seg = frames[start:start + K]
            if seg:
                seg_cotends.append(seg[0]["cotend"])
                seg_frames.append(seg)

    if not seg_cotends:
        return torch.zeros(1, device=device).squeeze()

    # ── Step 2: 배치 MLP forward ──────────────────────────────────────
    _t0 = time.perf_counter()
    cotend_batch  = torch.tensor(
        np.stack(seg_cotends), dtype=torch.float32, device=device
    )                                               # (N_segs, 4096)
    log_theta_all = model(cotend_batch)             # (N_segs, 4)
    log_theta_all = torch.clamp(log_theta_all, _log_min, _log_max)
    theta_all     = torch.exp(log_theta_all)        # (N_segs, 4)
    PERF.add("mlp_fwd", time.perf_counter() - _t0)

    # ── Step 3: 세그먼트별 QP ─────────────────────────────────────
    # U_opt 텐서를 모아뒀다가 Step 4에서 한 번에 batched GPU 연산
    all_U_opts: list[torch.Tensor] = []   # 각 (NNU,)
    all_scenes: list[dict]         = []   # 대응 scene

    for seg_idx, seg in enumerate(seg_frames):
        theta  = theta_all[seg_idx]
        scenes = [frame["scene"] for frame in seg]
        try:
            U_opts = BatchMpcDiffLayer.apply(theta, scenes)   # (K, NNU)
        except Exception:
            continue
        for fi in range(len(seg)):
            all_U_opts.append(U_opts[fi])
            all_scenes.append(scenes[fi])

    if not all_U_opts:
        return torch.zeros(1, device=device).squeeze()

    # ── Step 4: Batched extract_xy + ADE (GPU 연산 1회) ──────────
    # 프레임별 개별 호출(22k 소형 GPU op) → 단일 bmm으로 대체
    _t0 = time.perf_counter()
    N = len(all_U_opts)

    S_u_xy_t    = torch.tensor(
        np.stack([s["S_u_xy"]    for s in all_scenes]),
        dtype=torch.float32, device=device,
    )                                               # (N, N_EVAL*2, NNU)
    x_free_xy_t = torch.tensor(
        np.stack([s["x_free_xy"] for s in all_scenes]),
        dtype=torch.float32, device=device,
    )                                               # (N, N_EVAL*2)
    gt_xy_t     = torch.tensor(
        np.stack([s["gt_xy_eval"][:n_cmp] for s in all_scenes]),
        dtype=torch.float32, device=device,
    )                                               # (N, n_cmp, 2)

    U_all    = torch.stack(all_U_opts)              # (N, NNU) — 모든 QP branch 연결
    xy_flat  = x_free_xy_t + torch.bmm(
        S_u_xy_t, U_all.unsqueeze(-1)
    ).squeeze(-1)                                   # (N, N_EVAL*2)
    xy_pred  = xy_flat.reshape(N, N_EVAL, 2)        # (N, N_EVAL, 2)

    ade = torch.mean(
        torch.norm(xy_pred[:, :n_cmp] - gt_xy_t, dim=-1), dim=1
    )                                               # (N,)
    PERF.add("extract_xy", time.perf_counter() - _t0)

    return ade.mean()


# ══════════════════════════════════════════════════════
# Epoch 학습 / 평가
# ══════════════════════════════════════════════════════

def train_epoch(
    model:         CotendMLP,
    loaded_clips:  list[list[dict]],
    optimizer:     torch.optim.Optimizer,
    device:        torch.device,
    K:             int,
    batch_clips:   int,
    logger:        logging.Logger,
    epoch:         int,
) -> float:
    model.train()
    random.shuffle(loaded_clips)
    PERF.reset()

    total_loss  = 0.0
    n_batches   = 0
    skipped     = 0
    t_epoch     = time.perf_counter()

    for batch_start in range(0, len(loaded_clips), batch_clips):
        batch = loaded_clips[batch_start:batch_start + batch_clips]

        optimizer.zero_grad()
        loss = _forward_batch(model, batch, K, device)

        if not loss.requires_grad or not torch.isfinite(loss):
            skipped += len(batch)
            continue

        _t0 = time.perf_counter()
        loss.backward()
        PERF.add("backward_total", time.perf_counter() - _t0)

        torch.nn.utils.clip_grad_value_(model.parameters(), _GRAD_CLIP_VAL)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    if n_batches == 0:
        return float("nan")

    epoch_t = time.perf_counter() - t_epoch
    avg     = total_loss / n_batches

    report = PERF.report()
    order  = ["mlp_fwd", "build_hg", "solve_qp", "batch_qp_wall",
              "extract_xy", "backward_total", "ift_grad", "batch_ift_wall"]
    parts  = []
    for k in order:
        if k in report:
            r = report[k]
            parts.append(f"{k}={r['avg_ms']:.1f}ms×{r['n']}")
    logger.info("  [profiling ep%d] total=%.1fs  batch=%d  %s",
                epoch, epoch_t, batch_clips, "  ".join(parts))

    if skipped:
        logger.debug("  [epoch %d] skipped %d clips", epoch, skipped)
    return avg


@torch.no_grad()
def eval_epoch(
    model:        CotendMLP,
    loaded_clips: list[list[dict]],
    device:       torch.device,
    K:            int,
    batch_clips:  int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch_start in range(0, len(loaded_clips), batch_clips):
        batch = loaded_clips[batch_start:batch_start + batch_clips]
        loss  = _forward_batch(model, batch, K, device)
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches if n_batches else float("nan")


# ══════════════════════════════════════════════════════
# 학습 곡선 시각화
# ══════════════════════════════════════════════════════

def save_training_plot(history: dict, best_epoch: int, out_dir: Path,
                       early_stop_epoch: int | None = None) -> None:
    n      = len(history["train_loss"])
    epochs = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train ADE", color="steelblue",  lw=1.5)
    ax.plot(epochs, history["val_loss"],   label="Val ADE",   color="darkorange", lw=1.5)
    ax.axvline(best_epoch, color="green", ls="--", lw=1.0,
               label=f"Best (epoch {best_epoch})")
    ax.scatter([best_epoch], [history["val_loss"][best_epoch - 1]],
               marker="*", s=200, color="green", zorder=5)
    if early_stop_epoch and early_stop_epoch != best_epoch:
        ax.axvline(early_stop_epoch, color="red", ls=":", lw=1.0,
                   label=f"Early Stop (epoch {early_stop_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ADE [m]")
    ax.set_title("Diff-WMPC Training Curves  (PCL Rolling Loss)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════
# 로거
# ══════════════════════════════════════════════════════

def setup_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("diff_wmpc")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    sh  = logging.StreamHandler();  sh.setFormatter(fmt);  logger.addHandler(sh)
    fh  = logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt);  logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diff-WMPC end-to-end 학습 (PCL rolling window)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir",     required=True,
                        help="prepare/ 베이스 디렉토리 (하위 train/ val/ test/)")
    parser.add_argument("--out",          default="diff_wmpc_out")
    parser.add_argument("--K",            type=int,   default=5,
                        help="세그먼트 크기 [프레임] (default: 5 = 0.5s @ 10Hz)")
    parser.add_argument("--batch-clips",  type=int,   default=16,
                        help="한 번에 처리할 클립 수 (GPU 배치, default: 16)")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="학습률 (논문 2.9e-5 기준, default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--hidden",       type=int, nargs="+",
                        help="히든 레이어 (default: 1024 256)")
    parser.add_argument("--patience",     type=int,   default=30)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--pretrain-pt",    type=str,   default=None,
                        help="IOC 지도학습 체크포인트 경로 (warm-start용, 옵션)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="scene .npz 캐시를 무시하고 재계산 (GT 변경 시 사용)")
    args = parser.parse_args()

    if args.hidden is None:
        args.hidden = [1024, 256]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir  = Path(args.out)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    from mpc_grad import _OSQP_AVAILABLE
    solver_name = "OSQP" if _OSQP_AVAILABLE else "scipy L-BFGS-B"
    logger.info("QP 솔버: %s  (constrained case)", solver_name)

    # ── 클립 경로 수집 ────────────────────────────────
    logger.info("클립 경로 수집 중 (비연속 클립 자동 제거)...")
    train_clip_paths = _build_clip_paths(data_dir / "train")
    val_clip_paths   = _build_clip_paths(data_dir / "val")
    test_clip_paths  = _build_clip_paths(data_dir / "test")
    logger.info("  train: %d 클립  val: %d 클립  test: %d 클립",
                len(train_clip_paths), len(val_clip_paths), len(test_clip_paths))
    logger.info("  (연속성 기준: 인접 프레임 간격 > 최소 간격 × 1.5 이면 클립 전체 제거)")

    # ── scene 사전 계산 + 메모리 preload ─────────────
    # split별 단일 npz (_diff_wmpc_cache_{split}.npz) 캐시
    # 첫 실행: h5 전체 순회 + precompute_scene → npz 저장 (수 분)
    # 이후 실행: npz 1개 로드 → 수 초
    logger.info("데이터 사전 로드 중%s...",
                " (--rebuild-cache: 캐시 재생성)" if args.rebuild_cache else "")
    t_load = time.perf_counter()
    train_loaded = preload_all_clips(
        train_clip_paths,
        cache_path=data_dir / "_diff_wmpc_cache_train.npz",
        rebuild_cache=args.rebuild_cache, desc="train")
    val_loaded   = preload_all_clips(
        val_clip_paths,
        cache_path=data_dir / "_diff_wmpc_cache_val.npz",
        rebuild_cache=args.rebuild_cache, desc="val")
    test_loaded  = preload_all_clips(
        test_clip_paths,
        cache_path=data_dir / "_diff_wmpc_cache_test.npz",
        rebuild_cache=args.rebuild_cache, desc="test")
    logger.info("  사전 로드 완료: %.1fs", time.perf_counter() - t_load)

    # ── 모델 ──────────────────────────────────────────
    model = CotendMLP(hidden_dims=tuple(args.hidden), dropout=args.dropout).to(device)
    logger.info("MLP: 4096 → %s → %d  (log-scale 출력)",
                " → ".join(str(h) for h in args.hidden), N_OUT)
    logger.info("파라미터: %d", sum(p.numel() for p in model.parameters()))

    if args.pretrain_pt:
        ckpt = torch.load(args.pretrain_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt if isinstance(ckpt, dict) and "net.0.weight" in ckpt
                              else ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info("Pretrained 가중치 로드: %s", args.pretrain_pt)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 학습 ──────────────────────────────────────────
    logger.info("학습 시작: K=%d (%.1fs 세그먼트), lr=%s, wd=%s, patience=%d",
                args.K, args.K * 0.1, args.lr, args.weight_decay, args.patience)
    logger.info("Gradient clipping: value ±%.2f  (논문 Section III)", _GRAD_CLIP_VAL)
    logger.info("%6s  %10s  %10s  %10s  %8s", "Epoch", "Train ADE", "Val ADE", "LR", "Best")
    logger.info("─" * 52)

    history          = {"train_loss": [], "val_loss": []}
    best_val         = float("inf")
    best_epoch       = 1
    no_improve       = 0
    early_stop_epoch = None
    t_start          = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loaded, optimizer, device,
                                 args.K, args.batch_clips, logger, epoch)
        val_loss   = eval_epoch(model, val_loaded, device, args.K, args.batch_clips)
        if not math.isnan(train_loss):   # optimizer.step() 미호출 시 scheduler 스킵
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            mark = "★" if improved else ""
            logger.info("%6d  %10.4f  %10.4f  %10.2e  %8.4f %s",
                        epoch, train_loss, val_loss,
                        scheduler.get_last_lr()[0], best_val, mark)

        if no_improve >= args.patience:
            early_stop_epoch = epoch
            logger.info("EarlyStopping: %d epoch 미개선 → 종료 (epoch %d)",
                        args.patience, epoch)
            break

    elapsed = time.perf_counter() - t_start
    logger.info("학습 완료: %.1fs  best_val_ADE=%.4f  best_epoch=%d",
                elapsed, best_val, best_epoch)

    # ── 테스트 평가 ────────────────────────────────────
    model.load_state_dict(
        torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loss = eval_epoch(model, test_loaded, device, args.K, args.batch_clips)
    logger.info("Test ADE: %.4f m", test_loss)

    # ── 저장 ──────────────────────────────────────────
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_hidden":     args.hidden,
            "dropout":          args.dropout,
            "predict_idx":      PREDICT_IDX,
            "log_min":          LOG_MIN.tolist(),
            "log_max":          LOG_MAX.tolist(),
        },
        out_dir / "model_full.pt",
    )

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                **vars(args),
                "best_val_ade": best_val,
                "test_ade":     test_loss,
                "best_epoch":   best_epoch,
                "predict_idx":  PREDICT_IDX,
                "weight_names": W_NAMES,
                "K_frames":     args.K,
                "K_seconds":    args.K * 0.1,
            },
            f, indent=2,
        )

    save_training_plot(history, best_epoch, out_dir, early_stop_epoch)

    logger.info("저장 완료: %s", out_dir)
    logger.info("  best_model.pt / model_full.pt / history.json / config.json / training_curves.png")


if __name__ == "__main__":
    main()
