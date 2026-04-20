#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""필터링 + 분할 통합 스크립트 (filter_error_samples + split_dataset 병합).

흐름:
  1. data/collected/*.h5 스캔
     └─ --clips-csv 지정 시: 미완성 클립(일부 t0_us 누락) 제외 → collected/ 에 유보
  2. 지표 계산 (ADE, FDE, lon_ade, lat_ade, hs_norm, v_mean)
  3. 오류 샘플 → data/prepare/error/  이동
  4. 유효 샘플 → data/prepare/train|val|test/  이동  (클립 단위 Stratified)
  5. data/prepare/dataset_manifest.json 저장

결과 구조:
    data/collected/        ← 수집 인박스 (이동 후 비워짐; 유보 클립만 남음)
    data/prepare/
        train/   *.h5
        val/     *.h5
        test/    *.h5
        error/   *.h5
        dataset_manifest.json

실행:
    # 기본
    python alpamayo_dataset/prepare_dataset.py

    # 수집 진행 중 실행 — 미완성 클립 제외 (clips.csv = collect_dataset.py 에 넘긴 것과 동일)
    python alpamayo_dataset/prepare_dataset.py --clips-csv /path/to/clips.csv

    # 분포 먼저 확인 (파일 이동 없음)
    python alpamayo_dataset/prepare_dataset.py --dry-run

    # GPU 배치 모드
    python alpamayo_dataset/prepare_dataset.py --use-gpu

    # 새 데이터 추가 후 재처리 (prepare/ 파일 → collected/ 로 복원 후 재분할)
    python alpamayo_dataset/prepare_dataset.py --overwrite

미완성 클립 감지 (--clips-csv 지정 시):
    CSV (clip_id, t0_us) 에 기재된 t0_us 중 .h5 가 없는 것이 하나라도 있으면
    해당 클립 전체를 건너뜀 → collected/ 에 남겨 다음 실행에서 재처리.

필터 기준 (OR — 하나라도 초과 시 error/ 이동):
    --ade-thresh   ADE [m]              (기본: 2.0,  p99≈0.82m)
    --lon-thresh   Longitudinal ADE [m] (기본: 1.5,  p99≈0.85m)
    --lat-thresh   Lateral ADE [m]      (기본: 1.0,  p99≈0.37m)
    --min-speed    미래 평균 속도 [m/s] (기본: 5.0  ≈ 18 km/h)

분할 설정:
    --train             학습 비율             (기본: 0.70)
    --val               검증 비율             (기본: 0.20, 나머지=test)
    --seed              랜덤 시드             (기본: 42)
    --split-lat-thresh  lateral 클립 판별 임계값: max(|GT_y[:20]|) [m]  (기본: 1.0)
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import logging
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).parent
COLLECTED_DIR = SCRIPT_DIR / "data" / "collected"
PREPARE_DIR   = SCRIPT_DIR / "data" / "prepare"
_MPC_HORIZON  = 20
_SPLIT_DIRS   = ("train", "val", "test", "error")


# ─────────────────────────────────────────────────────────────────────────────
# 미완성 클립 감지
# ─────────────────────────────────────────────────────────────────────────────

def _load_expected_clips(clips_csv: str) -> dict[str, set[int]]:
    """CSV에서 {clip_id: {t0_us, ...}} 로드."""
    expected: dict[str, set[int]] = defaultdict(set)
    with open(clips_csv, newline="") as fh:
        for row in csv_module.DictReader(fh):
            expected[row["clip_id"].strip()].add(int(row["t0_us"]))
    return dict(expected)


def _find_incomplete_clips(
    collected_dir: Path,
    expected: dict[str, set[int]],
) -> tuple[set[str], set[str]]:
    """완성/미완성 클립 ID를 반환.

    collected_dir 에 있는 파일 기준으로 expected 와 비교.
    t0_us 가 하나라도 누락된 클립 → incomplete.
    CSV 에 없는 클립(외부 수집 등) → complete 로 간주.

    Returns:
        (complete_clip_ids, incomplete_clip_ids)
    """
    # collected/ 에 있는 (clip_id, t0_us) 수집
    present: dict[str, set[int]] = defaultdict(set)
    for h5 in collected_dir.glob("*.h5"):
        parts = h5.stem.split("__")
        if len(parts) == 2:
            present[parts[0]].add(int(parts[1]))

    complete: set[str] = set()
    incomplete: set[str] = set()

    for cid, expected_ts in expected.items():
        if expected_ts.issubset(present.get(cid, set())):
            complete.add(cid)
        else:
            incomplete.add(cid)

    # CSV에 없는 클립은 complete로 처리
    for cid in present:
        if cid not in expected:
            complete.add(cid)

    return complete, incomplete


# ─────────────────────────────────────────────────────────────────────────────
# 지표 계산 (CPU 파일 단위)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(path_str: str) -> tuple[str, dict | None]:
    """h5 파일에서 지표를 계산. ProcessPoolExecutor 용."""
    p = Path(path_str)
    try:
        with h5py.File(p, "r") as f:
            gt_xyz   = f["gt/future_xyz"][:_MPC_HORIZON]            # (20, 3)
            pred_xyz = f["output/pred_xyz"][:_MPC_HORIZON]           # (20, 3)
            hist     = f["input/ego_history_ego_states"][:]
            fut_v    = f["gt/future_ego_states"][:_MPC_HORIZON, 2]   # (20,)
            hs_norm  = float(np.linalg.norm(f["output/cotend_hidden_state"][:])) \
                       if "output/cotend_hidden_state" in f else None

        diff_xy = pred_xyz[:, :2] - gt_xyz[:, :2]
        disp    = np.linalg.norm(diff_xy, axis=1)

        return p.stem, {
            "v0":      float(hist[-1, 2]),
            "v_mean":  float(fut_v.mean()),
            "ade":     float(disp.mean()),
            "fde":     float(disp[-1]),
            "lon_ade": float(np.abs(diff_xy[:, 0]).mean()),
            "lat_ade": float(np.abs(diff_xy[:, 1]).mean()),
            "hs_norm": hs_norm,
            "lat_max": float(np.max(np.abs(gt_xyz[:, 1]))),  # 분할 분류용
        }
    except Exception as exc:
        log.warning("읽기/계산 실패: %s — %s", p.name, exc)
        return p.stem, None


# ─────────────────────────────────────────────────────────────────────────────
# GPU 배치 모드
# ─────────────────────────────────────────────────────────────────────────────

def _load_for_gpu(path_str: str) -> dict | None:
    """h5에서 배열만 읽어 반환 (GPU Phase 1 용)."""
    p = Path(path_str)
    try:
        with h5py.File(p, "r") as f:
            gt_xy   = f["gt/future_xyz"][:_MPC_HORIZON, :2].astype(np.float32)
            pred_xy = f["output/pred_xyz"][:_MPC_HORIZON, :2].astype(np.float32)
            hist    = f["input/ego_history_ego_states"][:]
            fut_v   = f["gt/future_ego_states"][:_MPC_HORIZON, 2]
            hs = f["output/cotend_hidden_state"][:].astype(np.float32) \
                 if "output/cotend_hidden_state" in f else None
            lat_max = float(np.max(np.abs(f["gt/future_xyz"][:_MPC_HORIZON, 1])))
        return {
            "stem": p.stem, "path": p, "pred": pred_xy, "gt": gt_xy, "hs": hs,
            "v0": float(hist[-1, 2]), "v_mean": float(fut_v.mean()), "lat_max": lat_max,
        }
    except Exception as exc:
        log.warning("읽기 실패: %s — %s", p.name, exc)
        return None


def _run_gpu_batch(h5_files: list[Path], workers: int) -> dict[str, dict]:
    """{stem: metrics + path} 반환."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("GPU 배치 모드: device=%s", device)

    loaded: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_for_gpu, str(p)): p for p in h5_files}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        unit="샘플", desc="Phase 1 (h5 읽기)"):
            r = fut.result()
            if r is not None:
                loaded.append(r)

    if not loaded:
        return {}

    pred_t = torch.from_numpy(np.stack([d["pred"] for d in loaded])).to(device)
    gt_t   = torch.from_numpy(np.stack([d["gt"]   for d in loaded])).to(device)
    diff   = pred_t - gt_t
    disp   = diff.norm(dim=-1)

    ade_arr     = disp.mean(dim=-1).cpu().numpy()
    fde_arr     = disp[:, -1].cpu().numpy()
    lon_ade_arr = diff[:, :, 0].abs().mean(dim=-1).cpu().numpy()
    lat_ade_arr = diff[:, :, 1].abs().mean(dim=-1).cpu().numpy()

    hs_has: list[int] = [i for i, d in enumerate(loaded) if d["hs"] is not None]
    hs_norm_arr: list[float | None] = [None] * len(loaded)
    if hs_has:
        hs_t  = torch.from_numpy(np.stack([loaded[i]["hs"] for i in hs_has])).to(device)
        norms = hs_t.norm(dim=-1).cpu().numpy()
        for pos, idx in enumerate(hs_has):
            hs_norm_arr[idx] = float(norms[pos])

    return {
        d["stem"]: {
            "v0":      d["v0"],
            "v_mean":  d["v_mean"],
            "ade":     float(ade_arr[i]),
            "fde":     float(fde_arr[i]),
            "lon_ade": float(lon_ade_arr[i]),
            "lat_ade": float(lat_ade_arr[i]),
            "hs_norm": hs_norm_arr[i],
            "lat_max": d["lat_max"],
            "path":    d["path"],
        }
        for i, d in enumerate(loaded)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 통계 출력
# ─────────────────────────────────────────────────────────────────────────────

def _print_stat(label: str, values: list[float], unit: str = "") -> None:
    arr = np.array(values)
    sfx = f" {unit}" if unit else ""
    print(f"  {label:<22} 평균={arr.mean():.3f}{sfx}  "
          f"중앙={np.median(arr):.3f}{sfx}  "
          f"p95={np.percentile(arr, 95):.3f}{sfx}  "
          f"p99={np.percentile(arr, 99):.3f}{sfx}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="필터링 + 분할 통합 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--collected-dir", default=str(COLLECTED_DIR), metavar="DIR",
        help="수집된 .h5 파일 디렉토리 (기본: data/collected)",
    )
    parser.add_argument(
        "--prepare-dir", default=str(PREPARE_DIR), metavar="DIR",
        help="분할 결과 디렉토리 (기본: data/prepare)",
    )
    parser.add_argument(
        "--viz-dir", default=None, metavar="DIR",
        help="PNG 시각화 디렉토리 (기본: --collected-dir 와 동일). "
             "h5 이동 시 동명 .png 도 함께 이동.",
    )
    parser.add_argument(
        "--clips-csv", default=None, metavar="FILE",
        help="collect_dataset.py 에 사용한 CSV (clip_id, t0_us). "
             "지정 시 미완성 클립을 건너뜀.",
    )
    # 필터
    parser.add_argument("--ade-thresh",       type=float, default=2.0,
                        help="ADE 임계값 [m] (기본: 2.0)")
    parser.add_argument("--lon-thresh",       type=float, default=1.5,
                        help="Longitudinal ADE [m] (기본: 1.5)")
    parser.add_argument("--lat-thresh",       type=float, default=1.0,
                        help="Lateral ADE [m] (기본: 1.0)")
    parser.add_argument("--min-speed",        type=float, default=5.0,
                        help="최소 미래 평균 속도 [m/s] (기본: 5.0)")
    # 분할
    parser.add_argument("--train",            type=float, default=0.70)
    parser.add_argument("--val",              type=float, default=0.20)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--split-lat-thresh", type=float, default=1.0,
                        help="lateral 클립 판별: max(|GT_y[:20]|) [m] (기본: 1.0)")
    # 실행
    parser.add_argument("--workers",          type=int,   default=16)
    parser.add_argument("--use-gpu",          action="store_true")
    parser.add_argument("--dry-run",          action="store_true",
                        help="파일 이동 없이 통계만 출력")
    parser.add_argument("--overwrite",        action="store_true",
                        help="prepare/ 파일을 collected/ 로 복원 후 재처리")
    args = parser.parse_args()

    assert args.train + args.val < 1.0, "--train + --val 은 1.0 미만이어야 합니다"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    collected_dir = Path(args.collected_dir)
    prepare_dir   = Path(args.prepare_dir)
    viz_dir       = Path(args.viz_dir) if args.viz_dir else collected_dir
    manifest_path = prepare_dir / "dataset_manifest.json"

    # ── 이미 처리된 경우 안내 ────────────────────────────────────────────────
    if manifest_path.exists() and not args.overwrite and not args.dry_run:
        print(f"[INFO] 이미 처리됨: {manifest_path}")
        print("       재처리하려면 --overwrite 사용")
        with open(manifest_path) as f:
            m = json.load(f)
        print(f"  파일: train={m['n_train']}  val={m['n_val']}  "
              f"test={m['n_test']}  error={m['n_error']}")
        print(f"  클립: train={m['n_clips_train']}  val={m['n_clips_val']}  "
              f"test={m['n_clips_test']}")
        if m.get("n_pending"):
            print(f"  유보 (미완성 클립): {m['n_pending']}개 파일")
        return

    # ── --overwrite: prepare/ 파일을 collected/ 로 복원 ─────────────────────
    if args.overwrite:
        restored = 0
        for sub in _SPLIT_DIRS:
            for h5 in (prepare_dir / sub).glob("*.h5"):
                dst = collected_dir / h5.name
                if not dst.exists():
                    shutil.move(str(h5), str(dst))
                    restored += 1
                png = (prepare_dir / sub / h5.stem).with_suffix(".png")
                if png.exists():
                    dst_png = viz_dir / png.name
                    if not dst_png.exists():
                        shutil.move(str(png), str(dst_png))
        if manifest_path.exists():
            manifest_path.unlink()
        log.info("--overwrite: prepare/ 에서 collected/ 로 %d개 복원", restored)

    # ── h5 파일 스캔 ────────────────────────────────────────────────────────
    collected_dir.mkdir(parents=True, exist_ok=True)
    all_h5 = sorted(collected_dir.glob("*.h5"))
    if not all_h5:
        log.error("처리할 h5 파일 없음: %s", collected_dir)
        return
    log.info("collected/ 에서 %d개 h5 파일 발견", len(all_h5))

    # ── 미완성 클립 감지 ─────────────────────────────────────────────────────
    pending_stems: set[str] = set()   # 미완성 클립 소속 파일 (이번 실행에서 제외)

    if args.clips_csv:
        expected = _load_expected_clips(args.clips_csv)
        complete_ids, incomplete_ids = _find_incomplete_clips(collected_dir, expected)

        if incomplete_ids:
            log.info(
                "미완성 클립 %d개 제외 (수집 중): %s%s",
                len(incomplete_ids),
                ", ".join(sorted(incomplete_ids)[:5]),
                "..." if len(incomplete_ids) > 5 else "",
            )
            # 미완성 클립 소속 파일은 이번 실행에서 건너뜀
            for h5 in all_h5:
                clip_id = h5.stem.split("__")[0]
                if clip_id in incomplete_ids:
                    pending_stems.add(h5.stem)
        log.info(
            "완성 클립 %d개 / 미완성 %d개 (유보 파일 %d개)",
            len(complete_ids), len(incomplete_ids), len(pending_stems),
        )

    # 이번 실행에서 처리할 파일
    target_h5 = [p for p in all_h5 if p.stem not in pending_stems]
    if not target_h5:
        log.warning("처리 대상 파일 없음 (전부 미완성 클립 또는 빈 디렉토리)")
        return
    log.info("처리 대상: %d개 / 유보: %d개", len(target_h5), len(pending_stems))

    # ── 지표 계산 ────────────────────────────────────────────────────────────
    use_gpu = args.use_gpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                log.warning("CUDA 사용 불가 — CPU 모드로 폴백")
                use_gpu = False
        except ImportError:
            log.warning("torch 미설치 — CPU 모드로 폴백")
            use_gpu = False

    metrics: dict[str, dict] = {}   # stem → {지표, path}

    if use_gpu:
        metrics = _run_gpu_batch(target_h5, args.workers)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_compute_metrics, str(p)): p for p in target_h5}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="샘플"):
                stem, m = fut.result()
                if m is not None:
                    m["path"] = futures[fut]
                    metrics[stem] = m

    n_failed = len(target_h5) - len(metrics)
    log.info("지표 계산 완료: %d개 성공 / %d개 실패", len(metrics), n_failed)

    # ── 필터링 ───────────────────────────────────────────────────────────────
    error_stems: list[str] = []
    valid_stems: list[str] = []
    for stem, m in metrics.items():
        if (m["ade"]     > args.ade_thresh or
                m["lon_ade"] > args.lon_thresh or
                m["lat_ade"] > args.lat_thresh or
                m["v_mean"]  < args.min_speed):
            error_stems.append(stem)
        else:
            valid_stems.append(stem)

    # ── 통계 출력 ────────────────────────────────────────────────────────────
    vals = list(metrics.values())
    print("\n" + "═" * 70)
    print(f"  collected/ 전체    : {len(all_h5):>7,}")
    print(f"  유보 (미완성 클립) : {len(pending_stems):>7,}")
    print(f"  처리 대상          : {len(target_h5):>7,}")
    print(f"  지표 계산 실패     : {n_failed:>7,}")
    print(f"  오류 → error/      : {len(error_stems):>7,}  "
          f"(ade>{args.ade_thresh} OR lon>{args.lon_thresh} "
          f"OR lat>{args.lat_thresh} OR v_mean<{args.min_speed})")
    print(f"  유효               : {len(valid_stems):>7,}")
    if vals:
        print()
        _print_stat("v_mean (호라이즌)", [m["v_mean"] for m in vals], "m/s")
        _print_stat("ADE",              [m["ade"]     for m in vals], "m")
        _print_stat("FDE",              [m["fde"]     for m in vals], "m")
        _print_stat("Longitudinal ADE", [m["lon_ade"] for m in vals], "m")
        _print_stat("Lateral ADE",      [m["lat_ade"] for m in vals], "m")
        hs_vals = [m["hs_norm"] for m in vals if m["hs_norm"] is not None]
        if hs_vals:
            _print_stat("hs_norm", hs_vals)
    print("═" * 70)

    # ── Stratified 분할 ──────────────────────────────────────────────────────
    def _clip_id(stem: str) -> str:
        return stem.split("__")[0]

    clips: dict[str, list[str]] = defaultdict(list)
    for stem in valid_stems:
        clips[_clip_id(stem)].append(stem)

    clip_ids = sorted(clips.keys())
    n_clips  = len(clip_ids)

    clip_lat_max = {
        cid: max(metrics[s]["lat_max"] for s in clips[cid])
        for cid in clip_ids
    }
    lat_clips = [cid for cid in clip_ids if clip_lat_max[cid] >  args.split_lat_thresh]
    lon_clips = [cid for cid in clip_ids if clip_lat_max[cid] <= args.split_lat_thresh]

    random.seed(args.seed)
    random.shuffle(lat_clips)
    random.shuffle(lon_clips)

    def _split(cids: list[str]) -> tuple[list[str], list[str], list[str]]:
        n_tr = int(len(cids) * args.train)
        n_va = int(len(cids) * args.val)
        return cids[:n_tr], cids[n_tr:n_tr + n_va], cids[n_tr + n_va:]

    lat_tr, lat_va, lat_te = _split(lat_clips)
    lon_tr, lon_va, lon_te = _split(lon_clips)

    clip_splits: dict[str, list[str]] = {
        "train": lat_tr + lon_tr,
        "val":   lat_va + lon_va,
        "test":  lat_te + lon_te,
    }
    stem_splits: dict[str, list[str]] = {
        name: sorted(s for cid in cids for s in clips[cid])
        for name, cids in clip_splits.items()
    }
    n_lat_clips = {"train": len(lat_tr), "val": len(lat_va), "test": len(lat_te)}

    print(f"\n분할 (seed={args.seed}, stratified, split_lat_thresh={args.split_lat_thresh}m)")
    print(f"  lateral {len(lat_clips):,}개 / longitudinal {len(lon_clips):,}개 클립")
    for name in ("train", "val", "test"):
        nc    = len(clip_splits[name])
        nf    = len(stem_splits[name])
        lat_r = n_lat_clips[name] / max(nc, 1)
        print(f"  {name:5s}: 클립 {nc:,}개  파일 {nf:,}개  lateral={n_lat_clips[name]:,} ({lat_r:.1%})")
    print(f"  error:  파일 {len(error_stems):,}개")
    if pending_stems:
        print(f"  유보 (다음 실행):  파일 {len(pending_stems):,}개  → collected/ 에 유지")

    if args.dry_run:
        print("\n[dry-run] 파일 이동 생략")
        return

    # ── 파일 이동 ────────────────────────────────────────────────────────────
    for sub in _SPLIT_DIRS:
        (prepare_dir / sub).mkdir(parents=True, exist_ok=True)

    def _move(stem: str, dest_dir: Path) -> None:
        src = metrics[stem]["path"]
        shutil.move(str(src), str(dest_dir / src.name))
        png = (viz_dir / src.stem).with_suffix(".png")
        if png.exists():
            shutil.move(str(png), str(dest_dir / png.name))

    for stem in tqdm(error_stems, desc="error/ 이동"):
        _move(stem, prepare_dir / "error")

    for split_name, stems in stem_splits.items():
        for stem in tqdm(stems, desc=f"{split_name}/ 이동"):
            _move(stem, prepare_dir / split_name)

    # ── 매니페스트 저장 ───────────────────────────────────────────────────────
    manifest = {
        "filter": {
            "ade_thresh": args.ade_thresh,
            "lon_thresh": args.lon_thresh,
            "lat_thresh": args.lat_thresh,
            "min_speed":  args.min_speed,
        },
        "split": {
            "ratios":           {"train": args.train, "val": args.val,
                                 "test": round(1 - args.train - args.val, 4)},
            "seed":             args.seed,
            "split_lat_thresh": args.split_lat_thresh,
            "split_unit":       "clip_stratified",
        },
        "n_total":         len(all_h5),
        "n_pending":       len(pending_stems),
        "n_processed":     len(target_h5),
        "n_error":         len(error_stems),
        "n_valid":         len(valid_stems),
        "n_clips_total":   n_clips,
        "n_clips_train":   len(clip_splits["train"]),
        "n_clips_val":     len(clip_splits["val"]),
        "n_clips_test":    len(clip_splits["test"]),
        "n_train":         len(stem_splits["train"]),
        "n_val":           len(stem_splits["val"]),
        "n_test":          len(stem_splits["test"]),
        "lat_ratio_train": round(n_lat_clips["train"] / max(len(clip_splits["train"]), 1), 4),
        "lat_ratio_val":   round(n_lat_clips["val"]   / max(len(clip_splits["val"]),   1), 4),
        "lat_ratio_test":  round(n_lat_clips["test"]  / max(len(clip_splits["test"]),  1), 4),
        "clips_train":     clip_splits["train"],
        "clips_val":       clip_splits["val"],
        "clips_test":      clip_splits["test"],
        "error_stems":     error_stems,
        "pending_stems":   sorted(pending_stems),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("완료: %s", prepare_dir.resolve())
    log.info("매니페스트: %s", manifest_path)


if __name__ == "__main__":
    main()
