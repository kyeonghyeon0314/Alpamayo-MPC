#!/usr/bin/env python3
"""
데이터셋 분할 스크립트

labels/valid == True이고 cotend가 존재하는 h5 파일을
클립(UUID) 단위로 70/20/10 Stratified 분할해 train/, val/, test/ 디렉토리에
심볼릭 링크를 생성합니다. 원본 파일은 이동하지 않습니다.

Stratified 분할:
  lateral 클립과 longitudinal 클립을 각각 비율대로 나눈 뒤 합산.
  → train/val/test 모두 lateral 비율이 동일하게 유지됨.
  → 클립 내 임의의 샘플 중 max(|GT_y[:20]|) > lat_thresh 이면 lateral 클립으로 분류.

파일명 규칙: {clip_uuid}__{timestamp}.h5
  → 같은 clip_uuid를 가진 파일은 반드시 같은 split에 배치됩니다.
  → 데이터 누수(data leakage) 방지.

실행:
    python alpamayo_dataset/split_dataset.py \\
        --data-dir alpamayo_dataset/data/collected \\
        --out-dir  alpamayo_dataset/data/split

    # 비율 변경
    python alpamayo_dataset/split_dataset.py \\
        --data-dir alpamayo_dataset/data/collected \\
        --out-dir  alpamayo_dataset/data/split \\
        --train 0.70 --val 0.20 --seed 42

결과:
    out-dir/
        train/   *.h5 → (symlink to original)
        val/     *.h5 → (symlink to original)
        test/    *.h5 → (symlink to original)
        split_manifest.json
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


def is_valid(path: Path) -> bool:
    """라벨링 완료 + valid + cotend 존재 여부 확인."""
    try:
        with h5py.File(path, "r") as f:
            return (
                "labels/mpc_weights" in f
                and "output/cotend_hidden_state" in f
                and bool(f["labels/valid"][()])
            )
    except Exception:
        return False


def _scan_file(path_str: str) -> tuple[bool, float]:
    """ProcessPoolExecutor용 — (is_valid, max_lat) 반환. 실패 시 (False, 0.0)."""
    try:
        with h5py.File(path_str, "r") as f:
            valid = (
                "labels/mpc_weights" in f
                and "output/cotend_hidden_state" in f
                and bool(f["labels/valid"][()])
            )
            if not valid:
                return False, 0.0
            lat_y = f["gt/future_xyz"][:20, 1]   # GT 20스텝 횡방향 변위 [m]
        return True, float(np.max(np.abs(lat_y)))
    except Exception:
        return False, 0.0


def clip_id(path: Path) -> str:
    """파일명에서 클립 UUID 추출. {clip_uuid}__{timestamp}.h5 → clip_uuid."""
    return path.stem.split("__")[0]


def main():
    parser = argparse.ArgumentParser(
        description="h5 데이터셋 클립 단위 train/val/test Stratified 분할",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir",  required=True, help="원본 .h5 파일 디렉토리")
    parser.add_argument("--out-dir",   required=True, help="분할 결과 디렉토리")
    parser.add_argument("--train",     type=float, default=0.70, help="학습 비율 (default: 0.70)")
    parser.add_argument("--val",       type=float, default=0.20, help="검증 비율 (default: 0.20)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--workers",   type=int,   default=8,  help="병렬 스캔 프로세스 수 (default: 8)")
    parser.add_argument("--lat-thresh",type=float, default=1.0,
                        help="lateral 클립 판별 임계값: 클립 내 max(|GT_y[:20]|) [m] (default: 1.0)")
    parser.add_argument("--overwrite", action="store_true", help="기존 분할 덮어쓰기")
    args = parser.parse_args()

    assert args.train + args.val < 1.0, \
        "--train + --val 은 1.0 미만이어야 합니다 (나머지가 test)"

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "split_manifest.json"

    if manifest_path.exists() and not args.overwrite:
        print(f"[INFO] 이미 분할됨: {manifest_path}")
        print("       재분할하려면 --overwrite 사용")
        with open(manifest_path) as f:
            m = json.load(f)
        print(f"       클립: train={m['n_clips_train']}  val={m['n_clips_val']}  test={m['n_clips_test']}")
        print(f"       파일: train={m['n_train']}  val={m['n_val']}  test={m['n_test']}")
        if "lat_ratio_train" in m:
            print(f"       lateral 비율: train={m['lat_ratio_train']:.1%}"
                  f"  val={m['lat_ratio_val']:.1%}  test={m['lat_ratio_test']:.1%}")
        return

    # ── 유효 파일 스캔 (+ max_lat 동시 수집) ───────────
    all_files = sorted(Path(args.data_dir).glob("*.h5"))
    print(f"전체 .h5 파일: {len(all_files):,}개")
    print(f"유효 파일 스캔 중... (workers={args.workers})", end=" ", flush=True)

    path_strs = [str(p) for p in all_files]
    chunksize  = max(1, len(path_strs) // (args.workers * 4))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        scan_results = list(ex.map(_scan_file, path_strs, chunksize=chunksize))

    valid_files: list[Path] = []
    file_max_lat: dict[Path, float] = {}
    for p, (ok, ml) in zip(all_files, scan_results):
        if ok:
            valid_files.append(p)
            file_max_lat[p] = ml
    print(f"{len(valid_files):,}개")

    if not valid_files:
        print("[ERROR] 유효 샘플이 없습니다. 라벨링을 먼저 실행하세요.")
        return

    # ── 클립 단위 그룹화 ───────────────────────────────
    clips: dict[str, list[Path]] = defaultdict(list)
    for p in valid_files:
        clips[clip_id(p)].append(p)

    clip_ids = sorted(clips.keys())
    n_clips  = len(clip_ids)
    print(f"고유 클립: {n_clips:,}개  (클립당 평균 {len(valid_files)/n_clips:.1f}개 파일)")

    # ── 클립별 lateral 분류 ────────────────────────────
    # 클립 내 샘플 max_lat 중 최대값 > lat_thresh → lateral 클립
    clip_max_lat: dict[str, float] = {
        cid: max(file_max_lat[p] for p in clips[cid])
        for cid in clip_ids
    }
    lat_clips = [cid for cid in clip_ids if clip_max_lat[cid] > args.lat_thresh]
    lon_clips = [cid for cid in clip_ids if clip_max_lat[cid] <= args.lat_thresh]
    print(f"  lateral 클립: {len(lat_clips):,}개 ({len(lat_clips)/n_clips*100:.1f}%)"
          f"  longitudinal 클립: {len(lon_clips):,}개 ({len(lon_clips)/n_clips*100:.1f}%)"
          f"  (임계값={args.lat_thresh}m)")

    # ── Stratified 분할 ────────────────────────────────
    random.seed(args.seed)
    random.shuffle(lat_clips)
    random.shuffle(lon_clips)

    def _split(cids: list[str]) -> tuple[list[str], list[str], list[str]]:
        n    = len(cids)
        n_tr = int(n * args.train)
        n_va = int(n * args.val)
        return cids[:n_tr], cids[n_tr:n_tr + n_va], cids[n_tr + n_va:]

    lat_tr, lat_va, lat_te = _split(lat_clips)
    lon_tr, lon_va, lon_te = _split(lon_clips)

    clip_splits = {
        "train": lat_tr + lon_tr,
        "val":   lat_va + lon_va,
        "test":  lat_te + lon_te,
    }

    # 각 split에 속하는 파일 목록
    file_splits: dict[str, list[Path]] = {
        name: sorted(p for cid in cids for p in clips[cid])
        for name, cids in clip_splits.items()
    }

    n_total = len(valid_files)
    n_lat_clips = {
        "train": len(lat_tr),
        "val":   len(lat_va),
        "test":  len(lat_te),
    }

    print(f"\n분할 결과 (seed={args.seed}, stratified by lateral, lat_thresh={args.lat_thresh}m)")
    for name in ("train", "val", "test"):
        nc      = len(clip_splits[name])
        nf      = len(file_splits[name])
        n_lat_c = n_lat_clips[name]
        lat_r   = n_lat_c / max(nc, 1)
        print(f"  {name:5s}: 클립 {nc:,}개  파일 {nf:,}개  ({nf/n_total*100:.1f}%)"
              f"  lateral={n_lat_c:,}클립 ({lat_r:.1%})")

    # ── 심볼릭 링크 생성 ───────────────────────────────
    for split_name, file_list in file_splits.items():
        split_dir = out_dir / split_name
        if split_dir.exists() and args.overwrite:
            for lnk in split_dir.glob("*.h5"):
                lnk.unlink()
        split_dir.mkdir(parents=True, exist_ok=True)

        for src in file_list:
            dst = split_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        print(f"  {split_name}/  → {len(file_list):,}개 링크 생성")

    # ── 매니페스트 저장 ────────────────────────────────
    n_train = len(file_splits["train"])
    n_val   = len(file_splits["val"])
    n_test  = len(file_splits["test"])

    n_clips_train = len(clip_splits["train"])
    n_clips_val   = len(clip_splits["val"])
    n_clips_test  = len(clip_splits["test"])

    manifest = {
        "seed":             args.seed,
        "split_unit":       "clip_stratified",
        "lat_thresh":       args.lat_thresh,
        "n_clips_total":    n_clips,
        "n_clips_train":    n_clips_train,
        "n_clips_val":      n_clips_val,
        "n_clips_test":     n_clips_test,
        "n_total":          n_total,
        "n_train":          n_train,
        "n_val":            n_val,
        "n_test":           n_test,
        "ratios":           {"train": args.train, "val": args.val,
                             "test": round(1 - args.train - args.val, 4)},
        "n_lat_clips_total": len(lat_clips),
        "n_lat_clips_train": len(lat_tr),
        "n_lat_clips_val":   len(lat_va),
        "n_lat_clips_test":  len(lat_te),
        "lat_ratio_train":   round(len(lat_tr) / max(n_clips_train, 1), 4),
        "lat_ratio_val":     round(len(lat_va) / max(n_clips_val,   1), 4),
        "lat_ratio_test":    round(len(lat_te) / max(n_clips_test,  1), 4),
        "clips_train":       clip_splits["train"],
        "clips_val":         clip_splits["val"],
        "clips_test":        clip_splits["test"],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n완료: {out_dir.resolve()}")
    print(f"매니페스트: {manifest_path}")


if __name__ == "__main__":
    main()
