#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""수집 대상 (clip_id, t0_us) 쌍을 CSV로 생성.

CSV는 클립 순서로 정렬됩니다:
    clip_A, t0=2s
    clip_A, t0=4s
    clip_A, t0=6s
    clip_B, t0=2s
    ...
→ collect_dataset.py가 한 클립의 모든 샘플을 연속으로 처리합니다.

사용 예
-------
# 기본값 (200 클립 × 6 시점 = 1,200 샘플)
python /workspace/alpamayo_dataset/make_clips_csv.py

# 클립 수·시점 간격 조정
python /workspace/alpamayo_dataset/make_clips_csv.py \\
    --max-clips 500 \\
    --t0-start 3000000 --t0-end 12000000 --t0-step 1000000

# 국가·시간대 필터
python /workspace/alpamayo_dataset/make_clips_csv.py \\
    --country "United States" \\
    --hour-min 7 --hour-max 19

# split 지정 (train / val / test)
python /workspace/alpamayo_dataset/make_clips_csv.py --split train

# 이미 수집된 샘플 제외 (재실행 시 중복 방지)
python /workspace/alpamayo_dataset/make_clips_csv.py \\
    --skip-existing /workspace/alpamayo_dataset/data/collected
"""

import argparse
import csv
import pathlib

import pandas as pd


# ── 기본 경로 ─────────────────────────────────────────────────────────────────
_DATASET_DIR = pathlib.Path("/workspace/alpamayo_dataset/data/nvidia_test_scenario")
_OUTPUT_CSV  = pathlib.Path("/workspace/alpamayo_dataset/data/clips.csv")

# t0_us 유효 범위 (클립 내 상대 타임스탬프, 마이크로초)
#   클립 길이: 20,000,000 us (20초)
#   히스토리: 1,600,000 us (1.6초)
#   미래:     6,400,000 us (6.4초)
_T0_MIN = 1_600_000
_T0_MAX = 13_600_000   # 20_000_000 − 6_400_000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="수집 대상 (clip_id, t0_us) CSV 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 경로 ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--dataset-dir",
        default=str(_DATASET_DIR),
        metavar="DIR",
        help=f"로컬 데이터셋 디렉토리 (기본: {_DATASET_DIR})",
    )
    p.add_argument(
        "--output",
        default=str(_OUTPUT_CSV),
        metavar="FILE",
        help=f"출력 CSV 경로 (기본: {_OUTPUT_CSV})",
    )

    # ── 클립 필터 ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--max-clips",
        type=int,
        default=200,
        metavar="N",
        help="최대 클립 수 (0 = 전체, 기본: 200)",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
        help=(
            "데이터셋 split 필터 (기본: all). "
            "train=153,625 / val=90,928 / test=61,599 / all=306,152"
        ),
    )
    p.add_argument(
        "--country",
        default=None,
        metavar="NAME",
        help=(
            "국가 필터 (대소문자 일치). 미지정 시 전체. "
            "선택 가능: 'United States'(155360) 'Germany'(43900) 'France'(10364) "
            "'Italy'(8658) 'Sweden'(7330) 'Spain'(6459) 'Portugal'(6101) "
            "'Greece'(5885) 'Austria'(5451) 'Finland'(5176) 외 15개국"
        ),
    )
    p.add_argument(
        "--hour-min",
        type=int,
        default=None,
        metavar="H",
        help="수집 시간대 하한 (0~23). 예: 7 → 7시 이후",
    )
    p.add_argument(
        "--hour-max",
        type=int,
        default=None,
        metavar="H",
        help="수집 시간대 상한 (0~23). 예: 19 → 19시 이전",
    )
    p.add_argument(
        "--platform",
        default=None,
        metavar="CLASS",
        help="플랫폼 클래스 필터. 'hyperion_8'(87,564개) 또는 'hyperion_8.1'(218,588개). 미지정 시 전체",
    )

    # ── t0_us 샘플링 ──────────────────────────────────────────────────────────
    p.add_argument(
        "--t0-start",
        type=int,
        default=2_000_000,
        metavar="US",
        help=f"첫 번째 t0_us (기본: 1,600,000). 최소 {_T0_MIN:,}",
    )
    p.add_argument(
        "--t0-end",
        type=int,
        default=13_600_000,
        metavar="US",
        help=f"마지막 t0_us 포함 상한 (기본: 13,600,000). 최대 {_T0_MAX:,}",
    )
    p.add_argument(
        "--t0-step",
        type=int,
        default=1_000_000,
        metavar="US",
        help="t0_us 간격 (기본: 1,000,000 = 1초)",
    )

    # ── 기타 ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--skip-existing",
        default=None,
        metavar="DIR",
        help="이미 수집된 .h5 파일이 있는 디렉토리. 해당 (clip_id, t0_us)는 CSV에서 제외",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)

    # ── 메타데이터 로드 ───────────────────────────────────────────────────────
    print("메타데이터 로드 중...")
    clip_index  = pd.read_parquet(dataset_dir / "clip_index.parquet")
    sensor_pres = pd.read_parquet(dataset_dir / "metadata/feature_presence.parquet")
    data_coll   = pd.read_parquet(dataset_dir / "metadata/data_collection.parquet")

    # ── 필터 적용 ─────────────────────────────────────────────────────────────
    # 1. 유효 클립
    mask = clip_index["clip_is_valid"].copy()
    print(f"  유효 클립:        {mask.sum():>7,}")

    # 2. Alpamayo 필수 카메라 4개 모두 존재
    required_cameras = [
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    ]
    mask &= sensor_pres[required_cameras].all(axis=1)
    print(f"  필수 카메라 보유:  {mask.sum():>7,}")

    # 3. split 필터
    if args.split != "all":
        mask &= clip_index["split"] == args.split
        print(f"  split={args.split}:        {mask.sum():>7,}")

    # 4. 국가 필터
    if args.country:
        mask &= data_coll["country"] == args.country
        print(f"  country={args.country!r}: {mask.sum():>7,}")

    # 5. 시간대 필터
    if args.hour_min is not None:
        mask &= data_coll["hour_of_day"] >= args.hour_min
        print(f"  hour >= {args.hour_min}:          {mask.sum():>7,}")
    if args.hour_max is not None:
        mask &= data_coll["hour_of_day"] <= args.hour_max
        print(f"  hour <= {args.hour_max}:          {mask.sum():>7,}")

    # 6. 플랫폼 필터
    if args.platform:
        mask &= data_coll["platform_class"] == args.platform
        print(f"  platform={args.platform!r}: {mask.sum():>7,}")

    valid_clips = clip_index[mask].index.tolist()
    print(f"\n필터 적용 후 클립 수: {len(valid_clips):,}")

    # ── 클립 수 제한 ─────────────────────────────────────────────────────────
    if args.max_clips and args.max_clips < len(valid_clips):
        valid_clips = valid_clips[: args.max_clips]
        print(f"클립 수 제한 적용:   {len(valid_clips):,}")

    # ── t0_us 목록 생성 ───────────────────────────────────────────────────────
    t0_start = max(args.t0_start, _T0_MIN)
    t0_end   = min(args.t0_end,   _T0_MAX)
    t0_values = list(range(t0_start, t0_end + 1, args.t0_step))
    print(f"t0_us 시점:         {[f'{t/1e6:.1f}s' for t in t0_values]}")

    # ── 이미 수집된 샘플 파악 ─────────────────────────────────────────────────
    existing: set[tuple[str, int]] = set()
    if args.skip_existing:
        collected_dir = pathlib.Path(args.skip_existing)
        for h5 in collected_dir.glob("*.h5"):
            # 파일명 형식: {clip_id}__{t0_us}.h5
            parts = h5.stem.split("__")
            if len(parts) == 2:
                try:
                    existing.add((parts[0], int(parts[1])))
                except ValueError:
                    pass
        print(f"기존 수집 샘플 수:  {len(existing):,}  (제외 예정)")

    # ── CSV 생성: 클립 단위로 묶인 순서 ──────────────────────────────────────
    # clip_A, t0=2s → clip_A, t0=4s → clip_A, t0=6s → clip_B, t0=2s → ...
    rows = []
    for clip_id in valid_clips:
        for t0_us in t0_values:
            if (clip_id, t0_us) not in existing:
                rows.append({"clip_id": clip_id, "t0_us": t0_us})

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "t0_us"])
        writer.writeheader()
        writer.writerows(rows)

    skipped = len(valid_clips) * len(t0_values) - len(rows)
    print(f"\n생성 완료: {len(rows):,}개 샘플 → {output_path}")
    if skipped:
        print(f"  (기존 수집분 {skipped:,}개 제외)")
    print(f"\n수집 명령:")
    print(f"  python /workspace/collect_dataset.py \\")
    print(f"      --csv {output_path} \\")
    print(f"      --output-dir /workspace/alpamayo_dataset/data/collected")


if __name__ == "__main__":
    main()
