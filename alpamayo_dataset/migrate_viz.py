#!/usr/bin/env python3
"""PNG 시각화 파일이 h5와 같은 split 폴더에 있는지 검증하는 스크립트.

검증 기준:
  - prepare/{split}/ 안의 .h5 마다 동명 .png 가 같은 폴더에 있어야 함
  - 반대로 .png 가 있는데 .h5 가 없는 경우도 검출

실행:
    python alpamayo_dataset/migrate_viz.py
    python alpamayo_dataset/migrate_viz.py --show-missing   # 누락 목록 출력
"""

import argparse
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
PREPARE_DIR = SCRIPT_DIR / "data" / "prepare"
SPLIT_DIRS  = ("train", "val", "test", "error")


def main() -> None:
    parser = argparse.ArgumentParser(description="PNG/h5 매칭 검증")
    parser.add_argument("--prepare-dir",  default=str(PREPARE_DIR))
    parser.add_argument("--show-missing", action="store_true",
                        help="PNG 누락 파일 목록 출력")
    args = parser.parse_args()

    prepare_dir = Path(args.prepare_dir)

    total_h5 = total_png = 0
    missing_png: list[str] = []   # h5는 있는데 PNG 없음
    orphan_png:  list[str] = []   # PNG는 있는데 h5 없음

    for split in SPLIT_DIRS:
        split_dir = prepare_dir / split
        if not split_dir.exists():
            continue

        h5_stems  = {p.stem for p in split_dir.glob("*.h5")}
        png_stems = {p.stem for p in split_dir.glob("*.png")}

        total_h5  += len(h5_stems)
        total_png += len(png_stems)

        for stem in sorted(h5_stems - png_stems):
            missing_png.append(f"{split}/{stem}.png")
        for stem in sorted(png_stems - h5_stems):
            orphan_png.append(f"{split}/{stem}.png")

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  h5  파일 합계 : {total_h5:>7,}")
    print(f"  PNG 파일 합계 : {total_png:>7,}")
    print(f"  PNG 누락      : {len(missing_png):>7,}  (h5 있음, PNG 없음)")
    print(f"  PNG 고아      : {len(orphan_png):>7,}  (PNG 있음, h5 없음)")
    print(f"{'─'*50}")

    if not missing_png and not orphan_png:
        print("  ✓ 모든 h5와 PNG가 1:1로 매칭됩니다.")
    else:
        if args.show_missing and missing_png:
            print(f"\n[PNG 누락 목록 (최대 20개)]")
            for p in missing_png[:20]:
                print(f"  {p}")
            if len(missing_png) > 20:
                print(f"  ... 외 {len(missing_png) - 20}개")
        if orphan_png:
            print(f"\n[고아 PNG (최대 20개)]")
            for p in orphan_png[:20]:
                print(f"  {p}")


if __name__ == "__main__":
    main()
