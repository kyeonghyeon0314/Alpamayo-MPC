#!/usr/bin/env python3
"""
prepare/test/ 의 기존 test 샘플 사이 빈 타임스텝을 0.1초 간격으로 채우는 스크립트.

기존 collect_dataset.py는 클립당 4개 타임스텝만 수집 (약 0.5초 간격).
이 스크립트는 동일 test 클립의 전체 유효 구간을 step_us 간격으로 추론하여
prepare/test/ 에 동일 포맷({clip_id}__{t0_us}.h5)으로 저장한다.

실행 예시:
  python closed-loop-simul/build_pcl_dataset.py \\
      --prepare-dir alpamayo_dataset/data/prepare \\
      --model-dir   alpamayo_dataset/data/alpamayo_1.5_weights \\
      --dataset-dir alpamayo_dataset/data/nvidia_test_scenario

  # dry-run: 실제 추론 없이 처리 예정 타임스텝 수만 출력
  python closed-loop-simul/build_pcl_dataset.py \\
      --prepare-dir alpamayo_dataset/data/prepare \\
      --model-dir   alpamayo_dataset/data/alpamayo_1.5_weights \\
      --dataset-dir alpamayo_dataset/data/nvidia_test_scenario \\
      --n-clips 5 --dry-run
"""

import argparse
import contextlib
import json
import logging
import os
import pathlib
import queue
import sys
import threading

import torch

# TOKENIZERS_PARALLELISM: multiprocessing.Process 포크 전에 설정
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── 경로 설정: 워크스페이스 루트를 sys.path에 추가 ──────────────────────
_WORKSPACE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE))

_STEP_US    = 100_000   # 기본 샘플링 간격: 100ms
_HIST_DUR_US = 16 * _STEP_US   # 히스토리 1.6초
_FUT_DUR_US  = 64 * _STEP_US   # 미래 6.4초

_SENTINEL = object()


# ══════════════════════════════════════════════════════
# 로컬 AVDI (HuggingFace 없이 오프라인 동작)
# ══════════════════════════════════════════════════════

def _make_local_avdi(dataset_dir: str):
    import pandas as pd
    import physical_ai_av

    class _LocalFirstAVDI(physical_ai_av.PhysicalAIAVDatasetInterface):
        def download_file(self, filename: str, **kwargs) -> str:
            if filename == "metadata/sensor_presence.parquet":
                filename = "metadata/feature_presence.parquet"
            p = pathlib.Path(self.local_dir) / filename
            if p.exists():
                return str(p)
            return super().download_file(filename, **kwargs)

        @contextlib.contextmanager
        def open_file(self, filename: str, mode: str = "rb", maybe_stream: bool = False):
            p = pathlib.Path(self.local_dir) / filename
            if p.exists():
                with open(p, mode) as f:
                    yield f
            else:
                with super().open_file(filename, mode=mode, maybe_stream=maybe_stream) as f:
                    yield f

        def download_metadata(self) -> None:
            meta_dir = pathlib.Path(self.local_dir) / "metadata"
            if meta_dir.exists():
                self.metadata = {
                    p.stem: pd.read_parquet(p)
                    for p in sorted(meta_dir.glob("*.parquet"))
                }
            else:
                super().download_metadata()

    avdi = _LocalFirstAVDI(local_dir=dataset_dir, revision="main")
    avdi.download_metadata()
    return avdi


# ══════════════════════════════════════════════════════
# 다운로드 워커
# ══════════════════════════════════════════════════════

def _download_worker(
    worker_id: int,
    work_q: queue.Queue,
    result_q: queue.Queue,
    avdi,
    test_dir: pathlib.Path,
    step_us: int,
    overwrite: bool,
) -> None:
    """클립 피처를 1회 로드 후 미수집 t0_us별 raw 데이터를 result_q에 적재."""
    from dataset.load_physical_aiavdataset import (
        extract_sample_from_clip_features,
        load_clip_features,
    )

    log = logging.getLogger(__name__)

    while True:
        item = work_q.get()
        if item is _SENTINEL:
            break

        clip_id: str = item

        # ── 클립 피처 로드 ────────────────────────────────────────
        log.info("[W%d] 클립 로드 중: %s", worker_id, clip_id[:8])
        try:
            clip_ctx = load_clip_features(clip_id, avdi=avdi, maybe_stream=True)
        except Exception as exc:
            log.error("[W%d] 클립 피처 로드 실패 %s: %s", worker_id, clip_id[:8], exc)
            result_q.put({"type": "clip_error", "clip_id": clip_id, "exc": exc})
            continue

        # ── 유효 t0 범위 계산 ──────────────────────────────────────
        ego = clip_ctx["egomotion"]
        clip_start_us, ego_end_us = ego.time_range

        cam_end_us = min(
            int(cam.timestamps[-1])
            for cam in clip_ctx["cameras"]
            if cam.timestamps is not None and len(cam.timestamps) > 0
        )
        clip_end_us = min(ego_end_us, cam_end_us)

        t0_min = clip_start_us + _HIST_DUR_US
        t0_min = -(-t0_min // step_us) * step_us  # step_us 배수로 올림 정렬
        t0_max = clip_end_us   - _FUT_DUR_US

        if t0_min > t0_max:
            log.warning("[W%d] 클립 너무 짧음 (%s): %.1fs",
                        worker_id, clip_id[:8], (clip_end_us - clip_start_us) / 1e6)
            result_q.put({"type": "clip_error", "clip_id": clip_id,
                          "exc": RuntimeError("클립 길이 부족")})
            continue

        t0_list = list(range(t0_min, t0_max + 1, step_us))

        # 이미 존재하는 파일 제외
        pending = [
            t for t in t0_list
            if overwrite or not (test_dir / f"{clip_id}__{t}.h5").exists()
        ]

        log.info("[W%d] %s: %.1f~%.1fs  전체 %d개 / 신규 %d개",
                 worker_id, clip_id[:8],
                 t0_min / 1e6, t0_max / 1e6, len(t0_list), len(pending))

        if not pending:
            result_q.put({"type": "clip_done", "clip_id": clip_id,
                          "n_new": 0, "n_skip": len(t0_list), "n_err": 0})
            continue

        # ── 각 t0_us별 raw 추출 ────────────────────────────────────
        n_ok = n_err = 0
        for t0_us in pending:
            try:
                raw = extract_sample_from_clip_features(clip_ctx, t0_us)
                result_q.put({
                    "type":    "frame",
                    "clip_id": clip_id,
                    "t0_us":   t0_us,
                    "raw":     raw,
                })
                n_ok += 1
            except Exception as exc:
                log.warning("[W%d] %s t0=%.1fs 추출 실패: %s",
                            worker_id, clip_id[:8], t0_us / 1e6, exc)
                n_err += 1

        result_q.put({
            "type":   "clip_done",
            "clip_id": clip_id,
            "n_new":   n_ok,
            "n_skip":  len(t0_list) - len(pending),
            "n_err":   n_err,
        })


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="prepare/test/ 빈 타임스텝 채우기 (0.1s 간격)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--prepare-dir",
        default="./alpamayo_dataset/data/prepare",
        metavar="DIR",
        help="prepare/ 디렉토리 경로 (기본: ./alpamayo_dataset/data/prepare)",
    )
    p.add_argument(
        "--manifest", default=None, metavar="JSON",
        help="dataset_manifest.json 경로 (기본: prepare-dir/dataset_manifest.json)",
    )
    p.add_argument(
        "--model-dir",
        default="./alpamayo_dataset/data/alpamayo_1.5_weights",
        metavar="DIR",
    )
    p.add_argument(
        "--dataset-dir",
        default="./alpamayo_dataset/data/nvidia_test_scenario",
        metavar="DIR",
    )
    p.add_argument(
        "--step-us", type=int, default=_STEP_US, metavar="US",
        help=f"샘플링 간격 [μs] (기본: {_STEP_US} = 100ms)",
    )
    p.add_argument(
        "--n-clips", type=int, default=0, metavar="N",
        help="처리할 클립 수 (0 = 전체 test 클립)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="이미 존재하는 샘플도 재수집",
    )
    p.add_argument(
        "--download-workers", type=int, default=2, metavar="N",
    )
    p.add_argument(
        "--prefetch", type=int, default=8, metavar="N",
        help="result_q 최대 크기 (기본: 8)",
    )
    p.add_argument(
        "--viz", action="store_true",
        help="PNG 시각화 생성 (기본: 비활성화)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="모델 로드·추론 없이 처리 예정 타임스텝 수만 출력",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    prepare_dir = pathlib.Path(args.prepare_dir)
    test_dir    = prepare_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = pathlib.Path(args.manifest) if args.manifest else prepare_dir / "dataset_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    test_clips: list[str] = manifest["clips_test"]
    if args.n_clips > 0:
        test_clips = test_clips[: args.n_clips]

    log.info("test 클립: %d개  |  step: %dms  |  출력: %s",
             len(test_clips), args.step_us // 1000, test_dir)

    # ── dry-run: 클립당 예상 타임스텝 수 집계 ─────────────────────
    if args.dry_run:
        avdi = _make_local_avdi(args.dataset_dir)
        from dataset.load_physical_aiavdataset import load_clip_features

        total_new = total_skip = 0
        for clip_id in test_clips:
            try:
                ctx = load_clip_features(clip_id, avdi=avdi, maybe_stream=True)
                ego = ctx["egomotion"]
                clip_start_us, ego_end_us = ego.time_range
                cam_end_us = min(
                    int(cam.timestamps[-1])
                    for cam in ctx["cameras"]
                    if cam.timestamps is not None and len(cam.timestamps) > 0
                )
                clip_end_us = min(ego_end_us, cam_end_us)
                t0_min = clip_start_us + _HIST_DUR_US
                t0_min = -(-t0_min // args.step_us) * args.step_us
                t0_max = clip_end_us   - _FUT_DUR_US
                if t0_min > t0_max:
                    continue
                t0_list = list(range(t0_min, t0_max + 1, args.step_us))
                new  = sum(1 for t in t0_list if not (test_dir / f"{clip_id}__{t}.h5").exists())
                skip = len(t0_list) - new
                log.info("  %s: 전체 %d개 / 신규 %d개 / 건너뜀 %d개",
                         clip_id[:8], len(t0_list), new, skip)
                total_new  += new
                total_skip += skip
            except Exception as exc:
                log.warning("  %s: 로드 실패 — %s", clip_id[:8], exc)

        log.info("dry-run 완료 — 신규 예정: %d개 / 건너뜀: %d개", total_new, total_skip)
        return

    # ── 모델 로드 ─────────────────────────────────────────────────
    torch.cuda.manual_seed_all(args.seed)
    log.info("모델 로딩: %s", args.model_dir)
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    model = Alpamayo1_5.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda")
    model.eval()
    log.info("모델 로드 완료")

    from dataset.collector import AlpamayoDatasetCollector
    collector = AlpamayoDatasetCollector(
        model=model,
        output_dir=test_dir,
        viz_dir=test_dir if args.viz else False,
    )

    avdi = _make_local_avdi(args.dataset_dir)

    # ── 병렬 파이프라인 ────────────────────────────────────────────
    work_q:   queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue(maxsize=args.prefetch)

    for clip_id in test_clips:
        work_q.put(clip_id)
    for _ in range(args.download_workers):
        work_q.put(_SENTINEL)

    workers = []
    for wid in range(1, args.download_workers + 1):
        t = threading.Thread(
            target=_download_worker,
            args=(wid, work_q, result_q, avdi,
                  test_dir, args.step_us, args.overwrite),
            daemon=True,
            name=f"download-{wid}",
        )
        t.start()
        workers.append(t)

    def _monitor() -> None:
        for w in workers:
            w.join()
        result_q.put(_SENTINEL)

    threading.Thread(target=_monitor, daemon=True, name="monitor").start()

    log.info("워커 %d개 시작 | prefetch=%d", args.download_workers, args.prefetch)

    # ── GPU 메인 루프 ─────────────────────────────────────────────
    n_saved = n_skip = n_err = n_clips_done = 0

    while True:
        item = result_q.get()
        if item is _SENTINEL:
            break

        clip_id = item["clip_id"]

        if item["type"] == "clip_error":
            log.error("클립 오류 (%s): %s", clip_id[:8], item["exc"])
            n_err += 1

        elif item["type"] == "frame":
            try:
                collector.collect_from_raw(item["raw"], clip_id, item["t0_us"])
                del item["raw"]
                n_saved += 1
            except Exception:
                log.exception("추론/저장 실패: %s t0=%d", clip_id[:8], item["t0_us"])
                n_err += 1

        elif item["type"] == "clip_done":
            n_skip        += item["n_skip"]
            n_clips_done  += 1
            log.info(
                "[%d/%d] %s 완료 — 신규 %d개 / 건너뜀 %d개 / 실패 %d개",
                n_clips_done, len(test_clips), clip_id[:8],
                item["n_new"], item["n_skip"], item["n_err"],
            )
            log.info("  누적 — 저장: %d | 건너뜀: %d | 실패: %d",
                     n_saved, n_skip, n_err)

    log.info("=" * 65)
    log.info("완료")
    log.info("  저장된 샘플  : %d", n_saved)
    log.info("  건너뜀       : %d  (이미 존재)", n_skip)
    log.info("  실패         : %d", n_err)
    log.info("  출력 경로    : %s", test_dir)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
