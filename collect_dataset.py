#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 데이터셋 수집 CLI.

전체 가이드: /workspace/alpamayo_dataset/DATASET_GUIDE.md

─────────────────────────────────────────────────────────────────────────
빠른 시작 (Docker 컨테이너 내부)
─────────────────────────────────────────────────────────────────────────

1) 테스트용 단일 샘플 수집 (모델 첫 실행 시 ~22 GB 자동 다운로드):

    python /workspace/collect_dataset.py \\
        --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 \\
        --t0-us 5100000 \\
        --output-dir /workspace/alpamayo_dataset/data/collected

2) CSV로 여러 샘플 일괄 수집:

    # clips.csv 형식 (헤더 포함):
    # clip_id,t0_us
    # 030c760c-ae38-49aa-9ad8-f5650a545d26,5100000
    # 030c760c-ae38-49aa-9ad8-f5650a545d26,7200000

    python /workspace/collect_dataset.py \\
        --csv /workspace/clips.csv \\
        --output-dir /workspace/alpamayo_dataset/data/collected

3) PNG 시각화를 별도 디렉토리에 저장:

    python /workspace/collect_dataset.py \\
        --csv /workspace/clips.csv \\
        --output-dir /workspace/alpamayo_dataset/data/collected \\
        --viz-dir /workspace/alpamayo_dataset/data/viz

4) 프리페치 크기 조정 (네트워크 유휴 최소화):

    # 클립당 t0_us 수 × 워커 수 × 2 정도로 설정 (기본 16 = 4개 × 2 워커 × 2)
    python /workspace/collect_dataset.py \\
        --csv /workspace/clips.csv \\
        --output-dir /workspace/alpamayo_dataset/data/collected \\
        --prefetch 16

5) 시각화 없이 수집만 (속도 우선):

    python /workspace/collect_dataset.py \\
        --csv /workspace/clips.csv \\
        --output-dir /workspace/alpamayo_dataset/data/collected \\
        --no-viz

6) 이미 수집된 파일 건너뛰고 재시작 (중단 후 이어하기):

    # --overwrite 없으면 기존 .h5가 있는 샘플은 자동 건너뜀
    python /workspace/collect_dataset.py \\
        --csv /workspace/clips.csv \\
        --output-dir /workspace/alpamayo_dataset/data/collected

─────────────────────────────────────────────────────────────────────────
수집되는 데이터 (샘플당 1개 .h5 파일)
─────────────────────────────────────────────────────────────────────────
  gt/                  실제 미래 궤적 + ego 상태 (x, y, speed, yaw, lon_accel)
  input/               ego 히스토리 1.6초 (카메라 이미지는 h5에 저장 안 함)
  output/              CoC 텍스트, 예측 궤적, cotend hidden state (4096-dim)
                         └ cotend: MPC 가중치 MLP의 입력 벡터

  ※ 카메라 이미지는 추론 직후 PNG 시각화({clip_id}__{t0_us}.png)로만 저장됩니다.
     디스크 절약: 이미지 미저장 시 파일당 약 1~2 MB (기존 수백 MB 대비).

─────────────────────────────────────────────────────────────────────────
주요 옵션
─────────────────────────────────────────────────────────────────────────
  --overwrite          기존 파일 덮어쓰기
  --viz-dir DIR        PNG 시각화 저장 경로 (기본: output-dir와 동일)
  --no-viz             PNG 시각화 생성 안 함
  --download-workers N 병렬 다운로드 스레드 수 (기본: 2)
                       네트워크 유휴 구간을 채워 GPU 대기 시간 단축
  --prefetch N         결과 큐 최대 크기 (기본: 8). 큐가 꽉 차면 워커 자동 대기
  --num-traj-samples N 궤적 샘플 수 (기본 1, cotend hs 추출은 1 권장)
  --max-gen-len N      CoC 최대 생성 토큰 수 (기본 256)
  --seed N             CUDA 랜덤 시드 (기본 42)
"""

import argparse
import csv
import queue
import threading
import logging
import sys

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpamayo 추론 결과 + cotend hidden state 데이터셋 수집"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip-id", metavar="ID", help="단일 클립 ID")
    src.add_argument("--csv", metavar="FILE", help="클립 목록 CSV (컬럼: clip_id, t0_us)")

    parser.add_argument(
        "--t0-us",
        type=int,
        metavar="US",
        help="기준 타임스탬프 (마이크로초). --clip-id 사용 시 필수.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help=".h5 파일 저장 디렉토리",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        metavar="DIR",
        help="PNG 시각화 저장 디렉토리 (기본: output-dir와 동일). --no-viz로 비활성화 가능",
    )
    parser.add_argument("--no-viz", action="store_true", help="PNG 시각화 생성 안 함")
    parser.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1,
        metavar="N",
        help="샘플링할 궤적 수 (기본 1, 늘릴수록 VRAM 사용 증가)",
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=256,
        metavar="N",
        help="CoC 최대 생성 토큰 수 (기본 256)",
    )
    parser.add_argument("--seed", type=int, default=42, help="CUDA 랜덤 시드 (기본 42)")
    parser.add_argument(
        "--prefetch",
        type=int,
        default=16,
        metavar="N",
        help="결과 큐 최대 크기 (기본 16). 큐가 꽉 차면 다운로드 워커가 자동 대기",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=2,
        metavar="N",
        help="병렬 다운로드 스레드 수 (기본 2). 네트워크 대역폭이 남을 때 늘리면 효과적",
    )
    parser.add_argument(
        "--model-dir",
        default="/workspace/alpamayo_dataset/data/alpamayo_1.5_weights",
        metavar="DIR",
        help="로컬 모델 가중치 디렉토리 (기본: /workspace/alpamayo_dataset/data/alpamayo_1.5_weights)",
    )
    parser.add_argument(
        "--dataset-dir",
        default="/workspace/alpamayo_dataset/data/nvidia_test_scenario",
        metavar="DIR",
        help="로컬 데이터셋 디렉토리 (기본: /workspace/alpamayo_dataset/data/nvidia_test_scenario)",
    )
    return parser.parse_args()


_SENTINEL = object()  # 워커 종료 신호


def _download_worker(
    worker_id: int,
    work_q: queue.Queue,
    result_q: queue.Queue,
    collector,
    avdi,
    output_dir: str,
    overwrite: bool,
) -> None:
    """병렬 다운로드 워커 스레드 (클립 단위 배치 다운로드).

    work_q에서 (clip_id, [t0_us_list])를 꺼내 클립 피처를 한 번만 로드한 뒤
    각 t0_us별 raw 데이터를 추출해 result_q에 적재.

    클립당 egomotion + 카메라 4개를 한 번만 스트리밍하므로,
    t0_us가 N개인 클립의 경우 스트리밍 횟수가 5N → 5로 감소합니다.
    result_q가 꽉 차면 GPU가 처리할 때까지 자동 대기(블로킹).
    """
    import pathlib

    out_dir = pathlib.Path(output_dir)
    log = __import__("logging").getLogger(__name__)

    while True:
        item = work_q.get()
        if item is _SENTINEL:
            break
        clip_id, t0_us_list = item

        # 이미 존재하는 파일은 즉시 skip 결과로 처리, 나머지만 다운로드
        pending = []
        for t0_us in t0_us_list:
            out_path = out_dir / f"{clip_id}__{t0_us}.h5"
            if out_path.exists() and not overwrite:
                result_q.put({"type": "skip", "clip_id": clip_id, "t0_us": t0_us})
            else:
                pending.append(t0_us)

        if not pending:
            continue

        # 클립 피처 한 번만 스트리밍
        try:
            log.info(
                "  [W%d 클립 다운로드] %s  (%d개 타임스텝: %s)",
                worker_id, clip_id, len(pending),
                ", ".join(f"{t / 1e6:.1f}s" for t in pending),
            )
            clip_ctx = collector.download_clip_context(clip_id, avdi=avdi)
        except Exception as exc:
            for t0_us in pending:
                result_q.put({"type": "error", "clip_id": clip_id, "t0_us": t0_us, "exc": exc})
            continue

        # 각 t0_us별 데이터 추출 (seek+decode만 수행, 추가 스트리밍 없음)
        for t0_us in pending:
            try:
                raw = collector.extract_raw_from_context(clip_ctx, t0_us)
                result_q.put({"type": "data", "clip_id": clip_id, "t0_us": t0_us, "raw": raw})
            except Exception as exc:
                result_q.put({"type": "error", "clip_id": clip_id, "t0_us": t0_us, "exc": exc})


def _load_clip_list(args: argparse.Namespace) -> list[tuple[str, int]]:
    """CLI 인자에서 (clip_id, t0_us) 쌍 목록을 구성."""
    if args.csv:
        clips = []
        with open(args.csv, newline="") as fh:
            for row in csv.DictReader(fh):
                clips.append((row["clip_id"].strip(), int(row["t0_us"])))
        return clips

    if args.t0_us is None:
        print("오류: --clip-id 사용 시 --t0-us 가 필요합니다.", file=sys.stderr)
        sys.exit(1)
    return [(args.clip_id, args.t0_us)]


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    clips = _load_clip_list(args)
    log.info("총 %d개 샘플 수집 예정 → %s", len(clips), args.output_dir)

    # 클립 단위로 묶기: {clip_id: [t0_us, ...]} (CSV 순서 유지)
    _clip_order: list[str] = []
    _clip_map: dict[str, list[int]] = {}
    for cid, t0 in clips:
        if cid not in _clip_map:
            _clip_map[cid] = []
            _clip_order.append(cid)
        _clip_map[cid].append(t0)
    clip_groups = [(cid, _clip_map[cid]) for cid in _clip_order]
    log.info("고유 클립 %d개 (클립당 평균 %.1f개 타임스텝) → 클립당 스트리밍 1회",
             len(clip_groups), len(clips) / max(len(clip_groups), 1))

    # 재현성을 위한 CUDA 시드 고정
    torch.cuda.manual_seed_all(args.seed)

    # 로컬 가중치에서 모델 로드
    log.info("Alpamayo1_5 모델 로딩: %s", args.model_dir)
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    model = Alpamayo1_5.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda")
    model.eval()

    # 로컬 데이터셋 디렉토리로 AVDI 초기화 (HF 인증 없이 로컬 파일 직접 사용)
    import contextlib
    import pathlib

    import pandas as pd
    import physical_ai_av

    class _LocalFirstAVDI(physical_ai_av.PhysicalAIAVDatasetInterface):
        """local_dir 파일을 HF 캐시보다 우선 사용 — HF 인증/접근 승인 없이 오프라인 동작."""

        def download_file(self, filename: str, **kwargs) -> str:
            """로컬 파일이 존재하면 HF 다운로드 없이 직접 경로 반환.

            sensor_presence.parquet → feature_presence.parquet 파일명 변경 대응.
            """
            # HF 데이터셋 업데이트로 파일명이 변경된 경우 리다이렉트
            if filename == "metadata/sensor_presence.parquet":
                filename = "metadata/feature_presence.parquet"
            p = pathlib.Path(self.local_dir) / filename
            if p.exists():
                return str(p)
            return super().download_file(filename, **kwargs)

        @contextlib.contextmanager
        def open_file(self, filename: str, mode: str = "rb", maybe_stream: bool = False):
            """로컬 파일이 존재하면 직접 열고, 없으면 HF 스트리밍으로 폴백."""
            p = pathlib.Path(self.local_dir) / filename
            if p.exists():
                with open(p, mode) as f:
                    yield f
            else:
                with super().open_file(filename, mode=mode, maybe_stream=maybe_stream) as f:
                    yield f

        def download_metadata(self) -> None:
            """local_dir/metadata/ 의 parquet 파일을 직접 로드."""
            meta_dir = pathlib.Path(self.local_dir) / "metadata"
            if meta_dir.exists():
                self.metadata = {
                    p.stem: pd.read_parquet(p) for p in sorted(meta_dir.glob("*.parquet"))
                }
            else:
                super().download_metadata()

    avdi = _LocalFirstAVDI(
        local_dir=args.dataset_dir,
        revision="main",
    )
    avdi.download_metadata()

    from dataset.collector import AlpamayoDatasetCollector

    # viz_dir: --no-viz이면 False(비활성화), --viz-dir 지정 시 해당 경로, 기본은 output_dir
    viz_dir = False if args.no_viz else (args.viz_dir or args.output_dir)

    collector = AlpamayoDatasetCollector(
        model=model,
        output_dir=args.output_dir,
        viz_dir=viz_dir,
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_gen_len,
    )

    # ── 병렬 다운로드 + GPU 추론 파이프라인 ──────────────────────────────────
    # work_q:   메인이 (clip_id, [t0_us_list]) 클립 그룹 적재 → 워커들이 꺼내 다운로드
    #           클립당 egomotion+카메라(5개 스트림)를 한 번만 로드 → N개 t0_us 추출
    # result_q: 워커들이 (clip_id, t0_us, raw) 단위로 적재 → GPU 메인이 꺼내 추론+저장
    #
    #  워커1: [─클립A 다운(1회)─→ t0₁,t0₂,t0₃ 추출─][─클립C 다운─→ ...]
    #  워커2:   [─클립B 다운(1회)─→ t0₁,t0₂ 추출──][─클립D 다운─→ ...]
    #  GPU:           [추론][추론][추론][추론][추론]...   ← result_q에서 소비
    n_workers = args.download_workers
    work_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue(maxsize=args.prefetch)

    # work_q에 클립 그룹 적재 후 워커 수만큼 종료 신호
    for clip_group in clip_groups:
        work_q.put(clip_group)
    for _ in range(n_workers):
        work_q.put(_SENTINEL)

    # 다운로드 워커 시작
    workers = []
    for wid in range(1, n_workers + 1):
        t = threading.Thread(
            target=_download_worker,
            args=(wid, work_q, result_q, collector, avdi, args.output_dir, args.overwrite),
            daemon=True,
            name=f"download-{wid}",
        )
        t.start()
        workers.append(t)
    log.info("다운로드 워커 %d개 시작  (결과 큐 크기: %d)", n_workers, args.prefetch)

    # 모든 워커 완료 시 result_q에 sentinel을 보내는 모니터 스레드
    def _monitor():
        for w in workers:
            w.join()
        result_q.put(_SENTINEL)

    threading.Thread(target=_monitor, daemon=True, name="monitor").start()

    # GPU 메인 루프: result_q에서 꺼내 추론+저장
    success, failed, skipped = 0, 0, 0
    n_total = len(clips)
    i = 0
    while True:
        item = result_q.get()
        if item is _SENTINEL:
            break
        i += 1
        clip_id, t0_us = item["clip_id"], item["t0_us"]
        log.info("─" * 60)
        log.info("[%d/%d] %s  t0=%.1fs", i, n_total, clip_id, t0_us / 1e6)

        if item["type"] == "skip":
            log.info("  건너뜀 (이미 존재)")
            skipped += 1
        elif item["type"] == "error":
            log.error("  다운로드 실패: %s", item["exc"])
            failed += 1
        else:
            try:
                collector.collect_from_raw(item["raw"], clip_id, t0_us)
                success += 1
            except Exception:
                log.exception("  추론/저장 실패")
                failed += 1

        log.info("진행: 저장 %d개 | 건너뜀 %d개 | 실패 %d개 / 전체 %d개",
                 success, skipped, failed, n_total)

    log.info("=" * 60)
    log.info("완료 — 저장: %d개, 건너뜀: %d개, 실패: %d개  (전체 %d개)",
             success, skipped, failed, n_total)


if __name__ == "__main__":
    main()
