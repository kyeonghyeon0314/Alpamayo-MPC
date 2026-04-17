# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GT vs 예측 궤적 오차 기반 샘플 분류 스크립트 (STEP 1.5 — 필터링).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
목적
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HuggingFace에서 다운로드한 h5 샘플 각각에 대해
  ① Alpamayo-R1 모델이 예측한 미래 궤적 (output/pred_xyz)
  ② 실제 차량이 주행한 GT 궤적              (gt/future_xyz)
  ③ CoTend 숨은 상태                        (output/cotend_hidden_state)
을 비교하여 오차가 심한 샘플을 data/error/ 폴더로 격리한다.

이 필터링은 이후 label_mpc_weights.py (STEP 2)에서 MPC 가중치를
역최적화하기 전에 실행하며, "모델이 GT와 크게 벗어난 샘플은
cotend_hidden_state의 신뢰도가 낮으므로 학습 데이터에서 제외"
하는 것이 핵심 목적이다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
오차 지표 (4종)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
공통 조건:
  - 계산 구간: 앞 20스텝 (t0+0.1s ~ t0+2.0s, 10Hz)
      → label_mpc_weights.py의 MPC 호라이즌 N=20과 동일
  - 계산 축: XY 평면만 사용 (Z=높이 제외)
  - 좌표계: 자차 중심 로컬 프레임 (x=전방, y=좌측)

① ADE  (Average Displacement Error) — 기본 필터 기준
    displacement[t] = ‖pred_xy[t] − gt_xy[t]‖₂
    ADE = mean(displacement[0..19])                   [단위: m]

② FDE  (Final Displacement Error) — 2초 후 위치 오차
    FDE = ‖pred_xy[19] − gt_xy[19]‖₂                 [단위: m]
    해석: ADE가 낮아도 FDE가 크면 궤적 끝이 발산 → 불안정 샘플

③ Longitudinal ADE — 종방향(x축, 전방) 오차 평균
    lon_disp[t] = |pred_x[t] − gt_x[t]|
    Longitudinal ADE = mean(lon_disp[0..19])           [단위: m]
    해석: 속도/가감속 정확도. 앞뒤 방향 오차로, 너무 빠르거나
          느리게 예측한 케이스를 포착.

④ Lateral ADE — 횡방향(y축) 오차 평균
    lat_disp[t] = |pred_y[t] − gt_y[t]|
    Lateral ADE = mean(lat_disp[0..19])               [단위: m]
    해석: 차선 준수 정확도. ADE가 작아도 Lateral ADE가 크면
          종방향(x)은 맞지만 횡방향이 크게 틀린 케이스.

⑤ hs_norm — cotend_hidden_state L2 노름
    hs_norm = ‖cotend_hidden_state‖₂
    해석: hidden state 크기가 비정상적으로 크거나 작으면
          임베딩 이상(포화/붕괴) 가능성 → 학습 노이즈 유발.
          정상 범위는 데이터셋에 따라 다르므로 p1/p99 분포로 파악.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
파일 구조
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력:
    data/collected/{clip_id}__{t0_us}.h5      ← 수집된 샘플
    data/collected/viz/{clip_id}__{t0_us}.png ← 수집 시 생성된 BEV 시각화

출력 (ADE > threshold 인 샘플):
    data/error/{clip_id}__{t0_us}.h5          ← h5 이동 (복사 아님)
    data/error/viz/{clip_id}__{t0_us}.png     ← png 이동 (복사 아님)
    filter_error_report.csv                   ← 전체 샘플 지표 기록 (ADE 내림차순)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 모드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[모드 A] CPU 멀티스레드 (기본)
  • 파일 1개씩 병렬로 h5 읽기 + 지표 계산
  • 파일 수가 많아도 안정적 (메모리 상수)
  • NVMe SSD 기준 workers=16 권장

[모드 B] GPU 배치 (--use-gpu)
  • Phase 1 (CPU): 전체 h5 파일을 workers 스레드로 읽어 numpy 배열 누적
      - pred_xyz, gt_xyz: (N, 20, 2) float32
      - cotend_hidden_state: (N, 4096) float32
  • Phase 2 (GPU): 전체를 한 번에 CUDA 텐서로 올려 일괄 계산
      - RTX 3090 (24GB VRAM) 기준: ~100MB(XYZ) + ~1.1GB(hs) → 여유
  • h5 읽기/gzip 해제는 여전히 CPU — Phase 1이 I/O 병목
  • Phase 2 계산은 CPU 대비 100x 이상 빠르지만 오버헤드가 작아
    전체 시간 단축은 10~30% 수준 (파일 수가 많을수록 효과 증가)
  • torch 없으면 자동으로 CPU 모드로 폴백

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
권장 실행 순서
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) dry-run으로 지표 분포 먼저 확인 (파일 이동 없음):
       python filter_error_samples.py --dry-run

   출력 예:
       ADE 통계 (전체)
         평균  : 1.243 m
         중앙값: 0.891 m
         p95   : 4.821 m   ← 상위 5%의 기준선
         p99   : 8.203 m   ← 상위 1%의 기준선

2) 분포를 보고 threshold 결정 후 실제 실행:
       python filter_error_samples.py --threshold 5.0 --workers 16

   GPU 배치 활용:
       python filter_error_samples.py --threshold 5.0 --workers 16 --use-gpu

필터 조건 (OR — 하나라도 초과 시 error 폴더로 이동):
    --ade-thresh   ADE 임계값 [m]        (기본값: 2.0)
    --lon-thresh   Longitudinal ADE [m]  (기본값: 1.5)
    --lat-thresh   Lateral ADE [m]       (기본값: 1.0)

기타 인자:
    --workers    병렬 프로세스 수. NVMe SSD 기준 16 권장 (기본값: 16)
    --use-gpu    GPU 배치 모드 활성화 (torch 필요, 없으면 자동 폴백)
    --dry-run    파일을 실제로 이동하지 않고 통계 + 대상 목록만 출력
    --limit      처리할 최대 파일 수. dry-run 기본값: 10,000 / 실제: 전체

의존성: h5py, numpy, tqdm  [GPU 모드 추가: torch]
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 기본 경로 (이 스크립트 기준 상대 경로)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
COLLECTED_DIR = SCRIPT_DIR / "data" / "collected"
VIZ_DIR       = COLLECTED_DIR / "viz"
ERROR_DIR     = SCRIPT_DIR / "data" / "error"
ERROR_VIZ_DIR = ERROR_DIR / "viz"
REPORT_PATH   = SCRIPT_DIR / "filter_error_report.csv"

# label_mpc_weights.py의 MPC 호라이즌과 동일 (N=20, 2초, 10Hz)
_MPC_HORIZON = 20


# ─────────────────────────────────────────────────────────────────────────────
# 지표 계산 (CPU 모드 — 파일 단위)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(h5_path: Path) -> dict:
    """h5 파일에서 7종 지표를 계산한다.

    cotend_hidden_state가 없는 파일은 hs_norm=None으로 처리하고
    나머지 지표는 정상 계산한다.

    Returns:
        dict with keys: v0, v_mean, ade, fde, lon_ade, lat_ade, hs_norm
        v0     : 현재(t0) 속도 [m/s]  — 참고용
        v_mean : GT 미래 20스텝 평균 속도 [m/s]  — 저속 필터 기준
                 v0 단독 기준의 문제점을 해소:
                   저속→가속: v_mean > v0 → 호라이즌 전반이 유효하면 유지
                   고속→감속: v_mean > v_final → 호라이즌 전반이 유효하면 유지
                   저속 내내: v_mean 낮음 → 필터
        (hs_norm은 float | None)
    """
    with h5py.File(h5_path, "r") as f:
        gt_xyz    = f["gt/future_xyz"][:_MPC_HORIZON]           # (20, 3)
        pred_xyz  = f["output/pred_xyz"][:_MPC_HORIZON]         # (20, 3)
        hist      = f["input/ego_history_ego_states"][:]         # (T, state_dim)
        fut_v     = f["gt/future_ego_states"][:_MPC_HORIZON, 2] # (20,) 미래 속도
        hs_norm   = float(np.linalg.norm(f["output/cotend_hidden_state"][:])) \
                    if "output/cotend_hidden_state" in f else None

    v0     = float(hist[-1, 2])          # ⑥ 현재 속도 [m/s]  (선형화 기준점)
    v_mean = float(fut_v.mean())         # ⑦ 미래 20스텝 평균 속도 [m/s]

    diff_xy = pred_xyz[:, :2] - gt_xyz[:, :2]           # (20, 2)  x=종방향, y=횡방향
    disp    = np.linalg.norm(diff_xy, axis=1)            # (20,)

    ade     = float(disp.mean())                         # ① ADE
    fde     = float(disp[-1])                            # ② FDE (t=19)
    lon_ade = float(np.abs(diff_xy[:, 0]).mean())        # ③ 종방향(x축) 평균 오차
    lat_ade = float(np.abs(diff_xy[:, 1]).mean())        # ④ 횡방향(y축) 평균 오차
    # ⑤ hs_norm: cotend_hidden_state 없는 파일은 None

    return {"v0": v0, "v_mean": v_mean, "ade": ade, "fde": fde,
            "lon_ade": lon_ade, "lat_ade": lat_ade, "hs_norm": hs_norm}


def process_one(
    h5_path: Path,
    ade_thresh: float,
    lon_thresh: float,
    lat_thresh: float,
    min_speed: float,
    dry_run: bool,
) -> dict:
    """단일 샘플 처리: 지표 계산 → 임계값 초과 시 error 폴더로 이동.

    필터 조건 (OR):
        ADE > ade_thresh  OR  lon_ade > lon_thresh  OR  lat_ade > lat_thresh
        OR  v0 < min_speed  (다이나믹 바이시클 모델 유효 범위 미만)

    Returns:
        dict with keys: stem, v0, ade, fde, lon_ade, lat_ade, hs_norm, moved, error_msg
    """
    stem   = h5_path.stem
    result = {"stem": stem, "v0": None, "v_mean": None, "ade": None, "fde": None,
              "lon_ade": None, "lat_ade": None, "hs_norm": None,
              "moved": False, "error_msg": ""}

    try:
        m = compute_metrics(h5_path)
        result.update({k: round(v, 4) if v is not None else None
                       for k, v in m.items()})

        is_error = (
            m["ade"]    > ade_thresh or
            m["lon_ade"] > lon_thresh or
            m["lat_ade"] > lat_thresh or
            m["v_mean"]  < min_speed    # 호라이즌 평균 속도 < 기준: 다이나믹 모델 유효 범위 밖
        )
        if is_error:
            if not dry_run:
                dst_h5 = ERROR_DIR / h5_path.name
                shutil.move(str(h5_path), str(dst_h5))

                src_png = VIZ_DIR / f"{stem}.png"
                if src_png.exists():
                    shutil.move(str(src_png), str(ERROR_VIZ_DIR / src_png.name))

            result["moved"] = True

    except Exception as exc:
        result["error_msg"] = str(exc)
        log.warning("처리 실패: %s — %s", h5_path.name, exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GPU 배치 모드
# ─────────────────────────────────────────────────────────────────────────────

def _load_one_for_batch(h5_path: Path) -> dict | None:
    """h5에서 배열만 읽어 반환 (GPU 배치 Phase 1용).

    cotend_hidden_state가 없는 파일은 hs=None으로 반환 (정상 처리 계속).
    """
    try:
        with h5py.File(h5_path, "r") as f:
            gt_xyz   = f["gt/future_xyz"][:_MPC_HORIZON, :2].astype(np.float32)
            pred_xyz = f["output/pred_xyz"][:_MPC_HORIZON, :2].astype(np.float32)
            hist     = f["input/ego_history_ego_states"][:]
            fut_v    = f["gt/future_ego_states"][:_MPC_HORIZON, 2]
            hs = f["output/cotend_hidden_state"][:].astype(np.float32) \
                 if "output/cotend_hidden_state" in f else None
        v0     = float(hist[-1, 2])
        v_mean = float(fut_v.mean())
        return {"stem": h5_path.stem, "path": h5_path,
                "pred": pred_xyz, "gt": gt_xyz, "hs": hs,
                "v0": v0, "v_mean": v_mean}
    except Exception as exc:
        log.warning("읽기 실패: %s — %s", h5_path.name, exc)
        return None


def run_gpu_batch(
    h5_files: list[Path],
    ade_thresh: float,
    lon_thresh: float,
    lat_thresh: float,
    min_speed: float,
    dry_run: bool,
    workers: int,
) -> list[dict]:
    """전체 h5를 GPU에 올려 일괄 계산하는 배치 모드.

    Phase 1: CPU workers 스레드로 h5 읽기 (I/O 병목)
    Phase 2: GPU CUDA 텐서로 4종 지표 일괄 계산
    """
    import torch  # GPU 모드에서만 import

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("GPU 배치 모드: device=%s", device)

    # ── Phase 1: CPU 병렬 h5 읽기 ──────────────────────────────────────────
    # ProcessPoolExecutor: GIL 우회 → gzip 해제를 실제 멀티코어로 병렬 처리
    loaded: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one_for_batch, p): p for p in h5_files}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        unit="샘플", desc="Phase 1 (h5 읽기)"):
            r = fut.result()
            if r is not None:
                loaded.append(r)

    if not loaded:
        log.error("읽기 성공한 파일이 없습니다.")
        return []

    # ── Phase 2: GPU 일괄 계산 ─────────────────────────────────────────────
    # ADE/FDE/Lateral ADE: 전체 배치 (hs 유무와 무관)
    pred_np = np.stack([d["pred"] for d in loaded])  # (N, 20, 2)
    gt_np   = np.stack([d["gt"]   for d in loaded])  # (N, 20, 2)

    pred_t = torch.from_numpy(pred_np).to(device)
    gt_t   = torch.from_numpy(gt_np).to(device)

    diff  = pred_t - gt_t                                          # (N, 20, 2)
    disp  = diff.norm(dim=-1)                                      # (N, 20)
    ade_arr     = disp.mean(dim=-1).cpu().numpy()                  # (N,)
    fde_arr     = disp[:, -1].cpu().numpy()                        # (N,)
    lon_ade_arr = diff[:, :, 0].abs().mean(dim=-1).cpu().numpy()   # (N,) 종방향(x)
    lat_ade_arr = diff[:, :, 1].abs().mean(dim=-1).cpu().numpy()   # (N,) 횡방향(y)

    # hs_norm: cotend_hidden_state가 있는 파일만 GPU 계산, 없으면 None
    hs_has    = [i for i, d in enumerate(loaded) if d["hs"] is not None]
    hs_norm_arr: list[float | None] = [None] * len(loaded)
    if hs_has:
        hs_np  = np.stack([loaded[i]["hs"] for i in hs_has])  # (M, 4096)
        hs_norms = torch.from_numpy(hs_np).to(device).norm(dim=-1).cpu().numpy()
        for pos, idx in enumerate(hs_has):
            hs_norm_arr[idx] = float(hs_norms[pos])

    no_hs_count = len(loaded) - len(hs_has)
    if no_hs_count:
        log.info("cotend_hidden_state 없는 파일: %d개 (hs_norm=None 처리)", no_hs_count)

    log.info("Phase 2 (GPU 계산) 완료: %d samples", len(loaded))

    # ── 결과 조립 + 파일 이동 ─────────────────────────────────────────────
    results = []
    for i, d in enumerate(loaded):
        stem  = d["stem"]
        path  = d["path"]
        ade     = float(ade_arr[i])
        lon_ade = float(lon_ade_arr[i])
        lat_ade = float(lat_ade_arr[i])
        moved   = False

        v0     = d["v0"]
        v_mean = d["v_mean"]
        is_error = (
            ade     > ade_thresh or
            lon_ade > lon_thresh or
            lat_ade > lat_thresh or
            v_mean  < min_speed     # 호라이즌 평균 속도 < 기준: 다이나믹 모델 유효 범위 밖
        )
        if is_error and not dry_run:
            try:
                shutil.move(str(path), str(ERROR_DIR / path.name))
                src_png = VIZ_DIR / f"{stem}.png"
                if src_png.exists():
                    shutil.move(str(src_png), str(ERROR_VIZ_DIR / src_png.name))
                moved = True
            except Exception as exc:
                log.warning("이동 실패: %s — %s", path.name, exc)
        elif is_error:
            moved = True  # dry_run: 이동 대상으로만 표시

        hs_val = hs_norm_arr[i]
        results.append({
            "stem":      stem,
            "v0":        round(v0, 3),
            "v_mean":    round(v_mean, 3),
            "ade":       round(ade, 4),
            "fde":       round(float(fde_arr[i]), 4),
            "lon_ade":   round(lon_ade, 4),
            "lat_ade":   round(lat_ade, 4),
            "hs_norm":   round(hs_val, 4) if hs_val is not None else None,
            "moved":     moved,
            "error_msg": "",
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 통계 출력 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _print_stat(label: str, values: list[float], unit: str = "") -> None:
    arr = np.array(values)
    suffix = f" {unit}" if unit else ""
    print(f"  {label:<22} 평균={arr.mean():.3f}{suffix}  "
          f"중앙={np.median(arr):.3f}{suffix}  "
          f"p95={np.percentile(arr, 95):.3f}{suffix}  "
          f"p99={np.percentile(arr, 99):.3f}{suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GT vs 예측 궤적 오차 기반 샘플 분류")

    # ── 필터 임계값 ────────────────────────────────────────────────────────
    # 세 조건은 OR로 결합 — 하나라도 초과하면 error 폴더로 이동.
    #
    # 67,608개 전체 기준 분포 (2026-03 측정):
    #   지표              평균     중앙    p95     p99
    #   ADE             0.186m  0.138m  0.499m  0.822m
    #   Longitudinal    0.171m  0.120m  0.495m  0.851m   ← 종방향(전후, x축)
    #   Lateral         0.052m  0.026m  0.193m  0.374m   ← 횡방향(좌우, y축)
    #
    # 임계값 설정 기준 (p99의 약 2~3배 → "명백한 예측 실패"만 걸러냄):
    #   --ade-thresh 2.0   : p99(0.822m)의 2.4배. 2D 합산 오차 전반 감지
    #   --lon-thresh 1.5   : p99(0.851m)의 1.8배. 속도/가감속 오류 감지
    #   --lat-thresh 1.0   : p99(0.374m)의 2.7배. 차선이탈/급회전 감지
    #
    # 기준을 낮추려면 (더 공격적 필터링):
    #   --ade-thresh 1.0 --lon-thresh 1.0 --lat-thresh 0.5
    #   → 전체의 ~1~2% 제거, 약 700~1400개
    #
    # 기준을 높이려면 (보수적, 극단 이상치만 제거):
    #   --ade-thresh 3.0 --lon-thresh 2.5 --lat-thresh 2.0
    #   → 전체의 ~0.01% 이하, 수 개~수십 개
    parser.add_argument(
        "--ade-thresh", type=float, default=2.0,
        help="ADE 임계값 [m]. 초과 시 error 이동 (기본값: 2.0m, p99≈0.82m)",
    )
    parser.add_argument(
        "--lon-thresh", type=float, default=1.5,
        help="Longitudinal ADE 임계값 [m]. 초과 시 error 이동 (기본값: 1.5m, p99≈0.85m)",
    )
    parser.add_argument(
        "--lat-thresh", type=float, default=1.0,
        help="Lateral ADE 임계값 [m]. 초과 시 error 이동 (기본값: 1.0m, p99≈0.37m)",
    )
    # 저속 필터: 다이나믹 바이시클 모델은 약 20~30 km/h (5~8 m/s) 이상에서만 유효.
    # 그 미만 속도에서 역최적화한 MPC 웨이트는 모델 오차를 억지로 보상한 값 → 레이블 오염.
    # 5.0 m/s (≈18 km/h) 미만 샘플을 제거하는 것이 기본값.
    # 데이터셋에서 저속 비율 확인 후 조정: --dry-run 으로 v0 분포 먼저 확인 권장.
    parser.add_argument(
        "--min-speed", type=float, default=5.0,
        help="최소 속도 임계값 [m/s]. 미만 시 error 이동 (기본값: 5.0 m/s ≈ 18 km/h)",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="병렬 처리 프로세스 수. NVMe SSD 기준 16 권장 (기본값: 16)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="GPU 배치 모드 활성화 (torch + CUDA 필요, 없으면 자동 CPU 폴백)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 이동 없이 대상 파일 목록만 출력",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처리할 최대 파일 수 (기본값: dry-run=10,000 / 실제 실행=전체)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    h5_files = sorted(COLLECTED_DIR.glob("*.h5"))
    if not h5_files:
        log.error("h5 파일 없음: %s", COLLECTED_DIR)
        return

    limit = args.limit if args.limit is not None else (10_000 if args.dry_run else 0)
    if limit > 0:
        h5_files = h5_files[:limit]
        log.info("파일 수 제한: 전체 중 %d개만 처리", limit)

    log.info(
        "총 %d개 샘플  |  ade>%.2f  lon>%.2f  lat>%.2f  v0<%.1f  |  workers=%d  |  dry_run=%s",
        len(h5_files),
        args.ade_thresh, args.lon_thresh, args.lat_thresh, args.min_speed,
        args.workers, args.dry_run,
    )

    if not args.dry_run:
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        ERROR_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # ── 처리 방식 선택 ────────────────────────────────────────────────────
    use_gpu = args.use_gpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                log.warning("CUDA 사용 불가 — CPU 모드로 폴백합니다.")
                use_gpu = False
        except ImportError:
            log.warning("torch 미설치 — CPU 모드로 폴백합니다.")
            use_gpu = False

    if use_gpu:
        results = run_gpu_batch(
            h5_files,
            args.ade_thresh, args.lon_thresh, args.lat_thresh, args.min_speed,
            args.dry_run, args.workers,
        )
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_one, p,
                    args.ade_thresh, args.lon_thresh, args.lat_thresh,
                    args.min_speed, args.dry_run,
                ): p
                for p in h5_files
            }
            for fut in tqdm(as_completed(futures), total=len(futures), unit="샘플"):
                results.append(fut.result())

    # ── 통계 ──────────────────────────────────────────────────────────────
    valid  = [r for r in results if r["ade"] is not None]
    moved  = [r for r in valid   if r["moved"]]
    failed = [r for r in results if r["error_msg"]]

    print("\n" + "═" * 70)
    print(f"  전체 샘플      : {len(results):>7,}")
    print(f"  정상 처리      : {len(valid):>7,}")
    print(f"  파싱 실패      : {len(failed):>7,}")
    print(f"  오류 이동 대상 : {len(moved):>7,}  "
          f"(ADE>{args.ade_thresh:.2f}m OR lon>{args.lon_thresh:.2f}m "
          f"OR lat>{args.lat_thresh:.2f}m OR v0<{args.min_speed:.1f}m/s)")

    if valid:
        print()
        _print_stat("v0  (초기 속도)",   [r["v0"]     for r in valid if r["v0"]     is not None], "m/s")
        _print_stat("v_mean (호라이즌)", [r["v_mean"] for r in valid if r["v_mean"] is not None], "m/s")
        _print_stat("ADE",              [r["ade"]     for r in valid if r["ade"]     is not None], "m")
        _print_stat("FDE",              [r["fde"]     for r in valid if r["fde"]     is not None], "m")
        _print_stat("Longitudinal ADE", [r["lon_ade"] for r in valid if r["lon_ade"] is not None], "m")
        _print_stat("Lateral ADE",      [r["lat_ade"] for r in valid if r["lat_ade"] is not None], "m")
        _print_stat("hs_norm",          [r["hs_norm"] for r in valid if r["hs_norm"] is not None])
    print("═" * 70)

    if args.dry_run:
        top = sorted(moved, key=lambda x: x["ade"] or 0, reverse=True)[:20]
        low_speed = [r for r in moved if r["v_mean"] is not None and r["v_mean"] < args.min_speed]
        print(f"\n[dry-run] 이동될 샘플 {len(moved)}개 (저속(v_mean) {len(low_speed)}개 포함, 상위 20개):")
        for r in top:
            hs     = f"{r['hs_norm']:.1f}" if r["hs_norm"] is not None else "N/A"
            v0_s   = f"{r['v0']:.1f}"      if r["v0"]     is not None else "N/A"
            vm_s   = f"{r['v_mean']:.1f}"  if r["v_mean"] is not None else "N/A"
            print(f"  v0={v0_s}  v_mean={vm_s}m/s  ADE={r['ade']:.3f}m  "
                  f"lon={r['lon_ade']:.3f}m  lat={r['lat_ade']:.3f}m  "
                  f"hs={hs}  {r['stem']}")
        if len(moved) > 20:
            print(f"  ... 외 {len(moved) - 20}개")

    # ── CSV 리포트 (이동 대상만, ADE 내림차순) ────────────────────────────
    error_rows = sorted(
        [r for r in results if r["moved"]],
        key=lambda x: x["ade"] if x["ade"] is not None else float("inf"),
        reverse=True,
    )
    fieldnames = ["stem", "v0", "v_mean", "ade", "fde", "lon_ade", "lat_ade", "hs_norm", "error_msg"]
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {k: r[k] for k in fieldnames} for r in error_rows
        )

    log.info("리포트 저장: %s", REPORT_PATH)
    if not args.dry_run:
        log.info("완료. error 폴더: %s", ERROR_DIR)


if __name__ == "__main__":
    main()
