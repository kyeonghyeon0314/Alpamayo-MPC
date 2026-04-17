#!/usr/bin/env python3
"""
R1 모델로 클립 전체를 100ms 간격으로 추론하여 연속 PCL 데이터셋 생성.

기존 collected/ 데이터는 클립당 4개 타임스텝만 존재 → 연속 PCL 시뮬레이션 불가.
이 스크립트는 split_manifest.json의 test 클립을 대상으로,
egomotion 시간 범위 전체를 100ms 간격으로 R1 추론하여 클립 단위 h5 파일로 저장한다.

출력 스키마 ({clip_id}.h5):
  attrs:
    clip_id               str     — 클립 UUID
    n_frames              int     — 총 프레임 수
    step_us               int     — 샘플링 간격 [μs] (기본 100000 = 100ms)
    t_start_s             float   — 클립 유효 구간 시작 [s]
    t_end_s               float   — 클립 유효 구간 끝 [s]
    visual_disconnect_note str    — 오프라인 평가 한계 설명

  추론 데이터 (매 step_us마다):
    times_us          (N,)        int64   — 타임스탬프 [μs]
    times_s           (N,)        float64 — 타임스탬프 [s]
    cotend            (N, 4096)   float32 — VLM cotend hidden state
    gt_xyz_local      (N, 64, 3)  float32 — GT 미래 궤적 (로컬 프레임, x=전방)
    gt_yaw_local      (N, 64)     float32 — GT 미래 yaw (로컬 프레임)
    global_origin_xy  (N, 2)      float64 — 전역 좌표 원점 [m] (전체 주행 궤적, 첫 프레임 기준)
    global_yaw        (N,)        float64 — 전역 헤딩 [rad] (첫 프레임 기준)

  raw egomotion (t0 시점, 전역 좌표계, 다운로드 원본 그대로):
    ego_xyz_global    (N, 3)      float64 — 전역 위치 [x, y, z]
    ego_quat_global   (N, 4)      float64 — 전역 자세 [qx, qy, qz, qw]
    ego_vel           (N, 3)      float64 — 속도 [vx, vy, vz]
    ego_acc           (N, 3)      float64 — 가속도 [ax, ay, az]
    ego_curv          (N, 1)      float64 — 곡률

  장면 컨텍스트 (시각화·검토용):
    front_frames      (N, H, W, 3) uint8  — 전방 카메라 (t0 시점, 다운스케일)

실행 예시 (Docker 컨테이너 내부):
  python /workspace/closed-loop-simul/build_pcl_dataset.py \\
      --manifest  /workspace/alpamayo_dataset/data/split/split_manifest.json \\
      --out-dir   /workspace/alpamayo_dataset/pcl_clips \\
      --model-dir /workspace/alpamayo_dataset/data/alpamayo_weights \\
      --dataset-dir /workspace/alpamayo_dataset/data/nvidia_test_scenario \\
      --n-clips   10

오프라인 평가 한계 (Visual Disconnect):
  hidden state는 GT 경로를 따른 카메라 뷰에서 추출됨.
  PCL 시뮬레이션 중 차량이 GT에서 이탈해도 이탈 위치의 hidden state는
  재생성 불가 → 기존 값을 그대로 사용. 논문/보고서에 명시 필요.
"""

import argparse
import contextlib
import json
import logging
import pathlib
import queue
import sys
import threading

import h5py
import numpy as np
import torch

# ══════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════

_STEP_US    = 100_000   # 샘플링 간격: 100ms = 10Hz
_HIST_STEPS = 16        # 히스토리 스텝 수 (1.6초)
_FUT_STEPS  = 64        # 미래 예측 스텝 수 (6.4초)
_HIST_DUR_US = _HIST_STEPS * _STEP_US   # 1,600,000 μs
_FUT_DUR_US  = _FUT_STEPS  * _STEP_US   # 6,400,000 μs
_DT_S       = 0.1       # 물리 시뮬레이션 타임스텝 [s]

# 장면 컨텍스트 영상 해상도 (원본의 1/4 다운스케일)
_FRAME_HEIGHT = 240     # 전방 카메라 저장 높이 [px]

# 카메라 정렬 후 인덱스 (0=cross_left, 1=front_wide, 2=cross_right, 3=front_tele)
_FRONT_CAM_IDX = 1

_VISUAL_DISCONNECT_NOTE = (
    "hidden_state는 GT 경로를 따른 카메라 뷰에서 추출됨. "
    "PCL 시뮬레이션 중 차량이 GT에서 이탈해도 이탈 위치의 hidden_state는 "
    "재생성 불가 — 오프라인 평가의 고유 한계."
)

_SENTINEL = object()    # 워커 종료 신호


# ══════════════════════════════════════════════════════
# 로컬 데이터셋 인터페이스 (_LocalFirstAVDI)
# collect_dataset.py의 패턴을 그대로 따름
# ══════════════════════════════════════════════════════

def _make_local_avdi(dataset_dir: str):
    """로컬 데이터셋 디렉토리를 HuggingFace보다 우선 사용하는 AVDI 생성."""
    import pandas as pd
    import physical_ai_av

    class _LocalFirstAVDI(physical_ai_av.PhysicalAIAVDatasetInterface):
        """local_dir 파일을 HF 캐시보다 우선 사용 — HF 인증 없이 오프라인 동작."""

        def download_file(self, filename: str, **kwargs) -> str:
            # HF 업데이트로 파일명이 변경된 경우 리다이렉트
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
                    p.stem: pd.read_parquet(p)
                    for p in sorted(meta_dir.glob("*.parquet"))
                }
            else:
                super().download_metadata()

    avdi = _LocalFirstAVDI(local_dir=dataset_dir, revision="main")
    avdi.download_metadata()
    return avdi


# ══════════════════════════════════════════════════════
# 좌표 변환 유틸리티
# ══════════════════════════════════════════════════════

def _build_global_poses(
    times_s: np.ndarray,
    gt_xyz_list: list[np.ndarray],
    gt_yaw_list: list[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    """각 프레임의 전역 좌표 (origin_xy, heading_rad) 계산.

    프레임 0을 전역 원점(0,0)·yaw=0으로 정의.
    이후 각 프레임 k에 대해 이전 프레임 j의 GT가 k를 커버하면
    j의 전역 포즈를 통해 k의 전역 포즈를 계산.
    """
    poses: list[tuple[np.ndarray, float]] = [(np.zeros(2), 0.0)]

    for k in range(1, len(times_s)):
        found = False
        for j in range(k - 1, -1, -1):
            dt  = times_s[k] - times_s[j]
            idx = round(dt / _DT_S) - 1
            gt  = gt_xyz_list[j]
            if 0 <= idx < len(gt):
                x_loc   = float(gt[idx, 0])
                y_loc   = float(gt[idx, 1])
                yaw_loc = float(gt_yaw_list[j][idx])

                origin_j, yaw_j = poses[j]
                c, s       = np.cos(yaw_j), np.sin(yaw_j)
                p_global   = np.array([[c, -s], [s, c]]) @ np.array([x_loc, y_loc]) + origin_j
                yaw_global = yaw_j + yaw_loc
                poses.append((p_global, yaw_global))
                found = True
                break
        if not found:
            # GT 커버리지 갭 → 이전 포즈 그대로 유지
            poses.append(poses[-1])

    return poses


# ══════════════════════════════════════════════════════
# 단일 프레임 처리 (추론 + 데이터 추출)
# ══════════════════════════════════════════════════════

def _resize_frame(frame_chw: np.ndarray, target_h: int = _FRAME_HEIGHT) -> np.ndarray:
    """(3, H, W) uint8 → (H_new, W_new, 3) uint8, 비율 유지."""
    from PIL import Image

    h, w = frame_chw.shape[1], frame_chw.shape[2]
    new_w = int(round(w * target_h / h))
    img = Image.fromarray(frame_chw.transpose(1, 2, 0))
    img = img.resize((new_w, target_h), Image.BILINEAR)
    return np.array(img)


def _infer_one(model, processor, raw: dict, device: torch.device) -> dict | None:
    """raw 데이터 하나에 대해 R1 추론 후 필요한 데이터만 추출.

    Returns:
        {cotend, gt_xyz_local, gt_yaw_local, ego_xyz_global, ego_quat_global,
         ego_vel, ego_acc, ego_curv, front_frame}
        추론 실패 시 None 반환.
    """
    from alpamayo1_5 import helper

    try:
        # ── 모델 입력 구성 ────────────────────────────────────
        # flatten: (N_cam, N_frm, C, H, W) → (N_cam*N_frm, C, H, W)
        messages  = helper.create_message(
            raw["image_frames"].flatten(0, 1),
            camera_indices=raw["camera_indices"],
        )
        tokenized = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": tokenized,
                "ego_history_xyz": raw["ego_history_xyz"],
                "ego_history_rot": raw["ego_history_rot"],
            },
            device=device,
        )

        # ── R1 추론 (cotend hidden state 캡처 활성화) ─────────
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                num_traj_samples=1,
                max_generation_length=256,
                top_p=0.98,
                temperature=0.6,
                return_extra=True,
                return_cotend_hidden_state=True,
            )

        # ── cotend hidden state 추출 ──────────────────────────
        raw_hs = extra.get("cotend_hidden_state")
        if raw_hs is None:
            logging.getLogger(__name__).warning("cotend hidden state 캡처 실패 — 0으로 채움")
            cotend = np.zeros(4096, dtype=np.float32)
        else:
            cotend = raw_hs[0].float().cpu().numpy()    # (4096,)

        # ── GT 미래 궤적 (로컬 프레임) ───────────────────────
        gt_xyz_local = raw["ego_future_xyz"].cpu().numpy()[0, 0]   # (64, 3)
        gt_rot_local = raw["ego_future_rot"].cpu().numpy()[0, 0]   # (64, 3, 3)
        # yaw: R[1,0] / R[0,0] 으로 지면 투영 방위각 추출
        gt_yaw_local = np.arctan2(gt_rot_local[:, 1, 0], gt_rot_local[:, 0, 0])  # (64,)

        # ── t0 시점 raw egomotion (히스토리 마지막 스텝 = t0) ─
        ego_xyz_global  = raw["ego_history_xyz_global"].numpy()[-1]   # (3,)  float64
        ego_quat_global = raw["ego_history_quat_global"].numpy()[-1]  # (4,)  float64
        ego_vel         = raw["ego_history_vel"].numpy()[-1]          # (3,)  float64
        ego_acc         = raw["ego_history_acc"].numpy()[-1]          # (3,)  float64
        ego_curv        = raw["ego_history_curv"].numpy()[-1]         # (1,)  float64

        # ── 전방 카메라 프레임 (t0 시점, 다운스케일) ─────────
        # image_frames: (N_cam, N_frm, C, H, W), 카메라는 인덱스 순 정렬됨
        # _FRONT_CAM_IDX=1 → front_wide_120fov, 마지막 프레임(인덱스 -1) = t0 시점
        front_tensor = raw["image_frames"][_FRONT_CAM_IDX, -1]    # (C, H, W)
        front_np     = front_tensor.numpy()                        # uint8 (C, H, W)
        front_frame  = _resize_frame(front_np, _FRAME_HEIGHT)      # (H_s, W_s, 3)

        return {
            "cotend":          cotend.astype(np.float32),
            "gt_xyz_local":    gt_xyz_local.astype(np.float32),
            "gt_yaw_local":    gt_yaw_local.astype(np.float32),
            "ego_xyz_global":  ego_xyz_global,
            "ego_quat_global": ego_quat_global,
            "ego_vel":         ego_vel,
            "ego_acc":         ego_acc,
            "ego_curv":        ego_curv,
            "front_frame":     front_frame,           # uint8 (H_s, W_s, 3)
        }

    except Exception as exc:
        logging.getLogger(__name__).warning("추론 실패: %s", exc)
        return None


# ══════════════════════════════════════════════════════
# 클립 h5 저장
# ══════════════════════════════════════════════════════

def _save_clip_h5(
    clip_id: str,
    frames: list[tuple[int, dict]],
    out_dir: pathlib.Path,
    step_us: int,
) -> pathlib.Path:
    """누적된 프레임 결과를 단일 클립 h5 파일로 저장."""
    # 시간순 정렬 (워커 경쟁 조건 방지)
    frames.sort(key=lambda x: x[0])

    times_us = np.array([t for t, _ in frames], dtype=np.int64)
    times_s  = (times_us / 1e6).astype(np.float64)

    gt_xyz_list  = [d["gt_xyz_local"] for _, d in frames]
    gt_yaw_list  = [d["gt_yaw_local"] for _, d in frames]

    # 전역 좌표 계산 (첫 프레임 기준)
    poses = _build_global_poses(times_s, gt_xyz_list, gt_yaw_list)

    out_path = out_dir / f"{clip_id}.h5"
    with h5py.File(out_path, "w") as f:
        # ── 메타데이터 ───────────────────────────────────────
        f.attrs["clip_id"]               = clip_id
        f.attrs["n_frames"]              = len(frames)
        f.attrs["step_us"]               = step_us
        f.attrs["t_start_s"]             = float(times_s[0])
        f.attrs["t_end_s"]               = float(times_s[-1])
        f.attrs["visual_disconnect_note"] = _VISUAL_DISCONNECT_NOTE

        # ── 시간축 ───────────────────────────────────────────
        f.create_dataset("times_us", data=times_us)
        f.create_dataset("times_s",  data=times_s)

        # ── 추론 데이터 ──────────────────────────────────────
        f.create_dataset("cotend", data=np.stack([d["cotend"] for _, d in frames]), compression="gzip")

        # ── raw egomotion (전역 좌표계, t0 시점) ─────────────
        f.create_dataset("ego_xyz_global",  data=np.stack([d["ego_xyz_global"]  for _, d in frames]), compression="gzip")
        f.create_dataset("ego_quat_global", data=np.stack([d["ego_quat_global"] for _, d in frames]), compression="gzip")
        f.create_dataset("ego_vel",         data=np.stack([d["ego_vel"]         for _, d in frames]), compression="gzip")
        f.create_dataset("ego_acc",         data=np.stack([d["ego_acc"]         for _, d in frames]), compression="gzip")
        f.create_dataset("ego_curv",        data=np.stack([d["ego_curv"]        for _, d in frames]), compression="gzip")

        # ── GT 궤적 (로컬 프레임) ────────────────────────────
        f.create_dataset("gt_xyz_local", data=np.stack(gt_xyz_list), compression="gzip")
        f.create_dataset("gt_yaw_local", data=np.stack(gt_yaw_list), compression="gzip")

        # ── 전역 좌표 (전체 주행 궤적) ──────────────────────
        f.create_dataset(
            "global_origin_xy",
            data=np.array([p[0] for p in poses], dtype=np.float64),
            compression="gzip",
        )
        f.create_dataset(
            "global_yaw",
            data=np.array([p[1] for p in poses], dtype=np.float64),
            compression="gzip",
        )

        # ── 장면 컨텍스트: 전방 카메라 영상 ─────────────────
        f.create_dataset(
            "front_frames",
            data=np.stack([d["front_frame"] for _, d in frames]),  # (N, H, W, 3) uint8
            compression="gzip",
        )
        f["front_frames"].attrs["camera"] = "camera_front_wide_120fov"
        f["front_frames"].attrs["note"]   = "t0 시점 단일 프레임 (4프레임 스택 아님)"

    return out_path


# ══════════════════════════════════════════════════════
# 클립 시각화 PNG 저장
# ══════════════════════════════════════════════════════

# 시각화용 샘플 프레임 수 (등간격 추출)
_N_SAMPLE_FRAMES = 8


def _save_clip_viz(
    clip_id: str,
    frames: list[tuple[int, dict]],
    out_path: pathlib.Path,
    step_us: int,
) -> None:
    """클립 데이터 개요 PNG 저장.

    레이아웃:
      상단: 전방 카메라 N장 샘플 (등간격, 시간 그라디언트 테두리)
      하단 좌(넓음): BEV 전체 주행 궤적 + 샘플별 GT 미래 궤적(점선)
      하단 우(좁음): 속도 프로파일 + 클립 요약 정보
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not frames:
        return

    # ── 데이터 준비 ───────────────────────────────────────────
    frames_s = sorted(frames, key=lambda x: x[0])   # 시간순 정렬
    N = len(frames_s)

    times_s  = np.array([t / 1e6 for t, _ in frames_s])
    # 속도 크기: ego_vel [vx, vy, vz] (전역 좌표계) 의 norm
    v0_arr   = np.array([np.linalg.norm(d["ego_vel"]) for _, d in frames_s], dtype=np.float32)  # [m/s]
    gt_xyz_list = [d["gt_xyz_local"] for _, d in frames_s]
    gt_yaw_list = [d["gt_yaw_local"] for _, d in frames_s]

    # 전역 포즈 계산 (h5와 동일 로직, 메모리에서 재계산)
    poses    = _build_global_poses(times_s, gt_xyz_list, gt_yaw_list)
    global_xy = np.array([p[0] for p in poses])     # (N, 2)
    dur_s    = float(times_s[-1] - times_s[0])
    t_norm   = (times_s - times_s[0]) / max(dur_s, 1e-6)  # 0→1 정규화

    # 샘플 인덱스 (등간격)
    n_sample = min(_N_SAMPLE_FRAMES, N)
    sample_idx = np.linspace(0, N - 1, n_sample, dtype=int)

    # ── 그림 레이아웃 ─────────────────────────────────────────
    fig = plt.figure(figsize=(max(18, n_sample * 3), 11))
    fig.patch.set_facecolor("#1a1a2e")   # 어두운 배경

    fig.suptitle(
        f"clip: {clip_id}  |  {N} frames  |  {dur_s:.1f}s  |  step={step_us // 1000}ms",
        fontsize=10, color="white", y=0.995, fontfamily="monospace",
    )

    # 전체: 2행 × 2열 (하단 오른쪽은 좁게)
    gs_root = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[3, 5],
        width_ratios=[3, 1],
        hspace=0.28, wspace=0.18,
        left=0.04, right=0.97, top=0.97, bottom=0.04,
    )

    # 상단 행: 카메라 프레임 (전체 너비)
    gs_cam = gridspec.GridSpecFromSubplotSpec(
        1, n_sample, subplot_spec=gs_root[0, :], wspace=0.03,
    )

    # ── 상단: 전방 카메라 샘플 ───────────────────────────────
    cmap_t = plt.cm.plasma   # 시간 그라디언트 색상맵

    for col, idx in enumerate(sample_idx):
        ax = fig.add_subplot(gs_cam[0, col])
        ax.set_facecolor("#0d0d1a")

        frame = frames_s[idx][1]["front_frame"]  # (H, W, 3) uint8
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])

        # 타임스탬프 레이블
        t_pct = t_norm[idx]
        color = cmap_t(t_pct)
        ax.set_title(
            f"t = {times_s[idx]:.1f}s",
            fontsize=8, color=color, pad=3,
        )

        # 시간에 비례한 테두리 색상
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor(color)

    # ── 하단 좌: BEV 궤적 ────────────────────────────────────
    ax_bev = fig.add_subplot(gs_root[1, 0])
    ax_bev.set_facecolor("#0d0d1a")

    # 경로를 LineCollection으로 시간 그라디언트 색상 렌더링
    pts  = global_xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="plasma",
                          norm=plt.Normalize(times_s[0], times_s[-1]),
                          lw=2.5, alpha=0.9, zorder=3)
    lc.set_array(times_s[:-1])
    ax_bev.add_collection(lc)

    # 방향 화살표 (등간격)
    arrow_step = max(1, N // 12)
    for i in range(0, N - 1, arrow_step):
        j   = min(i + 1, N - 1)
        dx  = global_xy[j, 0] - global_xy[i, 0]
        dy  = global_xy[j, 1] - global_xy[i, 1]
        dist = np.hypot(dx, dy)
        if dist > 0.05:
            ax_bev.annotate(
                "",
                xy=(global_xy[i, 0] + dx * 0.6, global_xy[i, 1] + dy * 0.6),
                xytext=(global_xy[i, 0], global_xy[i, 1]),
                arrowprops=dict(
                    arrowstyle="->", color=cmap_t(t_norm[i]),
                    lw=1.2, mutation_scale=12,
                ),
                zorder=4,
            )

    # 샘플 타임스텝의 GT 미래 궤적 (점선, 반투명)
    for idx in sample_idx[::2]:
        gt   = gt_xyz_list[idx]       # (64, 3) 로컬 프레임
        ox, oy  = poses[idx][0]
        yaw_g   = poses[idx][1]
        c, s    = np.cos(yaw_g), np.sin(yaw_g)
        R       = np.array([[c, -s], [s, c]])
        gt_glob = (R @ gt[:, :2].T).T + np.array([ox, oy])
        ax_bev.plot(
            gt_glob[:, 0], gt_glob[:, 1],
            color=cmap_t(t_norm[idx]), alpha=0.3, lw=1.2,
            ls="--", zorder=2,
        )

    # 시작/끝 마커
    ax_bev.plot(*global_xy[0],  "^", color="#00ff88", ms=11, zorder=6,
                label="Start", mec="white", mew=0.5)
    ax_bev.plot(*global_xy[-1], "s", color="#ff4466", ms=10, zorder=6,
                label="End",   mec="white", mew=0.5)

    ax_bev.autoscale_view()
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("X [m]",  fontsize=9, color="white")
    ax_bev.set_ylabel("Y [m]",  fontsize=9, color="white")
    ax_bev.set_title("Full Trajectory (Global Frame)", fontsize=10, color="white", pad=6)
    ax_bev.tick_params(colors="gray")
    ax_bev.legend(fontsize=8, loc="best",
                  facecolor="#1a1a2e", labelcolor="white", edgecolor="gray")
    ax_bev.grid(True, alpha=0.15, color="gray")
    for spine in ax_bev.spines.values():
        spine.set_edgecolor("#444466")

    # 타임 컬러바
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(times_s[0], times_s[-1]))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_bev, shrink=0.75, pad=0.02)
    cb.set_label("Time [s]", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    # ── 하단 우: 속도 프로파일 + 요약 ───────────────────────
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_root[1, 1], hspace=0.45,
    )

    # 속도 그래프
    ax_v = fig.add_subplot(gs_right[0])
    ax_v.set_facecolor("#0d0d1a")
    v_kmh = v0_arr * 3.6

    ax_v.plot(times_s, v_kmh, lw=1.8, color="#00ccff", zorder=3)
    ax_v.fill_between(times_s, 0, v_kmh, alpha=0.2, color="#00ccff")

    # 샘플 타임스텝 마커
    ax_v.scatter(times_s[sample_idx], v_kmh[sample_idx],
                 c=[cmap_t(t_norm[i]) for i in sample_idx],
                 s=30, zorder=4, edgecolors="white", linewidths=0.5)

    ax_v.set_xlabel("Time [s]", fontsize=8, color="white")
    ax_v.set_ylabel("Speed [km/h]", fontsize=8, color="white")
    ax_v.set_title("Speed Profile", fontsize=9, color="white", pad=4)
    ax_v.tick_params(colors="gray", labelsize=7)
    ax_v.grid(True, alpha=0.15, color="gray")
    for spine in ax_v.spines.values():
        spine.set_edgecolor("#444466")

    # 요약 텍스트 박스
    ax_info = fig.add_subplot(gs_right[1])
    ax_info.set_facecolor("#0d0d1a")
    ax_info.axis("off")

    total_dist = float(np.sum(np.linalg.norm(np.diff(global_xy, axis=0), axis=1)))
    avg_speed  = float(np.mean(v_kmh))
    max_speed  = float(np.max(v_kmh))

    info_lines = [
        ("Clip ID",    clip_id[:8] + "..."),
        ("Frames",     f"{N}  ({step_us // 1000}ms step)"),
        ("Range",      f"{times_s[0]:.1f}s -> {times_s[-1]:.1f}s"),
        ("Distance",   f"{total_dist:.0f} m"),
        ("Avg Speed",  f"{avg_speed:.1f} km/h"),
        ("Max Speed",  f"{max_speed:.1f} km/h"),
    ]

    for row, (label, value) in enumerate(info_lines):
        ax_info.text(
            0.0, 1.0 - row * 0.17, f"{label}",
            fontsize=7.5, color="#aaaacc", va="top",
            transform=ax_info.transAxes,
        )
        ax_info.text(
            0.0, 1.0 - row * 0.17 - 0.085, f"  {value}",
            fontsize=8, color="white", va="top", fontfamily="monospace",
            transform=ax_info.transAxes,
        )

    plt.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ══════════════════════════════════════════════════════
# 다운로드 워커 스레드
# ══════════════════════════════════════════════════════

def _download_worker(
    worker_id: int,
    work_q: queue.Queue,
    result_q: queue.Queue,
    avdi,
    out_dir: pathlib.Path,
    step_us: int,
    overwrite: bool,
) -> None:
    """다운로드 워커: 클립 피처를 한 번만 로드하고 각 t0_us별 raw 데이터를 result_q에 적재.

    GPU 추론이 이전 프레임을 처리하는 동안 다음 프레임의 seek+decode를 비동기로 수행.
    result_q.maxsize로 메모리 상한을 제한함 (큐가 꽉 차면 자동 대기).
    """
    from dataset.load_physical_aiavdataset import (
        extract_sample_from_clip_features,
        load_clip_features,
    )

    log = logging.getLogger(__name__)

    while True:
        item = work_q.get()
        if item is _SENTINEL:
            break

        clip_id = item

        # 이미 존재하면 건너뜀
        out_path = out_dir / f"{clip_id}.h5"
        if out_path.exists() and not overwrite:
            log.info("[W%d] 건너뜀 (이미 존재): %s", worker_id, clip_id[:8])
            result_q.put({"type": "skip", "clip_id": clip_id})
            continue

        # ── 클립 피처 로드 (egomotion + 카메라 4개, 1회) ────
        # maybe_stream=True: 로컬에 데이터가 없을 때 HF에서 스트리밍
        log.info("[W%d] 클립 로드 중: %s", worker_id, clip_id[:8])
        try:
            clip_ctx = load_clip_features(clip_id, avdi=avdi, maybe_stream=True)
        except Exception as exc:
            log.error("[W%d] 클립 피처 로드 실패 %s: %s", worker_id, clip_id[:8], exc)
            result_q.put({"type": "error", "clip_id": clip_id, "exc": exc})
            continue

        # ── 유효 t0_us 범위 계산 ─────────────────────────────
        # egomotion.time_range는 실제 카메라 데이터보다 훨씬 긴 메타데이터 범위를 반환하는
        # 경우가 있음 (데이터셋 메타데이터 불일치). 카메라 타임스탬프 실측값으로 상한 클리핑.
        ego = clip_ctx["egomotion"]
        clip_start_us, ego_end_us = ego.time_range

        # 카메라별 실제 마지막 타임스탬프 중 최솟값 사용 (extract_sample에서 카메라도 필요)
        cam_end_us = min(
            int(cam.timestamps[-1])
            for cam in clip_ctx["cameras"]
            if cam.timestamps is not None and len(cam.timestamps) > 0
        )
        # 유효 상한 = egomotion 상한과 카메라 상한 중 더 작은 값
        clip_end_us = min(ego_end_us, cam_end_us)

        # 히스토리 1.6초 + 미래 6.4초를 확보할 수 있는 구간
        t0_min = clip_start_us + _HIST_DUR_US
        t0_max = clip_end_us   - _FUT_DUR_US
        if t0_min > t0_max:
            log.warning("[W%d] 클립이 너무 짧음 (%s): %.1fs", worker_id, clip_id[:8],
                        (clip_end_us - clip_start_us) / 1e6)
            result_q.put({"type": "error", "clip_id": clip_id,
                          "exc": RuntimeError("클립 길이 부족")})
            continue

        t0_list = list(range(t0_min, t0_max + 1, step_us))
        log.info("[W%d] %s: %.1f~%.1fs, %d 프레임 추출 시작",
                 worker_id, clip_id[:8],
                 t0_min / 1e6, t0_max / 1e6, len(t0_list))

        # ── 각 t0_us별 raw 데이터 추출 (seek+decode만) ──────
        n_ok, n_err = 0, 0
        for t0_us in t0_list:
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

        # 클립 완료 마커 (GPU 메인이 h5 저장 트리거)
        result_q.put({
            "type":    "clip_done",
            "clip_id": clip_id,
            "n_ok":    n_ok,
            "n_err":   n_err,
        })


# ══════════════════════════════════════════════════════
# CLI 인자 파싱
# ══════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alpamayo1_5 모델로 PCL 연속 데이터셋 생성 (test 클립 전용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest", 
        default="../alpamayo_dataset/data/split/split_manifest.json", 
        metavar="JSON",
        help="split_manifest.json 경로 (clips_test 키 사용). "
             "생략 시 --dataset-dir/clip_index.parquet의 split==test 클립 전체 사용",
    )
    p.add_argument(
        "--out-dir", required=True, metavar="DIR",
        help="클립 단위 .h5 파일 저장 디렉토리",
    )
    p.add_argument(
        "--model-dir",
        default="../alpamayo_dataset/data/alpamayo_1.5_weights",
        metavar="DIR",
        help="Alpamayo1_5 가중치 디렉토리 (기본: alpamayo_1.5_weights)",
    )
    p.add_argument(
        "--dataset-dir",
        default="../alpamayo_dataset/data/nvidia_test_scenario",
        metavar="DIR",
        help="nvidia_test_scenario 로컬 데이터셋 디렉토리",
    )
    p.add_argument(
        "--n-clips", type=int, default=0, metavar="N",
        help="처리할 클립 수 (0 = 전체 test 클립, 기본: 0)",
    )
    p.add_argument(
        "--step-us", type=int, default=_STEP_US, metavar="US",
        help=f"샘플링 간격 [μs] (기본: {_STEP_US} = 100ms)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="이미 존재하는 클립 h5도 재생성",
    )
    p.add_argument(
        "--download-workers", type=int, default=2, metavar="N",
        help="병렬 다운로드 워커 수 (기본: 2)",
    )
    p.add_argument(
        "--prefetch", type=int, default=8, metavar="N",
        help="result_q 최대 크기 (기본: 8). 큐가 꽉 차면 워커 자동 대기",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="CUDA 랜덤 시드 (기본: 42)",
    )
    p.add_argument(
        "--viz-dir", default=None, metavar="DIR",
        help="시각화 PNG 저장 디렉토리 (기본: --out-dir 와 동일). --no-viz로 비활성화",
    )
    p.add_argument(
        "--no-viz", action="store_true",
        help="시각화 PNG 생성 안 함",
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

    # ── test 클립 목록 ────────────────────────────────────────
    # 우선순위: --manifest > clip_index.parquet (split==test)
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        test_clips: list[str] = manifest["clips_test"]
        n_total_test = manifest["n_clips_test"]
        log.info("manifest 사용: %s", args.manifest)
    else:
        import pandas as pd
        ci_path = pathlib.Path(args.dataset_dir) / "clip_index.parquet"
        ci = pd.read_parquet(ci_path)
        test_clips = ci[ci["split"] == "test"].index.tolist()
        n_total_test = len(test_clips)
        log.info("clip_index.parquet 사용: split==test 클립 %d개", n_total_test)

    if args.n_clips > 0:
        test_clips = test_clips[: args.n_clips]

    log.info("대상 클립: %d개 선택 (전체 test %d개)",
             len(test_clips), n_total_test)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 시각화 디렉토리: --no-viz면 None, --viz-dir 지정 시 해당 경로, 기본은 out_dir
    viz_dir: pathlib.Path | None = None
    if not args.no_viz:
        viz_dir = pathlib.Path(args.viz_dir) if args.viz_dir else out_dir
        viz_dir.mkdir(parents=True, exist_ok=True)

    # ── CUDA 시드 고정 ────────────────────────────────────────
    torch.cuda.manual_seed_all(args.seed)

    # ── R1 모델 로드 ──────────────────────────────────────────
    log.info("Alpamayo1_5 모델 로딩: %s", args.model_dir)
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper

    model = Alpamayo1_5.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    device    = next(model.parameters()).device
    log.info("모델 로드 완료 (device: %s)", device)

    # ── 로컬 AVDI 초기화 ──────────────────────────────────────
    avdi = _make_local_avdi(args.dataset_dir)

    # ══════════════════════════════════════════════════════════
    # 병렬 파이프라인
    #
    #  워커1: [─클립A 로드(1회)─→ 프레임 100개 추출─→ clip_done─][─클립C─→ ...]
    #  워커2:   [─클립B 로드(1회)─→ 프레임 100개 추출─→ clip_done─][─클립D─→ ...]
    #  GPU  :  [추론][추론]...[저장] [추론][추론]...[저장] ...
    #
    #  result_q maxsize(prefetch)로 워커가 앞서 달리는 양을 제한.
    # ══════════════════════════════════════════════════════════
    n_workers = args.download_workers
    work_q:   queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue(maxsize=args.prefetch)

    # 클립 목록 적재 + 종료 신호
    for clip_id in test_clips:
        work_q.put(clip_id)
    for _ in range(n_workers):
        work_q.put(_SENTINEL)

    # 다운로드 워커 시작
    workers = []
    for wid in range(1, n_workers + 1):
        t = threading.Thread(
            target=_download_worker,
            args=(wid, work_q, result_q, avdi,
                  out_dir, args.step_us, args.overwrite),
            daemon=True,
            name=f"download-{wid}",
        )
        t.start()
        workers.append(t)

    # 모든 워커 완료 시 result_q 종료 신호 전송
    def _monitor() -> None:
        for w in workers:
            w.join()
        result_q.put(_SENTINEL)

    threading.Thread(target=_monitor, daemon=True, name="monitor").start()

    log.info("워커 %d개 시작 | prefetch=%d | step=%dms",
             n_workers, args.prefetch, args.step_us // 1000)

    # ── GPU 메인 루프 ──────────────────────────────────────────
    # 클립별 프레임 누적 버퍼 (cotend 등 소형 데이터만 보관, 카메라 이미지는 추론 후 즉시 해제)
    accumulators: dict[str, list[tuple[int, dict]]] = {}

    n_clips_done = 0
    n_clips_skip = 0
    n_clips_fail = 0
    n_frames_total = 0

    while True:
        item = result_q.get()
        if item is _SENTINEL:
            break

        clip_id = item["clip_id"]

        if item["type"] == "skip":
            # 이미 존재하는 클립 → 건너뜀
            n_clips_skip += 1

        elif item["type"] == "error":
            log.error("오류 (%s): %s", clip_id[:8], item["exc"])
            # 누적 중이었다면 정리
            accumulators.pop(clip_id, None)
            n_clips_fail += 1

        elif item["type"] == "frame":
            # R1 추론 → 소형 데이터만 추출, raw 이미지 즉시 해제
            result = _infer_one(model, processor, item["raw"], device)
            del item["raw"]   # GPU 메모리 / 시스템 메모리 즉시 반환

            if result is not None:
                if clip_id not in accumulators:
                    accumulators[clip_id] = []
                accumulators[clip_id].append((item["t0_us"], result))

            # 진행 로그 (10프레임마다)
            n_accumulated = len(accumulators.get(clip_id, []))
            if n_accumulated % 10 == 0 and n_accumulated > 0:
                log.info("  %s  프레임 %d개 처리 중...", clip_id[:8], n_accumulated)

        elif item["type"] == "clip_done":
            # 클립 완료 → h5 저장
            frames = accumulators.pop(clip_id, [])
            n_ok, n_err = item["n_ok"], item["n_err"]

            if not frames:
                log.error("클립 저장 실패 (0 프레임): %s", clip_id[:8])
                n_clips_fail += 1
            else:
                try:
                    out_path = _save_clip_h5(clip_id, frames, out_dir, args.step_us)
                    size_mb  = out_path.stat().st_size / 1e6
                    dur_s    = frames[-1][0] / 1e6 - frames[0][0] / 1e6 if len(frames) > 1 else 0.0
                    log.info(
                        "저장 완료: %s  |  %d프레임  %.1fs  %.1fMB  (워커 오류 %d)",
                        out_path.name, len(frames), dur_s, size_mb, n_err,
                    )
                    n_clips_done   += 1
                    n_frames_total += len(frames)

                    # 시각화 PNG 저장 (h5 저장 직후, 프레임 데이터가 메모리에 있는 동안)
                    if viz_dir is not None:
                        viz_path = viz_dir / f"{clip_id}_overview.png"
                        try:
                            _save_clip_viz(clip_id, frames, viz_path, args.step_us)
                            log.info("시각화 저장: %s", viz_path.name)
                        except Exception:
                            log.warning("시각화 저장 실패 (계속 진행): %s", viz_path.name,
                                        exc_info=True)

                except Exception:
                    log.exception("h5 저장 실패: %s", clip_id[:8])
                    n_clips_fail += 1

            log.info("진행 현황 — 저장: %d | 건너뜀: %d | 실패: %d / 전체: %d",
                     n_clips_done, n_clips_skip, n_clips_fail, len(test_clips))

    # ── 최종 요약 ─────────────────────────────────────────────
    log.info("=" * 65)
    log.info("완료")
    log.info("  저장된 클립  : %d", n_clips_done)
    log.info("  건너뜀       : %d  (이미 존재)", n_clips_skip)
    log.info("  실패         : %d", n_clips_fail)
    log.info("  총 프레임    : %d  (평균 %.1f개/클립)",
             n_frames_total,
             n_frames_total / max(n_clips_done, 1))
    log.info("  h5 경로      : %s", out_dir)
    if viz_dir is not None:
        log.info("  시각화 경로  : %s", viz_dir)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
