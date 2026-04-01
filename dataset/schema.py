# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo → MPC 가중치 학습용 데이터셋 스키마.

이 모듈은 수집 데이터의 구조를 정의합니다.
전체 수집 방법은 alpamayo_dataset/DATASET_GUIDE.md 를 참조하세요.

샘플당 세 가지 데이터 축
------------------------
A  GT:     실제(ground-truth) 미래 궤적 + 파생 ego 상태 [x, y, speed, yaw, lon_accel]
B  입력:   egomotion 히스토리 (Alpamayo 모델 입력)
           카메라 이미지는 수집 시 PNG 시각화에만 사용하고 h5에는 저장하지 않음
C  출력:   Alpamayo 모델 결과
             - cot_text:             Chain-of-Causation 추론 텍스트
             - pred_xyz / pred_rot:  예측 미래 궤적 (64스텝, 6.4초)
             - cotend_hidden_state:  (4096,) VLM이 <traj_future_start>를 생성하는 순간의
                                    lm_head 입력 벡터. "왜 이 행동을 선택했는지"가 압축된
                                    상황 표현 벡터로, MPC 가중치 MLP의 입력으로 사용됨.

좌표계
------
모든 xyz/rot는 t0 시점의 ego를 원점으로 하는 로컬 좌표계 (차량 전방 = +x).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AlpamayoSample:
    """MPC 가중치 학습을 위한 단일 수집 데이터.

    기본 크기 가정:
        T_hist = 16   (10 Hz, 1.6초)
        T_fut  = 64   (10 Hz, 6.4초)
        N_cam  = 4    카메라 수
        N_frm  = 4    카메라당 프레임 수
        H_dim  = 4096 VLM 히든 크기 (Qwen3-VL-8B)

    카메라 이미지(camera_images, camera_ids, frame_timestamps_us)는 h5 저장 없이
    수집 시 PNG 시각화에만 사용됩니다. h5에서 로드한 경우에는 None입니다.
    """

    # --- 메타데이터 ---
    clip_id: str
    t0_us: int  # 기준 타임스탬프 (마이크로초)

    # --- A. GT 축 ---
    gt_future_xyz: np.ndarray         # (64, 3)    float32  로컬 ego 좌표계 (모델 출력 비교용)
    gt_future_rot: np.ndarray         # (64, 3, 3) float32  로컬 ego 좌표계 (모델 출력 비교용)
    gt_future_ego_states: np.ndarray  # (64, 5)    float32  [x, y, speed, yaw, lon_accel]
    future_timestamps_us: np.ndarray  # (64,)      int64
    # raw egomotion (전역 좌표계, 다운로드 원본 그대로)
    gt_future_xyz_global: np.ndarray  # (64, 3)    float64  전역 좌표계 [x, y, z]
    gt_future_quat_global: np.ndarray # (64, 4)    float64  [qx, qy, qz, qw]
    gt_future_vel: np.ndarray         # (64, 3)    float64  [vx, vy, vz]
    gt_future_acc: np.ndarray         # (64, 3)    float64  [ax, ay, az]
    gt_future_curv: np.ndarray        # (64, 1)    float64  curvature

    # --- B. 입력 축 (ego history) ---
    ego_history_xyz: np.ndarray         # (16, 3)    float32  로컬 ego 좌표계 (모델 입력용)
    ego_history_rot: np.ndarray         # (16, 3, 3) float32  로컬 ego 좌표계 (모델 입력용)
    ego_history_ego_states: np.ndarray  # (16, 5)    float32  [x, y, speed, yaw, lon_accel]
    history_timestamps_us: np.ndarray   # (16,)      int64
    # raw egomotion (전역 좌표계, 다운로드 원본 그대로)
    ego_history_xyz_global: np.ndarray  # (16, 3)    float64  전역 좌표계 [x, y, z]
    ego_history_quat_global: np.ndarray # (16, 4)    float64  [qx, qy, qz, qw]
    ego_history_vel: np.ndarray         # (16, 3)    float64  [vx, vy, vz]
    ego_history_acc: np.ndarray         # (16, 3)    float64  [ax, ay, az]
    ego_history_curv: np.ndarray        # (16, 1)    float64  curvature

    # --- C. 모델 출력 축 ---
    cot_text: str                       # Chain-of-Causation 추론 텍스트
    meta_action_text: str               # 메타 액션 설명 (없을 수 있음)
    pred_xyz: np.ndarray                # (64, 3)    float32  로컬 ego 좌표계
    pred_rot: np.ndarray                # (64, 3, 3) float32  로컬 ego 좌표계
    cotend_hidden_state: np.ndarray     # (4096,)    float32  CoT 끝 시점의 VLM 히든 스테이트

    # --- D. 카메라 (h5 미저장, 수집 시 시각화 전용) ---
    camera_images: np.ndarray | None = field(default=None)  # (N_cam, N_frm, 3, H, W)  uint8
    camera_ids: list | None = field(default=None)           # N_cam개 카메라 이름 문자열
    frame_timestamps_us: np.ndarray | None = field(default=None)  # (N_cam, N_frm)  int64


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def derive_ego_states(xyz: np.ndarray, rot: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """로컬 좌표계 xyz, rot 배열로부터 ego 상태 [x, y, speed, yaw, lon_accel]을 계산.

    Args:
        xyz: (T, 3)     로컬 ego 좌표계에서의 위치
        rot: (T, 3, 3)  로컬 ego 좌표계에서의 회전 행렬
        dt:  타임스텝 (초 단위, 기본값 0.1s = 10 Hz)

    Returns:
        states: (T, 5) float32 — 열 순서: [x, y, speed, yaw, lon_accel]

    비고:
        - yaw: atan2(R[1,0], R[0,0])으로 지면 투영 방위각 추출.
        - speed, lon_accel: numpy.gradient 사용 (내부 중앙차분, 경계 전방/후방차분).
    """
    x = xyz[:, 0]
    y = xyz[:, 1]

    # 회전 행렬에서 yaw 추출 (지면 평면 x축 방향)
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])

    # 속도: ||(dx/dt, dy/dt)||
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    speed = np.sqrt(dx**2 + dy**2)

    # 종방향 가속도: d(speed)/dt
    lon_accel = np.gradient(speed, dt)

    return np.stack([x, y, speed, yaw, lon_accel], axis=-1).astype(np.float32)
