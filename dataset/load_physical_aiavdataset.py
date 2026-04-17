# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""모델 추론을 위해 physical_ai_av.PhysicalAIAVDatasetInterface에서 데이터를 로드합니다."""

from typing import Any

import numpy as np
import physical_ai_av
import scipy.spatial.transform as spt
import torch
from einops import rearrange

# 카메라 이름 → 인덱스 매핑 (모듈 레벨 공유)
_CAMERA_NAME_TO_INDEX: dict[str, int] = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}


def load_clip_features(
    clip_id: str,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    maybe_stream: bool = True,
    camera_features: list | None = None,
) -> dict:
    """클립의 egomotion + 카메라 피처를 한 번만 스트리밍/로드.

    동일 클립의 여러 t0_us를 처리할 때 이 함수를 먼저 호출하여
    피처 객체를 공유하면 중복 스트리밍을 제거할 수 있습니다.

    인자:
        clip_id: 클립 ID
        avdi: 선택 사항인 사전 초기화된 PhysicalAIAVDatasetInterface
        maybe_stream: HuggingFace 스트리밍 여부
        camera_features: 로드할 카메라 피처 목록. None이면 4개 기본 카메라 사용.

    반환값:
        extract_sample_from_clip_features()에 전달할 클립 컨텍스트 딕셔너리:
            - clip_id: 클립 ID
            - egomotion: egomotion 피처 객체 (callable)
            - cameras: 카메라 피처 객체 리스트
            - camera_features: 카메라 피처 이름 리스트
    """
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    if camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    egomotion = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=maybe_stream
    )
    cameras = [
        avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=maybe_stream)
        for cam_feature in camera_features
    ]

    return {
        "clip_id": clip_id,
        "egomotion": egomotion,
        "cameras": cameras,
        "camera_features": camera_features,
    }


def extract_sample_from_clip_features(
    clip_features: dict,
    t0_us: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    num_frames: int = 4,
) -> dict[str, Any]:
    """미리 로드된 클립 피처에서 특정 t0_us의 샘플을 추출.

    load_clip_features()의 반환값을 받아 load_physical_aiavdataset()과 동일한
    형식의 딕셔너리를 반환합니다.

    동일 클립의 여러 t0_us를 처리할 때 클립당 스트리밍 횟수를 1회로 줄입니다.
    egomotion 및 camera 피처 객체는 여러 타임스탬프 호출을 지원합니다.
    """
    clip_id = clip_features["clip_id"]
    egomotion = clip_features["egomotion"]
    cameras = clip_features["cameras"]
    camera_features = clip_features["camera_features"]

    assert t0_us > num_history_steps * time_step * 1_000_000, (
        "t0_us는 과거 시간 범위보다 커야 합니다."
    )

    # 타임스탬프 계산
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    # egomotion 추출 (이미 로드된 객체에서 타임스탬프 인덱싱만 수행)
    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()
    ego_history_vel  = ego_history.velocity        # (T_hist, 3) [vx, vy, vz]  global frame
    ego_history_acc  = ego_history.acceleration    # (T_hist, 3) [ax, ay, az]  global frame
    ego_history_curv = ego_history.curvature       # (T_hist, 1)

    ego_future = egomotion(future_timestamps)
    ego_future_xyz  = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()
    ego_future_vel  = ego_future.velocity          # (T_fut, 3) [vx, vy, vz]  global frame
    ego_future_acc  = ego_future.acceleration      # (T_fut, 3) [ax, ay, az]  global frame
    ego_future_curv = ego_future.curvature         # (T_fut, 1)

    # 로컬 좌표계 변환 (모델 입력용)
    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = (
        torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_rot_tensor = (
        torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)
    )

    # 카메라 이미지 추출 (이미 로드된 카메라 객체에서 seek+decode만 수행)
    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    for cam_feature, camera in zip(camera_features, cameras):
        frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")

        if isinstance(cam_feature, str):
            cam_name = cam_feature.split("/")[-1] if "/" in cam_feature else cam_feature
            cam_name = cam_name.lower()
        else:
            raise ValueError(f"예상치 못한 카메라 기능 유형: {type(cam_feature)}")
        cam_idx = _CAMERA_NAME_TO_INDEX.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        # 모델 입력용 (로컬 좌표계, t0 기준)
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        # 원본 egomotion (전역 좌표계, 변조 없음) — h5 저장용
        "ego_history_xyz_global": torch.from_numpy(ego_history_xyz.astype(np.float64)),
        "ego_history_quat_global": torch.from_numpy(ego_history_quat.astype(np.float64)),
        "ego_history_vel":  torch.from_numpy(ego_history_vel.astype(np.float64)),
        "ego_history_acc":  torch.from_numpy(ego_history_acc.astype(np.float64)),
        "ego_history_curv": torch.from_numpy(ego_history_curv.astype(np.float64)),
        "ego_future_xyz_global": torch.from_numpy(ego_future_xyz.astype(np.float64)),
        "ego_future_quat_global": torch.from_numpy(ego_future_quat.astype(np.float64)),
        "ego_future_vel":  torch.from_numpy(ego_future_vel.astype(np.float64)),
        "ego_future_acc":  torch.from_numpy(ego_future_acc.astype(np.float64)),
        "ego_future_curv": torch.from_numpy(ego_future_curv.astype(np.float64)),
        "history_timestamps_us": torch.from_numpy(history_timestamps),
        "future_timestamps_us":  torch.from_numpy(future_timestamps),
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    maybe_stream: bool = True,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
) -> dict[str, Any]:
    """모델 추론을 위해 physical_ai_av에서 데이터를 로드합니다.

    단일 샘플 로드 시 사용. 동일 클립에서 여러 t0_us를 처리할 때는
    load_clip_features() + extract_sample_from_clip_features() 조합을 사용하세요.

    인자:
        clip_id: 데이터를 로드할 클립 ID.
        t0_us: 궤적을 샘플링할 타임스탬프(마이크로초).
        avdi: 선택 사항인 사전 초기화된 PhysicalAIAVDatasetInterface.
        maybe_stream: HuggingFace에서 데이터를 스트리밍할지 여부.
        num_history_steps: 과거 궤적 단계 수(기본값: 10Hz에서 1.6초 동안 16단계).
        num_future_steps: 미래 궤적 단계 수(기본값: 10Hz에서 6.4초 동안 64단계).
        time_step: 궤적 지점 간의 시간 간격(초, 기본값: 0.1s = 10Hz).
        camera_features: 로드할 카메라 기능 목록. None이면 4개 기본 카메라 사용.
        num_frames: 카메라당 로드할 프레임 수(기본값: 4).

    반환값:
        extract_sample_from_clip_features()와 동일한 형식의 딕셔너리.
    """
    clip_ctx = load_clip_features(
        clip_id, avdi=avdi, maybe_stream=maybe_stream, camera_features=camera_features
    )
    return extract_sample_from_clip_features(
        clip_ctx, t0_us,
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
        time_step=time_step,
        num_frames=num_frames,
    )
