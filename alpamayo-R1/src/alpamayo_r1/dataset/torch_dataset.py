# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MPC 가중치 학습용 PyTorch Dataset.

AlpamayoDatasetCollector가 저장한 .h5 파일들을 로드해 텐서 딕셔너리로 반환합니다.
전체 수집 방법은 alpamayo_dataset/DATASET_GUIDE.md 를 참조하세요.

주요 텐서 (배치 기준)
---------------------
    cotend_hidden_state  (B, 4096)   ← MLP 입력: 상황별 VLM 표현 벡터
    gt_future_ego_states (B, 64, 5)  ← GT: [x, y, speed, yaw, lon_accel], 레이블 도출 가능
    pred_xyz             (B, 64, 3)  ← Alpamayo 예측 궤적 (검증/비교용)
    gt_future_xyz        (B, 64, 3)  ← GT 미래 궤적
    ego_history_ego_states (B,16, 5) ← 히스토리 ego 상태 (컨텍스트 특징으로 추가 가능)
    cot_text             list[str]   ← CoC 추론 텍스트

사용 예
-------
    from alpamayo_r1.dataset.torch_dataset import AlpamayoMPCDataset
    from torch.utils.data import DataLoader

    dataset = AlpamayoMPCDataset(
        "/workspace/alpamayo_dataset/data/collected",
        load_images=False,   # 이미지 불필요 시 False → 로딩 빠름
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        hs = batch["cotend_hidden_state"]    # (B, 4096) → MLP 입력
        gt = batch["gt_future_ego_states"]   # (B, 64, 5) → 레이블 도출
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AlpamayoMPCDataset(Dataset):
    """AlpamayoDatasetCollector가 저장한 HDF5 파일을 읽는 PyTorch Dataset.

    Args:
        data_dir:  .h5 파일이 저장된 디렉토리
        transform: 샘플 딕셔너리에 적용할 선택적 변환 함수
        load_images: 카메라 이미지를 로드할지 여부 (기본 True, 크기가 크므로 필요 시 False)
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable[[dict], dict] | None = None,
        load_images: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f".h5 파일이 없습니다: {self.data_dir}")
        self.transform = transform
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            sample: dict = {
                # 메타데이터
                "clip_id": str(f.attrs["clip_id"]),
                "t0_us": int(f.attrs["t0_us"]),

                # A. GT 축
                "gt_future_xyz": torch.from_numpy(f["gt/future_xyz"][:]),          # (64, 3)
                "gt_future_rot": torch.from_numpy(f["gt/future_rot"][:]),          # (64, 3, 3)
                "gt_future_ego_states": torch.from_numpy(                           # (64, 5)
                    f["gt/future_ego_states"][:]
                ),

                # B. 입력 축
                "frame_timestamps_us": torch.from_numpy(                            # (N_cam, N_frm)
                    f["input/frame_timestamps_us"][:]
                ),
                "ego_history_xyz": torch.from_numpy(f["input/ego_history_xyz"][:]),   # (16, 3)
                "ego_history_rot": torch.from_numpy(f["input/ego_history_rot"][:]),   # (16, 3, 3)
                "ego_history_ego_states": torch.from_numpy(                           # (16, 5)
                    f["input/ego_history_ego_states"][:]
                ),
                "history_timestamps_us": torch.from_numpy(                           # (16,)
                    f["input/history_timestamps_us"][:]
                ),

                # C. 모델 출력 축
                "cot_text": str(f["output"].attrs["cot_text"]),
                "meta_action_text": str(f["output"].attrs["meta_action_text"]),
                "pred_xyz": torch.from_numpy(f["output/pred_xyz"][:]),              # (64, 3)
                "pred_rot": torch.from_numpy(f["output/pred_rot"][:]),              # (64, 3, 3)
                "cotend_hidden_state": torch.from_numpy(                             # (4096,)
                    f["output/cotend_hidden_state"][:]
                ),
            }

            if self.load_images:
                sample["camera_images"] = torch.from_numpy(                         # (N_cam, N_frm, 3, H, W)
                    f["input/camera_images"][:]
                )
                sample["camera_ids"] = [
                    s.decode("utf-8") for s in f["input/camera_ids"][:]
                ]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_metadata(self) -> list[dict]:
        """모든 샘플의 clip_id와 t0_us를 빠르게 조회 (텐서 로드 없음)."""
        meta = []
        for p in self.files:
            with h5py.File(p, "r") as f:
                meta.append({
                    "clip_id": str(f.attrs["clip_id"]),
                    "t0_us": int(f.attrs["t0_us"]),
                    "path": str(p),
                })
        return meta
