# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 데이터셋 수집기.

전체 사용 방법은 alpamayo_dataset/DATASET_GUIDE.md 를 참조하세요.

AlpamayoDatasetCollector.collect() 내부 처리 순서
--------------------------------------------------
1. load_physical_aiavdataset()  →  카메라 이미지, ego 히스토리/미래 궤적 로드
2. processor.apply_chat_template()  →  멀티모달 토크나이징
3. model.sample_trajectories_from_data_with_vlm_rollout(
       return_cotend_hidden_state=True   ← alpamayo1_5.py 내부 hook이 활성화됨
   )
   ┌─ VLM generate(): CoC 텍스트 생성
   │    lm_head pre-hook이 매 디코드 스텝마다 hidden state를 덮어씌움
   │    → 최종값 = <traj_future_start> 위치의 hidden state = cotend (4096-dim)
   └─ Diffusion 샘플링: 미래 64스텝 궤적 생성
4. AlpamayoSample 구성  →  PNG 시각화 저장 (viz_dir 지정 시)
5. HDF5 저장 (카메라 이미지 제외, 학습 필요 데이터만)

저장 파일명: {clip_id}__{t0_us}.h5
저장 그룹:   gt/  |  input/  |  output/
상세 스키마: alpamayo_dataset/DATASET_GUIDE.md §4 참조
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

from .load_physical_aiavdataset import (
    extract_sample_from_clip_features,
    load_clip_features,
    load_physical_aiavdataset,
)
from .schema import AlpamayoSample, derive_ego_states

log = logging.getLogger(__name__)

# 카메라 인덱스 → 이름 매핑 (load_physical_aiavdataset.py의 camera_name_to_index와 동일)
_CAMERA_IDX_TO_NAME: dict[int, str] = {
    0: "camera_cross_left_120fov",
    1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    4: "camera_rear_tele_30fov",
    5: "camera_rear_right_70fov",
    6: "camera_front_tele_30fov",
}

# 히스토리 타임스텝: 10 Hz = 100 ms = 100_000 us
_DT_US = 100_000


def _first_str(x, default: str = "") -> str:
    """중첩 리스트/배열에서 첫 번째 문자열을 재귀적으로 추출."""
    while not isinstance(x, str):
        try:
            x = x[0]
        except (IndexError, KeyError, TypeError):
            return default
    return x


class AlpamayoDatasetCollector:
    """Alpamayo 추론 샘플을 수집해 HDF5 파일로 저장.

    Args:
        model:           로드된 Alpamayo1_5 모델 (CUDA 배치 완료 상태)
        output_dir:      .h5 파일이 저장될 디렉토리
        viz_dir:         PNG 시각화를 저장할 디렉토리.
                         None이면 output_dir와 같은 위치에 저장.
                         False이면 시각화를 생성하지 않음.
        num_traj_samples: 추론 시 샘플링할 궤적 수 (기본 1)
        max_generation_length: VLM CoC 최대 생성 토큰 수
        top_p:           nucleus sampling p
        temperature:     sampling temperature
    """

    def __init__(
        self,
        model: Alpamayo1_5,
        output_dir: str | Path,
        *,
        viz_dir: str | Path | None | bool = None,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
        top_p: float = 0.98,
        temperature: float = 0.6,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = helper.get_processor(model.tokenizer)

        # viz_dir 처리:  None → output_dir 사용 / False → 비활성화 / Path → 해당 경로
        if viz_dir is False:
            self.viz_dir: Path | None = None
            self._do_viz = False
        else:
            self._do_viz = True
            self.viz_dir = Path(viz_dir) if viz_dir else self.output_dir
            self.viz_dir.mkdir(parents=True, exist_ok=True)

        # sample_trajectories_from_data_with_vlm_rollout에 전달할 고정 인자
        self._infer_kwargs = dict(
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            top_p=top_p,
            temperature=temperature,
            return_extra=True,
            return_cotend_hidden_state=True,  # alpamayo1_5.py 내부 hook으로 캡처
        )

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def download_raw(self, clip_id: str, t0_us: int, *, avdi=None) -> dict:
        """데이터셋에서 raw 데이터만 로드 (추론 없음, 단일 샘플).

        단일 (clip_id, t0_us) 샘플만 처리할 때 사용.
        동일 클립에서 여러 t0_us를 처리할 때는
        download_clip_context() + extract_raw_from_context() 조합을 사용하세요.
        """
        return load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)

    def download_clip_context(self, clip_id: str, *, avdi=None) -> dict:
        """클립의 egomotion + 카메라 피처를 한 번만 로드 (배치 다운로드용).

        동일 클립의 여러 t0_us를 처리할 때 클립당 스트리밍 횟수를 1회로 줄입니다.
        반환값을 extract_raw_from_context()에 전달해 각 t0_us별 데이터를 추출하세요.

        사용 예::
            ctx = collector.download_clip_context(clip_id, avdi=avdi)
            for t0_us in t0_us_list:
                raw = collector.extract_raw_from_context(ctx, t0_us)
                collector.collect_from_raw(raw, clip_id, t0_us)
        """
        return load_clip_features(clip_id, avdi=avdi)

    def extract_raw_from_context(self, clip_context: dict, t0_us: int) -> dict:
        """미리 로드된 클립 컨텍스트에서 특정 t0_us의 raw 데이터를 추출.

        download_clip_context()의 반환값을 받아 download_raw()와 동일한 형식으로 반환.
        egomotion/카메라 스트리밍 없이 seek+decode만 수행하므로 매우 빠릅니다.
        """
        return extract_sample_from_clip_features(clip_context, t0_us)

    def collect(
        self,
        clip_id: str,
        t0_us: int,
        *,
        avdi=None,
        overwrite: bool = False,
    ) -> Path | None:
        """단일 샘플을 수집하고 HDF5로 저장 (다운로드 포함 순차 버전).

        프리페치 없이 단일 샘플을 처리할 때 사용.
        병렬 수집 시에는 download_raw() + collect_from_raw() 조합을 사용.
        """
        out_path = self.output_dir / f"{clip_id}__{t0_us}.h5"
        if out_path.exists() and not overwrite:
            log.info("건너뜀 (이미 존재): %s", out_path.name)
            return None

        log.info("  [1/4] 데이터셋 로드 중...")
        raw = self.download_raw(clip_id, t0_us, avdi=avdi)
        return self.collect_from_raw(raw, clip_id, t0_us)

    def collect_from_raw(self, raw: dict, clip_id: str, t0_us: int) -> Path:
        """미리 다운로드된 raw 데이터로 추론 + 저장.

        프리페치 스레드가 download_raw()로 받아온 데이터를 GPU 스레드에서
        처리할 때 호출. 다운로드 단계를 건너뛰어 GPU 대기 시간을 최소화.
        """
        out_path = self.output_dir / f"{clip_id}__{t0_us}.h5"

        # 1. 모델 입력 준비 + 추론 (cotend hidden state는 모델 내부 hook으로 extra에 포함)
        log.info("  [2/4] 추론 중  (VLM CoC 생성 → Diffusion 샘플링)...")
        model_inputs = self._prepare_inputs(raw)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    **self._infer_kwargs,
                )
            )

        # 2. AlpamayoSample 구성 (카메라 이미지 포함 — 시각화에 사용 후 h5에는 저장 안 함)
        sample = self._build_sample(raw, pred_xyz, pred_rot, extra)

        # 3. PNG 시각화 (카메라 이미지가 메모리에 있는 동안 생성)
        if self._do_viz and sample.camera_images is not None:
            log.info("  [3/4] 시각화 저장 중...")
            viz_path = self.viz_dir / f"{clip_id}__{t0_us}.png"
            try:
                _visualize_sample(sample, viz_path)
                log.info("        → %s", viz_path.name)
            except Exception:
                log.warning("시각화 실패 (계속 진행): %s", viz_path.name, exc_info=True)
        else:
            log.info("  [3/4] 시각화 건너뜀")

        # 4. HDF5 저장 (카메라 이미지 제외)
        log.info("  [4/4] HDF5 저장 중...")
        _write_hdf5(sample, out_path)
        log.info("        → %s", out_path.name)

        torch.cuda.empty_cache()
        return out_path

    # ------------------------------------------------------------------
    # 내부: 모델 입력 준비
    # ------------------------------------------------------------------

    def _prepare_inputs(self, raw: dict) -> dict:
        """raw 데이터에서 모델 입력 딕셔너리를 구성."""
        messages = helper.create_message(
            raw["image_frames"].flatten(0, 1),
            camera_indices=raw["camera_indices"],
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return helper.to_device(
            {
                "tokenized_data": tokenized,
                "ego_history_xyz": raw["ego_history_xyz"],
                "ego_history_rot": raw["ego_history_rot"],
            },
            device=device,
        )

    # ------------------------------------------------------------------
    # 내부: AlpamayoSample 구성
    # ------------------------------------------------------------------

    def _build_sample(
        self,
        raw: dict,
        pred_xyz: torch.Tensor,
        pred_rot: torch.Tensor,
        extra: dict,
    ) -> AlpamayoSample:
        # --- A. GT ---
        gt_future_xyz = raw["ego_future_xyz"].cpu().numpy()[0, 0]   # (64, 3)
        gt_future_rot = raw["ego_future_rot"].cpu().numpy()[0, 0]   # (64, 3, 3)
        gt_future_ego = derive_ego_states(gt_future_xyz, gt_future_rot)

        # --- A. GT raw egomotion (전역 좌표계) ---
        gt_future_xyz_global  = raw["ego_future_xyz_global"].cpu().numpy()   # (64, 3)
        gt_future_quat_global = raw["ego_future_quat_global"].cpu().numpy()  # (64, 4)
        gt_future_vel         = raw["ego_future_vel"].cpu().numpy()          # (64, 3)
        gt_future_acc         = raw["ego_future_acc"].cpu().numpy()          # (64, 3)
        gt_future_curv        = raw["ego_future_curv"].cpu().numpy()         # (64, 1)
        future_ts             = raw["future_timestamps_us"].cpu().numpy()    # (64,)

        # --- B. 입력 (ego history) ---
        ego_hist_xyz = raw["ego_history_xyz"].cpu().numpy()[0, 0]   # (16, 3)
        ego_hist_rot = raw["ego_history_rot"].cpu().numpy()[0, 0]   # (16, 3, 3)
        ego_hist_ego = derive_ego_states(ego_hist_xyz, ego_hist_rot)

        # --- B. 입력 raw egomotion (전역 좌표계) ---
        ego_hist_xyz_global  = raw["ego_history_xyz_global"].cpu().numpy()   # (16, 3)
        ego_hist_quat_global = raw["ego_history_quat_global"].cpu().numpy()  # (16, 4)
        ego_hist_vel         = raw["ego_history_vel"].cpu().numpy()          # (16, 3)
        ego_hist_acc         = raw["ego_history_acc"].cpu().numpy()          # (16, 3)
        ego_hist_curv        = raw["ego_history_curv"].cpu().numpy()         # (16, 1)

        t0_us = int(raw["t0_us"])
        history_ts = raw["history_timestamps_us"].cpu().numpy()              # (16,)

        # --- C. 모델 출력 ---
        cot_text = str(_first_str(extra.get("cot", [""])))
        meta_text = str(_first_str(extra.get("meta_action", [""])))

        pred_xyz_np = pred_xyz.cpu().float().numpy()[0, 0, 0]       # (64, 3)
        pred_rot_np = pred_rot.cpu().float().numpy()[0, 0, 0]       # (64, 3, 3)

        raw_hs = extra.get("cotend_hidden_state")
        if raw_hs is not None:
            cotend_hs = raw_hs[0].float().cpu().numpy()             # (4096,)
        else:
            log.warning("cotend hidden state 캡처 실패 — 0으로 채움")
            cotend_hs = np.zeros(4096, dtype=np.float32)

        # --- D. 카메라 (시각화 전용, h5 미저장) ---
        camera_images = raw["image_frames"].cpu().numpy()           # (N_cam, N_frm, 3, H, W)
        camera_indices = raw["camera_indices"].cpu().numpy()
        camera_ids = [_CAMERA_IDX_TO_NAME.get(int(i), f"camera_{i}") for i in camera_indices]
        frame_ts = raw["absolute_timestamps"].cpu().numpy().astype(np.int64)

        return AlpamayoSample(
            clip_id=raw["clip_id"],
            t0_us=t0_us,
            gt_future_xyz=gt_future_xyz.astype(np.float32),
            gt_future_rot=gt_future_rot.astype(np.float32),
            gt_future_ego_states=gt_future_ego,
            future_timestamps_us=future_ts,
            gt_future_xyz_global=gt_future_xyz_global,
            gt_future_quat_global=gt_future_quat_global,
            gt_future_vel=gt_future_vel,
            gt_future_acc=gt_future_acc,
            gt_future_curv=gt_future_curv,
            ego_history_xyz=ego_hist_xyz.astype(np.float32),
            ego_history_rot=ego_hist_rot.astype(np.float32),
            ego_history_ego_states=ego_hist_ego,
            history_timestamps_us=history_ts,
            ego_history_xyz_global=ego_hist_xyz_global,
            ego_history_quat_global=ego_hist_quat_global,
            ego_history_vel=ego_hist_vel,
            ego_history_acc=ego_hist_acc,
            ego_history_curv=ego_hist_curv,
            cot_text=cot_text,
            meta_action_text=meta_text,
            pred_xyz=pred_xyz_np.astype(np.float32),
            pred_rot=pred_rot_np.astype(np.float32),
            cotend_hidden_state=cotend_hs.astype(np.float32),
            camera_images=camera_images,
            camera_ids=camera_ids,
            frame_timestamps_us=frame_ts,
        )


# ------------------------------------------------------------------
# HDF5 I/O
# ------------------------------------------------------------------

def _write_hdf5(sample: AlpamayoSample, path: Path) -> None:
    """AlpamayoSample을 HDF5 파일로 저장.

    카메라 이미지(input/camera_images)는 저장하지 않습니다.
    학습에 필요한 데이터(gt, ego history, cotend, pred)만 저장됩니다.
    """
    with h5py.File(path, "w") as f:
        f.attrs["clip_id"] = sample.clip_id
        f.attrs["t0_us"] = sample.t0_us

        # A. GT
        g = f.create_group("gt")
        g.create_dataset("future_xyz", data=sample.gt_future_xyz, compression="gzip")
        g.create_dataset("future_rot", data=sample.gt_future_rot, compression="gzip")
        g.create_dataset("future_ego_states", data=sample.gt_future_ego_states, compression="gzip")
        g["future_ego_states"].attrs["columns"] = ["x", "y", "speed", "yaw", "lon_accel"]
        g.create_dataset("future_timestamps_us", data=sample.future_timestamps_us)
        # raw egomotion (전역 좌표계, 다운로드 원본)
        g.create_dataset("future_xyz_global",  data=sample.gt_future_xyz_global,  compression="gzip")
        g.create_dataset("future_quat_global", data=sample.gt_future_quat_global, compression="gzip")
        g["future_quat_global"].attrs["columns"] = ["qx", "qy", "qz", "qw"]
        g.create_dataset("future_vel",  data=sample.gt_future_vel,  compression="gzip")
        g["future_vel"].attrs["columns"] = ["vx", "vy", "vz"]
        g.create_dataset("future_acc",  data=sample.gt_future_acc,  compression="gzip")
        g["future_acc"].attrs["columns"] = ["ax", "ay", "az"]
        g.create_dataset("future_curv", data=sample.gt_future_curv, compression="gzip")

        # B. 입력 (ego history만, 카메라 이미지 제외)
        g = f.create_group("input")
        g.create_dataset("ego_history_xyz", data=sample.ego_history_xyz, compression="gzip")
        g.create_dataset("ego_history_rot", data=sample.ego_history_rot, compression="gzip")
        g.create_dataset(
            "ego_history_ego_states",
            data=sample.ego_history_ego_states,
            compression="gzip",
        )
        g["ego_history_ego_states"].attrs["columns"] = ["x", "y", "speed", "yaw", "lon_accel"]
        g.create_dataset("history_timestamps_us", data=sample.history_timestamps_us)
        # raw egomotion (전역 좌표계, 다운로드 원본)
        g.create_dataset("ego_history_xyz_global",  data=sample.ego_history_xyz_global,  compression="gzip")
        g.create_dataset("ego_history_quat_global", data=sample.ego_history_quat_global, compression="gzip")
        g["ego_history_quat_global"].attrs["columns"] = ["qx", "qy", "qz", "qw"]
        g.create_dataset("ego_history_vel",  data=sample.ego_history_vel,  compression="gzip")
        g["ego_history_vel"].attrs["columns"] = ["vx", "vy", "vz"]
        g.create_dataset("ego_history_acc",  data=sample.ego_history_acc,  compression="gzip")
        g["ego_history_acc"].attrs["columns"] = ["ax", "ay", "az"]
        g.create_dataset("ego_history_curv", data=sample.ego_history_curv, compression="gzip")

        # C. 모델 출력
        g = f.create_group("output")
        g.attrs["cot_text"] = sample.cot_text
        g.attrs["meta_action_text"] = sample.meta_action_text
        g.create_dataset("pred_xyz", data=sample.pred_xyz, compression="gzip")
        g.create_dataset("pred_rot", data=sample.pred_rot, compression="gzip")
        g.create_dataset(
            "cotend_hidden_state",
            data=sample.cotend_hidden_state,
            compression="gzip",
        )


def load_hdf5(path: str | Path) -> AlpamayoSample:
    """HDF5 파일에서 AlpamayoSample을 로드.

    카메라 이미지는 h5에 저장되지 않으므로 camera_images/camera_ids/frame_timestamps_us는
    None으로 반환됩니다.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        return AlpamayoSample(
            clip_id=str(f.attrs["clip_id"]),
            t0_us=int(f.attrs["t0_us"]),
            gt_future_xyz=f["gt/future_xyz"][:],
            gt_future_rot=f["gt/future_rot"][:],
            gt_future_ego_states=f["gt/future_ego_states"][:],
            future_timestamps_us=f["gt/future_timestamps_us"][:],
            gt_future_xyz_global=f["gt/future_xyz_global"][:],
            gt_future_quat_global=f["gt/future_quat_global"][:],
            gt_future_vel=f["gt/future_vel"][:],
            gt_future_acc=f["gt/future_acc"][:],
            gt_future_curv=f["gt/future_curv"][:],
            ego_history_xyz=f["input/ego_history_xyz"][:],
            ego_history_rot=f["input/ego_history_rot"][:],
            ego_history_ego_states=f["input/ego_history_ego_states"][:],
            history_timestamps_us=f["input/history_timestamps_us"][:],
            ego_history_xyz_global=f["input/ego_history_xyz_global"][:],
            ego_history_quat_global=f["input/ego_history_quat_global"][:],
            ego_history_vel=f["input/ego_history_vel"][:],
            ego_history_acc=f["input/ego_history_acc"][:],
            ego_history_curv=f["input/ego_history_curv"][:],
            cot_text=str(f["output"].attrs["cot_text"]),
            meta_action_text=str(f["output"].attrs["meta_action_text"]),
            pred_xyz=f["output/pred_xyz"][:],
            pred_rot=f["output/pred_rot"][:],
            cotend_hidden_state=f["output/cotend_hidden_state"][:],
            camera_images=None,
            camera_ids=None,
            frame_timestamps_us=None,
        )


# ------------------------------------------------------------------
# 시각화 (카메라 이미지가 메모리에 있는 동안 수집 직후 호출)
# ------------------------------------------------------------------

def _visualize_sample(sample: AlpamayoSample, output_path: Path) -> None:
    """AlpamayoSample의 카메라 이미지 + BEV 궤적을 PNG로 저장.

    카메라 이미지는 h5에 저장하지 않으므로, 수집 직후 이 함수로 시각화를 먼저 저장합니다.
    visualize_sample.py의 visualize()와 동일한 레이아웃입니다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    camera_images = sample.camera_images   # (N_cam, N_frm, 3, H, W)
    camera_ids = sample.camera_ids
    ego_hist_xyz = sample.ego_history_xyz  # (16, 3)
    gt_xyz = sample.gt_future_xyz          # (64, 3)
    pred_xyz = sample.pred_xyz             # (64, 3)
    cotend_norm = float(np.linalg.norm(sample.cotend_hidden_state))

    def _rot90cc(xy: np.ndarray) -> np.ndarray:
        """BEV 시각화용 90도 반시계 회전 (차량 전방 = 화면 위쪽)."""
        return np.stack([-xy[1], xy[0]], axis=0)

    N_cam, N_frm = camera_images.shape[:2]
    img_h, img_w = camera_images.shape[3], camera_images.shape[4]
    cell_w = 4.0
    fig_w = max(N_frm * cell_w, 12)
    img_rows_h = N_cam * cell_w * (img_h / img_w)
    bev_h = 5.0
    fig_h = img_rows_h + bev_h + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"clip: {sample.clip_id}  |  t₀ = {sample.t0_us / 1e6:.2f}s  |  "
        f"cotend‖h‖ = {cotend_norm:.2f}",
        fontsize=9, y=1.0,
    )

    gs_root = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[img_rows_h, bev_h],
        hspace=0.35,
    )

    # 카메라 이미지 그리드
    gs_img = gridspec.GridSpecFromSubplotSpec(
        N_cam, N_frm, subplot_spec=gs_root[0],
        wspace=0.01, hspace=0.05,
    )
    for cam_i, cam_id in enumerate(camera_ids):
        for t_i in range(N_frm):
            ax = fig.add_subplot(gs_img[cam_i, t_i])
            ax.imshow(camera_images[cam_i, t_i].transpose(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            if t_i == 0:
                short = (cam_id.replace("camera_", "")
                         .replace("_120fov", "").replace("_30fov", ""))
                ax.set_ylabel(short, fontsize=7, rotation=90, va="center", labelpad=3)
            if cam_i == 0:
                dt_s = (t_i - (N_frm - 1)) * 0.1
                is_cur = (t_i == N_frm - 1)
                ax.set_title(
                    f"t={dt_s:+.1f}s" + ("  ★" if is_cur else ""),
                    fontsize=7, color="red" if is_cur else "dimgray", pad=2,
                )
            for sp in ax.spines.values():
                if t_i == N_frm - 1:
                    sp.set_edgecolor("red"); sp.set_linewidth(2); sp.set_visible(True)
                else:
                    sp.set_visible(False)

    # BEV
    ax_bev = fig.add_subplot(gs_root[1])
    hist_rot = _rot90cc(ego_hist_xyz[:, :2].T)
    gt_rot   = _rot90cc(gt_xyz[:, :2].T)
    pred_rot = _rot90cc(pred_xyz[:, :2].T)

    ax_bev.plot(*hist_rot, "b--o", ms=3, lw=1.5, alpha=0.8, label="History (1.6s)")
    ax_bev.plot(0, 0, "k*", ms=12, zorder=6, label="t=0 (current)")
    ax_bev.plot(*pred_rot, "g-o", ms=3, lw=2, label="Predicted (6.4s)")
    ax_bev.plot(*gt_rot, "r--", lw=2, alpha=0.85, label="GT (6.4s)")

    ade = float(np.linalg.norm(pred_xyz[:, :2] - gt_xyz[:, :2], axis=1).mean())
    ax_bev.set_title(
        f'CoC: "{sample.cot_text}"\n'
        f'meta: {sample.meta_action_text or "N/A"}  |  ADE = {ade:.3f} m',
        fontsize=8,
    )
    ax_bev.set_xlabel("Lateral (m)  ← right | left →", fontsize=8)
    ax_bev.set_ylabel("Longitudinal (m)  forward ↑", fontsize=8)
    ax_bev.legend(fontsize=7, loc="best")
    ax_bev.axis("equal")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_bev.axvline(0, color="k", lw=0.5, alpha=0.3)

    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
