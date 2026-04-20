#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""수집된 HDF5 샘플 시각화 스크립트.

단일 파일:
    python /workspace/alpamayo_dataset/visualize_sample.py \\
        /workspace/alpamayo_dataset/data/prepare/train/030c760c-...h5

    # 출력 경로 지정
    python /workspace/alpamayo_dataset/visualize_sample.py \\
        /workspace/alpamayo_dataset/data/prepare/train/030c760c-...h5 \\
        --output /tmp/viz.png

폴더 전체:
    python /workspace/alpamayo_dataset/visualize_sample.py \\
        --dir /workspace/alpamayo_dataset/data/prepare/train

    # 출력 디렉토리 별도 지정
    python /workspace/alpamayo_dataset/visualize_sample.py \\
        --dir /workspace/alpamayo_dataset/data/prepare/train \\
        --output-dir /workspace/alpamayo_dataset/data/viz

    # 이미 생성된 PNG도 덮어쓰기
    python /workspace/alpamayo_dataset/visualize_sample.py \\
        --dir /workspace/alpamayo_dataset/data/prepare/train \\
        --overwrite
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# 카메라 인덱스 → 표시 이름
_CAM_NAMES = {
    0: "Cross-Left",
    1: "Front-Wide",
    2: "Cross-Right",
    3: "Rear-Left",
    4: "Rear-Tele",
    5: "Rear-Right",
    6: "Front-Tele",
}


def _rotate_90cc(xy: np.ndarray) -> np.ndarray:
    """BEV 시각화용 90도 반시계 회전 (차량 전방 = 화면 위쪽)."""
    return np.stack([-xy[1], xy[0]], axis=0)


def visualize(h5_path: str, output: str | None = None) -> None:
    with h5py.File(h5_path, "r") as f:
        clip_id = str(f.attrs["clip_id"])
        t0_us = int(f.attrs["t0_us"])
        cot_text = str(f["output"].attrs["cot_text"])
        meta_action = str(f["output"].attrs["meta_action_text"])

        # A. GT 궤적
        gt_xyz = f["gt/future_xyz"][:]                    # (64, 3)

        # B. 입력
        camera_images = f["input/camera_images"][:]       # (N_cam, 4, 3, H, W)  uint8
        camera_ids_raw = f["input/camera_ids"][:]         # (N_cam,) bytes
        camera_ids = [s.decode("utf-8") for s in camera_ids_raw]
        ego_hist_xyz = f["input/ego_history_xyz"][:]      # (16, 3)

        # C. 예측 궤적
        pred_xyz = f["output/pred_xyz"][:]                # (64, 3)

        # cotend hidden state 통계
        hs = f["output/cotend_hidden_state"][:]
        hs_norm = np.linalg.norm(hs)

    N_cam, N_frm = camera_images.shape[:2]

    # ── 그리드 레이아웃 계산 ─────────────────────────────────────────────────
    # 이미지 섹션 (N_cam 행 × N_frm 열) + BEV 섹션 + 속도 섹션
    img_h, img_w = camera_images.shape[3], camera_images.shape[4]   # H, W
    cell_w = 4.0
    fig_w = max(N_frm * cell_w, 12)
    img_rows_h = N_cam * cell_w * (img_h / img_w)
    bev_h = 5.0
    fig_h = img_rows_h + bev_h + 0.8   # 0.8: 하단 텍스트 여유

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"clip: {clip_id}  |  t₀ = {t0_us / 1e6:.2f}s  |  "
        f"cotend‖h‖ = {hs_norm:.2f}",
        fontsize=9, y=1.0,
    )

    gs_root = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[img_rows_h, bev_h],
        hspace=0.35,
    )

    # ── 카메라 이미지 그리드 ─────────────────────────────────────────────────
    gs_img = gridspec.GridSpecFromSubplotSpec(
        N_cam, N_frm, subplot_spec=gs_root[0],
        wspace=0.01, hspace=0.05,
    )
    for cam_i, cam_id in enumerate(camera_ids):
        # camera_images: (N_cam, N_frm, 3, H, W) → (H, W, 3) 변환
        for t_i in range(N_frm):
            ax = fig.add_subplot(gs_img[cam_i, t_i])
            frame_hwc = camera_images[cam_i, t_i].transpose(1, 2, 0)   # CHW→HWC
            ax.imshow(frame_hwc)
            ax.set_xticks([])
            ax.set_yticks([])

            if t_i == 0:
                short_name = cam_id.replace("camera_", "").replace("_120fov", "").replace("_30fov", "")
                ax.set_ylabel(short_name, fontsize=7, rotation=90, va="center", labelpad=3)

            if cam_i == 0:
                dt_s = (t_i - (N_frm - 1)) * 0.1
                is_cur = (t_i == N_frm - 1)
                ax.set_title(
                    f"t={dt_s:+.1f}s" + ("  ★" if is_cur else ""),
                    fontsize=7,
                    color="red" if is_cur else "dimgray",
                    pad=2,
                )

            # 현재 프레임 테두리 강조
            for sp in ax.spines.values():
                if t_i == N_frm - 1:
                    sp.set_edgecolor("red")
                    sp.set_linewidth(2)
                    sp.set_visible(True)
                else:
                    sp.set_visible(False)

    # ── BEV (Bird's-Eye View) ────────────────────────────────────────────────
    ax_bev = fig.add_subplot(gs_root[1])

    hist_xy_rot = _rotate_90cc(ego_hist_xyz[:, :2].T)
    gt_xy_rot = _rotate_90cc(gt_xyz[:, :2].T)
    pred_xy_rot = _rotate_90cc(pred_xyz[:, :2].T)

    dt = 0.1
    ax_bev.plot(*hist_xy_rot, "b--o", ms=3, lw=1.5, alpha=0.8,
                label=f"History ({len(ego_hist_xyz) * dt:.1f}s)")
    ax_bev.plot(0, 0, "k*", ms=12, zorder=6, label="t=0 (current)")
    ax_bev.plot(*pred_xy_rot, "g-o", ms=3, lw=2,
                label=f"Predicted ({len(pred_xyz) * dt:.1f}s)")
    ax_bev.plot(*gt_xy_rot, "r--", lw=2, alpha=0.85,
                label=f"Ground Truth ({len(gt_xyz) * dt:.1f}s)")

    # ADE 계산
    ade = np.linalg.norm(pred_xyz[:, :2] - gt_xyz[:, :2], axis=1).mean()
    ax_bev.set_title(
        f'CoC: "{cot_text}"\n'
        f'meta: {meta_action or "N/A"}  |  ADE = {ade:.3f} m',
        fontsize=8,
    )
    ax_bev.set_xlabel("Lateral (m)  ← right | left →", fontsize=8)
    ax_bev.set_ylabel("Longitudinal (m)  forward ↑", fontsize=8)
    ax_bev.legend(fontsize=7, loc="best")
    ax_bev.axis("equal")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_bev.axvline(0, color="k", lw=0.5, alpha=0.3)

    # ── 저장 / 표시 ──────────────────────────────────────────────────────────
    if output is None:
        output = str(Path(h5_path).with_suffix(".png"))

    plt.savefig(output, dpi=100, bbox_inches="tight")
    print(f"저장됨: {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="수집된 HDF5 샘플 시각화")

    # 단일 파일 또는 폴더 중 하나 선택
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("h5_path", nargs="?", default=None, help="단일 .h5 파일 경로")
    src.add_argument("--dir", metavar="DIR", help="폴더 내 모든 .h5 파일을 일괄 시각화")

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 PNG 경로 (단일 파일 모드 전용. 기본: h5와 같은 위치에 .png 저장)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="폴더 모드 전용: PNG 저장 디렉토리 (기본: h5 파일과 같은 폴더)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 PNG도 덮어쓰기 (기본: 건너뜀)",
    )

    args = parser.parse_args()

    # ── 단일 파일 모드 ────────────────────────────────────────────────────────
    if args.h5_path:
        visualize(args.h5_path, args.output)
        return

    # ── 폴더 전체 모드 ────────────────────────────────────────────────────────
    collected_dir = Path(args.dir)
    h5_files = sorted(collected_dir.glob("*.h5"))
    if not h5_files:
        print(f".h5 파일이 없습니다: {collected_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    total = len(h5_files)
    skipped = 0
    done = 0
    failed = 0

    for i, h5_path in enumerate(h5_files, 1):
        out_png = (output_dir or h5_path.parent) / h5_path.with_suffix(".png").name
        if out_png.exists() and not args.overwrite:
            skipped += 1
            print(f"[{i}/{total}] 건너뜀 (이미 존재): {out_png.name}")
            continue
        try:
            visualize(str(h5_path), str(out_png))
            done += 1
            print(f"[{i}/{total}] 완료: {out_png.name}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{total}] 실패: {h5_path.name}  ({e})")

    print(f"\n완료 {done}개 | 건너뜀 {skipped}개 | 실패 {failed}개  (전체 {total}개)")


if __name__ == "__main__":
    main()
