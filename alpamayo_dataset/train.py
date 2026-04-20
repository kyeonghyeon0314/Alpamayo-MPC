#!/usr/bin/env python3
"""
STEP 3: cotend → MPC weights MLP 학습

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력:  cotend hidden state (4096,)  — VLA lm_head 직전 hidden state
출력:  MPC 비용함수 가중치 5개 (log-scale)
         [long_pos, heading, accel, steer_rate, accel_rate]
         lat_pos_weight 는 1.0 고정이므로 예측 제외

모델: model.py CotendMLP 참조
  4096 → 1024 → 256 → 5  (unbounded log-scale 출력)

학습:
  - Loss:       균등 MSE (log-scale)
  - 옵티마이저: AdamW (weight_decay=1e-4)
  - 스케줄러:   CosineAnnealingLR (T_max=epochs, eta_min=1e-5)
  - 조기종료:   val loss 기준 patience=30 epoch

데이터 디렉토리 구조 (split_dataset.py 로 사전 생성):
    split-dir/
        train/  *.h5  (70%)
        val/    *.h5  (20%)
        test/   *.h5  (10%)

출력 파일:
    <out>/best_model.pt        val loss 최소 체크포인트 (가중치만)
    <out>/model_full.pt        가중치 + 아키텍처 메타데이터 (추론 재현용)
    <out>/history.json         epoch별 train/val loss
    <out>/config.json          하이퍼파라미터 + 최종 test loss
    <out>/test_results.npz     테스트셋 pred / target (분석용)
    <out>/training_curves.png  train/val loss 곡선
    <out>/train.log            학습 로그 (타임스탬프 포함)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실행 예시:
    python alpamayo_dataset/train.py \\
        --split-dir alpamayo_dataset/data/split \\
        --out       alpamayo_dataset/mlp_out \\
        --balance --lat-thresh 1.0
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CotendMLP, PREDICT_IDX, W_NAMES, N_OUT, LOG_MIN, LOG_MAX


# ══════════════════════════════════════════════════════
# Dataset 로딩 워커 (모듈 top-level 필수 — ProcessPoolExecutor 피클링)
# ══════════════════════════════════════════════════════

def _load_one(path_str: str):
    """h5 파일 하나를 읽어 (cotend, theta, max_lat) 반환. 실패 시 None."""
    try:
        with h5py.File(path_str, "r") as f:
            cotend  = f["output/cotend_hidden_state"][:].astype(np.float32)
            w       = f["labels/mpc_weights"][:]
            lat_y   = f["gt/future_xyz"][:20, 1]
        w_pred  = w[PREDICT_IDX].astype(np.float32)
        theta   = np.log(np.clip(w_pred, 1e-6, None))
        max_lat = float(np.max(np.abs(lat_y)))
        return cotend, theta, max_lat
    except Exception:
        return None


# ══════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════

class CotendDataset(Dataset):
    """
    split 디렉토리(train/ or val/ or test/)의 h5 파일을 로드.
    반환: (cotend float32 (4096,), theta float32 (5,))
    theta = log(mpc_weights[PREDICT_IDX])  — log-scale
    """

    def __init__(self, split_dir: Path, name: str = "", workers: int = 8,
                 rebuild_cache: bool = False, data_dir: Path | None = None):
        label      = f"[{name}]" if name else ""
        cache_path = split_dir / "_mlp_cache.npz"

        # ── 캐시 히트: npz 1개 읽기 (~2초) ────────────────
        if cache_path.exists() and not rebuild_cache:
            data = np.load(cache_path)
            self.cotend  = torch.from_numpy(np.array(data["cotend"]))
            self.theta   = torch.from_numpy(np.array(data["theta"]))
            self.max_lat = torch.from_numpy(np.array(data["max_lat"]))
            print(f"  {label:<8} {len(self.cotend):,}개 (캐시)")
            return

        # ── 캐시 미스: h5 파일 순차 로드 후 캐시 저장 ────
        # IPC pickle 비용(~1GB) > 병렬화 이득 → 순차 로드
        h5_files = sorted(split_dir.glob("*.h5"))

        # 심볼릭 링크 없음 → manifest + data_dir 폴백 (Windows symlink 불가 환경)
        if not h5_files and data_dir is not None:
            manifest_path = split_dir.parent / "dataset_manifest.json"
            split_name    = split_dir.name          # "train" / "val" / "test"
            if manifest_path.exists():
                with open(manifest_path) as _f:
                    _m = json.load(_f)
                clip_ids = set(_m.get(f"clips_{split_name}", []))
                if clip_ids:
                    h5_files = sorted(
                        p for p in Path(data_dir).glob("*.h5")
                        if p.stem.split("__")[0] in clip_ids
                    )
                    print(f"  {label:<8} manifest 기반 {len(h5_files):,}개 ({data_dir})")

        if not h5_files:
            raise RuntimeError(
                f"{split_dir} 에 .h5 파일이 없습니다.\n"
                "  prepare_dataset.py 실행 후 data/prepare/ 구조를 확인하세요."
            )

        cotends, thetas, max_lats = [], [], []
        skipped = 0
        for path_str in (str(p) for p in h5_files):
            r = _load_one(path_str)
            if r is None:
                skipped += 1
            else:
                cotends.append(r[0])
                thetas.append(r[1])
                max_lats.append(r[2])

        if not cotends:
            raise RuntimeError(f"{split_dir}: 로드 가능한 샘플이 없습니다.")

        cotend_arr  = np.stack(cotends)
        theta_arr   = np.stack(thetas)
        max_lat_arr = np.array(max_lats, dtype=np.float32)

        np.savez(cache_path, cotend=cotend_arr, theta=theta_arr, max_lat=max_lat_arr)

        self.cotend  = torch.from_numpy(cotend_arr)
        self.theta   = torch.from_numpy(theta_arr)
        self.max_lat = torch.from_numpy(max_lat_arr)

        print(f"  {label:<8} {len(self.cotend):,}개 로드"
              + (f"  (스킵 {skipped}개)" if skipped else "")
              + f"  → 캐시 저장: {cache_path.name}")

    def __len__(self):
        return len(self.cotend)

    def __getitem__(self, idx):
        return self.cotend[idx], self.theta[idx]


# ══════════════════════════════════════════════════════
# 학습 루프
# ══════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    all_pred, all_tgt = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total += nn.functional.mse_loss(pred, y).item() * len(x)
        all_pred.append(pred.cpu())
        all_tgt.append(y.cpu())
    preds     = torch.cat(all_pred)
    targets   = torch.cat(all_tgt)
    per_w_mse = ((preds - targets) ** 2).mean(dim=0)
    return total / len(loader.dataset), preds, targets, per_w_mse


# ══════════════════════════════════════════════════════
# 로깅 및 시각화 유틸리티
# ══════════════════════════════════════════════════════

def setup_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_mlp")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_training_plot(history, best_epoch, out_dir, early_stop_epoch=None):
    n          = len(history["train_loss"])
    epochs     = list(range(1, n + 1))
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="steelblue",  lw=1.5, alpha=0.9)
    ax.plot(epochs, val_loss,   label="Val Loss",   color="darkorange", lw=1.5, alpha=0.9)
    ax.axvline(best_epoch, color="green", ls="--", lw=1.0,
               label=f"Best (epoch {best_epoch})")
    ax.scatter([best_epoch], [val_loss[best_epoch - 1]],
               marker="*", s=200, color="green", zorder=5)
    if early_stop_epoch is not None and early_stop_epoch != best_epoch:
        ax.axvline(early_stop_epoch, color="red", ls=":", lw=1.0,
                   label=f"Early Stop (epoch {early_stop_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log-scale)")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="cotend → MPC weights MLP 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--split-dir",    required=True,
                        help="split_dataset.py 로 생성된 디렉토리 (split_manifest.json 포함)")
    parser.add_argument("--data-dir",     default=None,
                        help="원본 .h5 디렉토리. 심볼릭 링크 없을 때 manifest와 함께 사용")
    parser.add_argument("--out",          default="mlp_out",
                        help="출력 디렉토리 (default: mlp_out)")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch-size",   type=int,   default=512,
                        help="배치 크기 (default: 512, RTX 5060 8GB 충분)")
    parser.add_argument("--lr",           type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--hidden",       type=int,   nargs="+",
                        help="히든 레이어 크기 (default: 1024 256)")
    parser.add_argument("--patience",     type=int,   default=30)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--compile",      action="store_true",
                        help="torch.compile (Linux + PyTorch 2.0+, Windows 불가)")
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--load-workers", type=int,   default=8)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="캐시 무시하고 h5에서 재로드")
    parser.add_argument("--balance",      action="store_true",
                        help="lateral/longitudinal 균형 샘플링 (WeightedRandomSampler)")
    parser.add_argument("--lat-thresh",   type=float, default=1.0,
                        help="lateral 판별 임계값: max(|GT_y[:20]|) [m] (default: 1.0)")
    args = parser.parse_args()

    if args.hidden is None:
        args.hidden = [1024, 256]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.set_float32_matmul_precision("high")

    out_dir   = Path(args.out)
    split_dir = Path(args.split_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)

    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} 없음. split_dataset.py를 먼저 실행하세요."
        )

    data_dir = Path(args.data_dir) if args.data_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 데이터 로드 ────────────────────────────────────
    print("\n데이터 로드 중...")
    t0 = time.perf_counter()
    train_ds = CotendDataset(split_dir / "train", name="train",
                             workers=args.load_workers, rebuild_cache=args.rebuild_cache,
                             data_dir=data_dir)
    val_ds   = CotendDataset(split_dir / "val",   name="val",
                             workers=args.load_workers, rebuild_cache=args.rebuild_cache,
                             data_dir=data_dir)
    test_ds  = CotendDataset(split_dir / "test",  name="test",
                             workers=args.load_workers, rebuild_cache=args.rebuild_cache,
                             data_dir=data_dir)
    print(f"  로드 완료: {time.perf_counter()-t0:.1f}s")

    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    n_total = n_train + n_val + n_test
    print(f"\n  train: {n_train:,}개  ({n_train/n_total*100:.1f}%)")
    print(f"  val:   {n_val:,}개  ({n_val/n_total*100:.1f}%)")
    print(f"  test:  {n_test:,}개  ({n_test/n_total*100:.1f}%)")

    if args.balance:
        is_lat  = (train_ds.max_lat > args.lat_thresh).numpy()
        n_lat   = int(is_lat.sum())
        n_lon   = len(is_lat) - n_lat
        weights = np.where(is_lat, n_lon / max(n_lat, 1), 1.0).astype(np.float32)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights))
        print(f"\n  균형 샘플링: lateral={n_lat:,}개 ({n_lat/len(is_lat)*100:.1f}%)"
              f"  longitudinal={n_lon:,}개  임계값={args.lat_thresh}m")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=sampler, num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True,
        )
    val_loader  = DataLoader(val_ds,  batch_size=512, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ── 모델 ──────────────────────────────────────────
    model = CotendMLP(hidden_dims=tuple(args.hidden), dropout=args.dropout).to(device)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile 활성화")
        except Exception as e:
            print(f"[WARN] torch.compile 실패, eager 모드로 진행: {e}")

    n_params = sum(p.numel() for p in model.parameters())
    arch_str = f"4096 → {' → '.join(str(h) for h in args.hidden)} → {N_OUT}"
    print(f"\n[MLP] {arch_str}  (unbounded log-scale 출력)")
    print(f"파라미터 수: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── 학습 ──────────────────────────────────────────
    logger.info("학습 시작 (epochs=%d, batch=%d, lr=%s, wd=%s, patience=%d)",
                args.epochs, args.batch_size, args.lr, args.weight_decay, args.patience)
    logger.info("Loss: 균등 MSE (log-scale)")
    logger.info("%6s  %9s  %9s  %9s  %9s", "Epoch", "Train", "Val", "LR", "Best")
    logger.info("─" * 55)

    history          = {"train_loss": [], "val_loss": []}
    best_val         = float("inf")
    best_epoch       = 1
    no_improve       = 0
    early_stop_epoch = None
    t_start          = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, _, _, per_w = eval_epoch(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            mark = "★" if improved else ""
            logger.info("%6d  %9.5f  %9.5f  %9.2e  %9.5f %s",
                        epoch, train_loss, val_loss,
                        scheduler.get_last_lr()[0], best_val, mark)

        if no_improve >= args.patience:
            early_stop_epoch = epoch
            logger.info("EarlyStopping: %depoch 개선 없음 → 조기 종료 (epoch %d)",
                        args.patience, epoch)
            break

    elapsed = time.perf_counter() - t_start
    logger.info("학습 완료: %.1fs  best_val_loss=%.5f  best_epoch=%d",
                elapsed, best_val, best_epoch)

    # ── 테스트 평가 ────────────────────────────────────
    model.load_state_dict(
        torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loss, preds, targets, per_w = eval_epoch(model, test_loader, device)

    logger.info("─" * 55)
    logger.info("Test MSE  (log-scale): %.5f", test_loss)
    logger.info("Test RMSE (log-scale): %.5f", test_loss**0.5)
    logger.info("%-15s  %10s  %12s  %20s", "가중치", "RMSE(log)", "RMSE(상대%)", "예측범위")
    logger.info("─" * 62)
    for i, name in enumerate(W_NAMES):
        mse_i   = float(per_w[i])
        rmse_i  = mse_i ** 0.5
        rel_pct = (np.exp(rmse_i) - 1) * 100
        p_min   = float(torch.exp(preds[:, i]).min())
        p_max   = float(torch.exp(preds[:, i]).max())
        logger.info("%-15s  %10.4f  %11.1f%%  [%.3f, %.3f]",
                    name, rmse_i, rel_pct, p_min, p_max)

    # ── 결과 저장 ──────────────────────────────────────
    np.savez(out_dir / "test_results.npz", preds=preds.numpy(), targets=targets.numpy())
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                **vars(args),
                "n_train": n_train, "n_val": n_val, "n_test": n_test,
                "best_val_loss": best_val, "test_loss": test_loss,
                "predict_idx": PREDICT_IDX, "weight_names": W_NAMES,
            },
            f, indent=2,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_hidden":     args.hidden,
            "dropout":          args.dropout,
            "predict_idx":      PREDICT_IDX,
            "log_min":          LOG_MIN.tolist(),
            "log_max":          LOG_MAX.tolist(),
        },
        out_dir / "model_full.pt",
    )

    save_training_plot(history, best_epoch, out_dir, early_stop_epoch)

    logger.info("저장 완료:")
    logger.info("  %s/best_model.pt       ← 가중치만", out_dir)
    logger.info("  %s/model_full.pt       ← 가중치 + 아키텍처 (추론용)", out_dir)
    logger.info("  %s/history.json        ← train/val loss 곡선", out_dir)
    logger.info("  %s/test_results.npz    ← 테스트 pred/target", out_dir)
    logger.info("  %s/config.json         ← 하이퍼파라미터", out_dir)
    logger.info("  %s/training_curves.png ← 학습 곡선 플롯", out_dir)
    logger.info("  %s/train.log           ← 학습 로그", out_dir)


if __name__ == "__main__":
    main()
