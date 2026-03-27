"""
CotendMLP 모델 정의 및 공유 상수 / 로드 유틸리티

train.py 와 test.py 가 이 모듈을 import 해 동일한 아키텍처를 공유합니다.

모델 구조 (Plain MLP):
    4096 → LayerNorm → Linear(1024) → ReLU → Dropout
          → Linear(256)  → ReLU → Dropout
          → Linear(4)    (unbounded log-scale 출력)

출력 공간:
    log-scale MPC 가중치 4개 [long_pos, heading, steer_rate, accel_rate]
    lat_pos_weight 는 1.0 고정이므로 예측 제외
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════
# 공유 상수
# ══════════════════════════════════════════════════════

# mpc_weights 저장 순서: [long, lat(고정), heading, steer_rate, accel_rate]
PREDICT_IDX = [0, 2, 3, 4]
W_NAMES     = ["long_pos", "heading", "steer_rate", "accel_rate"]
N_OUT       = len(PREDICT_IDX)  # 4

# log-scale 범위 (label_mpc_weights.py _THETA_MIN/MAX 와 동일)
#              long   head   steer_r  accel_r
LOG_MIN = np.array(np.log([0.1,  0.1,  1.0,  0.2 ]), dtype=np.float32)
LOG_MAX = np.array(np.log([10.,  10.,  50.,  10. ]), dtype=np.float32)


# ══════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════

class CotendMLP(nn.Module):
    """
    cotend hidden state (4096,) → log-scale MPC weights (4,)

    구조:
        LayerNorm(4096)
        Linear(1024) → ReLU → Dropout
        Linear(256)  → ReLU → Dropout
        Linear(4)    (unbounded — MSE loss가 범위 학습 담당)
    """

    def __init__(
        self,
        input_dim:   int   = 4096,
        hidden_dims: tuple = (1024, 256),
        output_dim:  int   = N_OUT,
        dropout:     float = 0.1,
    ):
        super().__init__()
        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """log-scale theta (4,) 반환 — 비제약 선형 출력."""
        return self.net(x)

    def predict_weights(self, cotend: torch.Tensor) -> torch.Tensor:
        """추론용: forward → exp → 실제 MPC 가중치 (no_grad)."""
        with torch.no_grad():
            return torch.exp(self.forward(cotend))


# ══════════════════════════════════════════════════════
# 모델 로드 유틸리티
# ══════════════════════════════════════════════════════

def load_mlp(model_dir: Path, device: torch.device) -> tuple["CotendMLP", dict]:
    """model_full.pt 또는 best_model.pt + config.json 으로 MLP 복원.

    Returns:
        (model, config_dict)
    """
    model_dir  = Path(model_dir)
    full_pt    = model_dir / "model_full.pt"
    config_path = model_dir / "config.json"

    if full_pt.exists():
        ckpt    = torch.load(full_pt, map_location=device, weights_only=True)
        hidden  = tuple(ckpt.get("model_hidden", [1024, 256]))
        dropout = ckpt.get("dropout", 0.1)
        model   = CotendMLP(hidden_dims=hidden, dropout=dropout).to(device)
        # strict=False: 구버전 sigmoid 모델(log_min/log_max buffer 포함)과 호환
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        cfg = {"predict_idx": ckpt.get("predict_idx", PREDICT_IDX)}
    else:
        cfg = {}
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
        hidden  = tuple(cfg.get("hidden", [1024, 256]))
        dropout = cfg.get("dropout", 0.1)
        model   = CotendMLP(hidden_dims=hidden, dropout=dropout).to(device)
        model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True),
            strict=False,
        )

    model.eval()
    src = full_pt if full_pt.exists() else model_dir / "best_model.pt"
    print(f"[MLP] 로드 완료: {src}")
    return model, cfg
