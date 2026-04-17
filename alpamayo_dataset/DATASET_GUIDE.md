# Alpamayo → MPC 가중치 학습용 데이터셋 수집 가이드

## 1. 목적과 전체 흐름

이 데이터셋은 **Alpamayo(VLA) hidden state → MPC 비용함수 가중치** 매핑을 학습하기 위해 수집합니다.

```
PhysicalAI AV 데이터셋 (실제 주행 클립)
        │
        ▼
[Alpamayo 추론]
   멀티카메라 이미지 + ego 히스토리 입력
        │
        ├─── VLM 단계: Chain-of-Causation 추론 텍스트 생성
        │              "보행자가 튀어나오는 상황 → 감속 판단"
        │              └→ lm_head 직전의 hidden state = cotend (4096-dim)
        │                  → 이 벡터에 MLP를 붙여 MPC 가중치 w1~w6 예측
        │
        └─── Diffusion 단계: 미래 64스텝 궤적 생성 (6.4초 @ 10 Hz)
        │
        ▼
HDF5 파일로 저장 (샘플당 1개)
  ├── gt/       실제 미래 궤적 + ego 상태
  ├── input/    카메라 이미지 + ego 히스토리
  └── output/   CoC 텍스트, 예측 궤적, cotend hidden state
```

---

## 2. 사전 준비

### 2-1. Docker 컨테이너 실행

```bash
# 호스트에서 실행 (컨테이너 없으면 먼저 빌드)
docker build -t alpamayo-r1 /mnt/alpamayo

docker run \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  -v /mnt/alpamayo:/workspace \
  --name alpamayo-r1 \
  -it alpamayo-r1

# 재접속할 때
docker start alpamayo-r1 && docker attach alpamayo-r1
```

### 2-2. HuggingFace 토큰 등록 (필수)

두 리소스 모두 **접근 승인 요청** 후 사용 가능합니다.
**컨테이너를 새로 만들 때마다 반드시 실행해야 합니다.**

```bash
# 컨테이너 내부에서 실행
huggingface-cli login
# 토큰 입력 후 엔터 (토큰: https://huggingface.co/settings/tokens)
```

필요한 접근 권한:
- [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) — 모델 가중치 (~22 GB)
- [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) — 주행 데이터셋

> **왜 필요한가?**
> `PhysicalAIAVDatasetInterface`는 초기화 시 `features.csv`, `clip_index.parquet` 등 소형 메타데이터 파일(~수십 MB)을
> HuggingFace에서 가져옵니다. 이 파일들이 HF 캐시(`/workspace/hf_cache`)에 없으면 HF API 인증이 필요합니다.
> 로컬 데이터셋 디렉토리(`/workspace/alpamayo_dataset/data/nvidia_test_scenario`)의 파일은
> HF 캐시 형식이 아니라 `is_file_cached()` 가 인식하지 못합니다 (구조적 한계).
> 인증 후 첫 실행 시 메타데이터가 캐시에 등록되어 이후 실행은 빠릅니다.

---

## 3. 파일 구조

```
/mnt/alpamayo/
├── collect_dataset.py                         ← 수집 CLI 진입점 (여기서 시작)
│
├── alpamayo_dataset/
│   ├── DATASET_GUIDE.md                       ← 이 파일
│   └── data/
│       └── collected/                         ← 수집된 .h5 파일들이 쌓이는 곳
│
└── alpamayo-R1/src/alpamayo_r1/
    ├── dataset/
    │   ├── schema.py        AlpamayoSample 데이터 구조 정의
    │   ├── collector.py     수집 로직 + HDF5 저장
    │   └── torch_dataset.py 학습용 PyTorch Dataset
    │
    └── models/
        └── alpamayo_r1.py   모델 코드 (hidden state 추출 hook 포함)
```

---

## 4. 데이터 스키마 (HDF5 파일 내부 구조)

파일명: `{clip_id}__{t0_us}.h5`

### A축 — GT (실제 미래 궤적)

| 키 | shape | dtype | 설명 |
|----|-------|-------|------|
| `gt/future_xyz` | (64, 3) | float32 | t0 기준 로컬 좌표계에서의 실제 미래 위치 (x, y, z) |
| `gt/future_rot` | (64, 3, 3) | float32 | 실제 미래 회전 행렬 |
| `gt/future_ego_states` | (64, 5) | float32 | 파생 ego 상태: **[x, y, speed, yaw, lon_accel]** |

`ego_states` 열 설명:
- `x, y`: 로컬 좌표 (m)
- `speed`: 전진 속도 크기 (m/s)
- `yaw`: 차량 방위각 (rad), `atan2(R[1,0], R[0,0])`
- `lon_accel`: 종방향 가속도 (m/s²)

### B축 — 입력 (Alpamayo에 들어간 데이터)

| 키 | shape | dtype | 설명 |
|----|-------|-------|------|
| `input/camera_images` | (4, 4, 3, H, W) | uint8 | 4카메라 × 4프레임, 채널 우선 |
| `input/camera_ids` | (4,) | bytes | 카메라 이름 (UTF-8) |
| `input/frame_timestamps_us` | (4, 4) | int64 | 프레임별 절대 타임스탬프 (μs) |
| `input/ego_history_xyz` | (16, 3) | float32 | 1.6초 히스토리 위치 (로컬 좌표계) |
| `input/ego_history_rot` | (16, 3, 3) | float32 | 1.6초 히스토리 회전 행렬 |
| `input/ego_history_ego_states` | (16, 5) | float32 | 히스토리 ego 상태 [x, y, speed, yaw, lon_accel] |
| `input/history_timestamps_us` | (16,) | int64 | 히스토리 타임스탬프 (μs) |

카메라 4개: `camera_cross_left_120fov`, `camera_front_wide_120fov`,
`camera_cross_right_120fov`, `camera_front_tele_30fov`

### C축 — 모델 출력

| 키 | shape / type | 설명 |
|----|-------------|------|
| `output/cotend_hidden_state` | (4096,) float32 | **핵심**: CoC 끝 시점 VLM hidden state → MPC 가중치 예측 입력 |
| `output/pred_xyz` | (64, 3) float32 | Alpamayo가 예측한 미래 궤적 위치 |
| `output/pred_rot` | (64, 3, 3) float32 | Alpamayo가 예측한 미래 궤적 회전 |
| `output attrs/cot_text` | str | Chain-of-Causation 추론 텍스트 |
| `output attrs/meta_action_text` | str | 메타 액션 설명 |

---

## 5. 수집 실행 방법

### 5-1. clip_id와 t0_us 구하기

`clip_id`는 PhysicalAI AV 데이터셋의 클립 식별자입니다.
`t0_us`는 해당 클립에서 **샘플링 기준 시각** (마이크로초)입니다.

```python
# 컨테이너 내부 Python에서 클립 목록 탐색
# collect_dataset.py와 동일한 _LocalFirstAVDI 방식으로 로컬 데이터셋 접근
import pathlib, pandas as pd

DATASET_DIR = "/workspace/alpamayo_dataset/data/nvidia_test_scenario"

# clip_index 직접 로드 (AVDI 없이)
clip_index = pd.read_parquet(f"{DATASET_DIR}/clip_index.parquet")
print(f"총 클립 수: {len(clip_index)}")
print(clip_index.head())

# 유효한 클립만 필터링
valid_clips = clip_index[clip_index["clip_is_valid"]].index.tolist()
print(f"유효 클립 수: {len(valid_clips)}")

# 특정 클립의 t0_us 범위: 클립 길이에 따라 다름
# 일반적으로 2_000_000 ~ 14_000_000 us (2초 ~ 14초) 범위에서 샘플링
# t0_us 간격은 최소 1_000_000 us (1초) 권장 (겹치는 구간 최소화)
```

테스트용 기본 값:
- `clip_id`: `030c760c-ae38-49aa-9ad8-f5650a545d26`
- `t0_us`: `5100000` (클립 시작 후 5.1초 시점)

> **제약**: `t0_us > 1.6초 × 1_000_000 = 1_600_000` (히스토리 범위 확보 필요)

### 5-1-1. CSV 생성 (`make_clips_csv.py`)

`/workspace/alpamayo_dataset/make_clips_csv.py`를 사용합니다.
클립 단위로 묶인 순서로 CSV를 생성합니다 (clip_A 전체 → clip_B 전체 순).

#### t0_us 개념

`t0_us`는 **클립 내 상대 타임스탬프**입니다 (클립 시작 = 0).

| 항목 | 값 |
|------|----|
| 클립 길이 | 20,000,000 μs (20초) |
| 히스토리 필요 | 1,600,000 μs (1.6초) |
| 미래 필요 | 6,400,000 μs (6.4초) |
| **유효 범위** | **1,600,000 ~ 13,600,000 μs** |

#### 기본 실행

```bash
# 기본값: 200 클립 × 6 시점 = 1,200 샘플, /workspace/clips.csv 생성
python /workspace/alpamayo_dataset/make_clips_csv.py
```

#### split 기준

NVIDIA가 데이터셋 배포 시 미리 지정한 파티션입니다. 분류 기준은 공개되지 않았습니다.

| split | 클립 수 | 비율 | 권장 용도 |
|-------|---------|------|-----------|
| train | 153,625 | 50% | MLP 학습용 |
| val   |  90,928 | 30% | 검증·하이퍼파라미터 튜닝 |
| test  |  61,599 | 20% | 최종 평가 (학습 중 사용 금지) |
| all   | 306,152 | 100% | 탐색·시각화 목적 |

#### 옵션

| 옵션 | 기본값 | 설명 및 설정 가능한 값 |
|------|--------|----------------------|
| `--max-clips N` | 200 | 최대 클립 수. `0` = 전체 사용 |
| `--split` | all | `train` / `val` / `test` / `all` |
| `--country NAME` | 전체 | 국가 이름 (대소문자 일치). 선택 가능한 값: `"United States"` (155,360) `"Germany"` (43,900) `"France"` (10,364) `"Italy"` (8,658) `"Sweden"` (7,330) `"Spain"` (6,459) `"Portugal"` (6,101) `"Greece"` (5,885) `"Austria"` (5,451) `"Finland"` (5,176) 외 15개국 |
| `--hour-min H` | 없음 | 수집 시간대 하한. 범위: `0`~`23` |
| `--hour-max H` | 없음 | 수집 시간대 상한. 범위: `0`~`23` |
| `--platform CLASS` | 전체 | `"hyperion_8"` (87,564개) 또는 `"hyperion_8.1"` (218,588개) |
| `--t0-start US` | 1,600,000 | 첫 번째 t0_us. 범위: `1,600,000`~`13,600,000` |
| `--t0-end US` | 13,600,000 | 마지막 t0_us 상한. 범위: `1,600,000`~`13,600,000` |
| `--t0-step US` | 1,000,000 | t0_us 간격 (마이크로초). 예: `1,000,000` = 1초 간격 |
| `--skip-existing DIR` | 없음 | 이미 수집된 .h5 디렉토리 → 해당 샘플 제외 |
| `--output FILE` | `/workspace/alpamayo_dataset/data/collected` | 컨테이너 내부, 출력 CSV 경로 |

#### 사용 예

```bash
# 낮 시간대(7~19시) 미국 클립 500개, 1초 간격
python /workspace/alpamayo_dataset/make_clips_csv.py \
    --max-clips 500 \
    --country "United States" \
    --hour-min 7 --hour-max 19 \
    --t0-step 1000000

# train split만, 전체 클립
python /workspace/alpamayo_dataset/make_clips_csv.py \
    --split train \
    --max-clips 0

# 이미 수집된 샘플 제외하고 추가 수집
python /workspace/alpamayo_dataset/make_clips_csv.py \
    --skip-existing /workspace/alpamayo_dataset/data/collected \
    --max-clips 0
```

```bash
# 생성된 CSV로 일괄 수집
python /workspace/collect_dataset.py \
    --csv /workspace/clips.csv \
    --output-dir /workspace/alpamayo_dataset/data/collected
```

### 5-2. 단일 샘플 수집

```bash
# 컨테이너 내부에서 실행 (로컬 가중치/데이터셋 자동 사용)
python /workspace/collect_dataset.py \
    --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
    --t0-us 5100000 \
    --output-dir /workspace/alpamayo_dataset/data/collected
```

### 5-3. 여러 샘플 일괄 수집 (CSV)

```bash
# clips.csv 형식 (헤더 필수):
# clip_id,t0_us
# 030c760c-ae38-49aa-9ad8-f5650a545d26,5100000
# 030c760c-ae38-49aa-9ad8-f5650a545d26,7200000

python /workspace/collect_dataset.py \
    --csv /workspace/clips.csv \
    --output-dir /workspace/alpamayo_dataset/data/collected
```

### 5-4. 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model-dir DIR` | `/workspace/alpamayo_dataset/data/alpamayo_weights` | 로컬 모델 가중치 경로 |
| `--dataset-dir DIR` | `/workspace/alpamayo_dataset/data/nvidia_test_scenario` | 로컬 데이터셋 경로 |
| `--overwrite` | False | 이미 존재하는 .h5 파일 덮어쓰기 |
| `--num-traj-samples N` | 1 | 궤적 샘플 수 (늘리면 VRAM 증가, cotend hs는 1 권장) |
| `--max-gen-len N` | 256 | CoC 최대 생성 토큰 수 |
| `--seed N` | 42 | CUDA 랜덤 시드 |

---

## 6. 시각화

수집된 샘플을 PNG로 시각화합니다. 카메라 이미지(4대 × 4프레임) + BEV 궤적 + CoC 텍스트를 한 장에 표시합니다.

```bash
# 컨테이너 내부에서 실행
python /workspace/alpamayo_dataset/visualize_sample.py \
    /workspace/alpamayo_dataset/data/collected/030c760c-ae38-49aa-9ad8-f5650a545d26__5100000.h5

# 저장 경로 직접 지정
python /workspace/alpamayo_dataset/visualize_sample.py \
    /workspace/alpamayo_dataset/data/collected/030c760c-ae38-49aa-9ad8-f5650a545d26__5100000.h5 \
    --output /workspace/alpamayo_dataset/data/collected/viz.png
```

출력 PNG에 포함되는 정보:
- 카메라 영상: 4대 × 4프레임 (현재 프레임 빨간 테두리)
- BEV 궤적: 파란 점선(히스토리) + 초록 실선(예측) + 빨간 점선(GT) + ADE
- CoC 텍스트: 모델이 추론한 상황 설명

---

## 7. 수집된 데이터 확인 (스크립트)

```python
import h5py
import numpy as np

path = "/workspace/alpamayo_dataset/data/collected/030c760c-ae38-49aa-9ad8-f5650a545d26__5100000.h5"

with h5py.File(path, "r") as f:
    print("clip_id:", f.attrs["clip_id"])
    print("t0_us  :", f.attrs["t0_us"])

    # cotend hidden state — 이 벡터가 MPC 가중치 예측에 쓰임
    hs = f["output/cotend_hidden_state"][:]
    print("cotend hidden state shape:", hs.shape)   # (4096,)
    print("cotend hidden state norm :", np.linalg.norm(hs))

    # CoC 텍스트
    print("\nCoC 텍스트:")
    print(f["output"].attrs["cot_text"])

    # GT 미래 궤적 (처음 5 스텝)
    print("\nGT future xyz (첫 5스텝):")
    print(f["gt/future_xyz"][:5])
```
```python
python - <<'EOF'
import h5py
import numpy as np

path = "/workspace/alpamayo_dataset/data/collected/030c760c-ae38-49aa-9ad8-f5650a545d26__5100000.h5"

with h5py.File(path, "r") as f:
    print("clip_id:", f.attrs["clip_id"])
    print("t0_us  :", f.attrs["t0_us"])

    # cotend hidden state — MPC 가중치 예측 입력 벡터
    hs = f["output/cotend_hidden_state"][:]
    print("\ncotend hidden state shape:", hs.shape)   # (4096,)
    print("cotend hidden state norm :", np.linalg.norm(hs))  # 0이면 hook 실패

    # CoC 텍스트
    print("\nCoC 텍스트:")
    print(f["output"].attrs["cot_text"])

    # GT 미래 궤적 처음 5스텝
    print("\nGT future xyz (첫 5스텝):")
    print(f["gt/future_xyz"][:5])

    # 전체 키 구조 확인
    print("\n전체 키:")
    f.visititems(lambda name, obj: print(" ", name))
EOF
```

---

## 7. 학습에 사용하기 (PyTorch Dataset)

```python
from alpamayo_r1.dataset.torch_dataset import AlpamayoMPCDataset
from torch.utils.data import DataLoader

dataset = AlpamayoMPCDataset(
    "/workspace/alpamayo_dataset/data/collected",
    load_images=False,  # 이미지가 필요 없으면 False로 로딩 속도 향상
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # MLP 입력: cotend hidden state
    hs  = batch["cotend_hidden_state"]       # (B, 4096)

    # 학습 레이블 도출에 사용 가능한 GT 데이터
    gt  = batch["gt_future_ego_states"]      # (B, 64, 5) [x, y, speed, yaw, lon_accel]
    xyz = batch["gt_future_xyz"]             # (B, 64, 3)

    # 모델 예측 궤적 (비교/검증용)
    pred = batch["pred_xyz"]                 # (B, 64, 3)

    # 메타데이터
    cot  = batch["cot_text"]                 # list of B strings

    break
```

---

## 8. cotend hidden state 기술 설명

### 무엇인가?

Alpamayo는 두 단계로 작동합니다:

```
1단계 (VLM): 이미지 + 히스토리 → Chain-of-Causation 추론 → <traj_future_start> 토큰
                                                                        ↑
                                                          이 토큰이 생성될 때
                                                    lm_head에 입력되는 hidden state
                                                    = cotend (4096-dim)

2단계 (Expert + Diffusion): <traj_future_start> 이후 → 미래 궤적 64개 웨이포인트
```

`cotend`는 "이 상황에서 왜 이 행동을 선택했는지"가 압축된 벡터입니다.
**보행자 돌발 상황**과 **고속도로 순항 상황**에서 서로 다른 벡터가 생성됩니다.

### 어떻게 추출하나?

```python
# alpamayo_r1.py 내부 코드 (참고용)
_cotend_ref = {}

def _cotend_pre_hook(_module, args):
    # args[0]: lm_head 입력 텐서 (batch, 1, hidden_size=4096)
    # 디코드 스텝마다 덮어씌워지므로 마지막 값 = <traj_future_start> 위치
    _cotend_ref["state"] = args[0][:, -1, :].detach()

hook = self.vlm.lm_head.register_forward_pre_hook(_cotend_pre_hook)
# ... vlm.generate() 실행 ...
hook.remove()
# _cotend_ref["state"] → extra["cotend_hidden_state"]에 저장
```

`output_hidden_states=True`를 사용하지 않는 이유:
generate() 전체에서 누적 저장 → **256 스텝 × 29 레이어 × bfloat16 ≈ 320 MB** VRAM 추가 사용.
pre-hook 방식은 마지막 텐서 1개만 유지 → **오버헤드 거의 없음**.

---

## 9. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `cotend_hidden_state` 값이 모두 0 | hook 미발화 | `num_traj_samples=1` 확인, Flash Attention 오류 시 `config.attn_implementation="sdpa"` 설정 |
| CUDA OOM | VRAM 부족 | `--num-traj-samples 1` (기본값), 다른 프로세스 종료 확인 |
| `401 Unauthorized` | HF 토큰 없음 | `huggingface-cli login` 재실행 |
| `IndexError: list index out of range` (AVDI 초기화 시) | HF 인증 없음 → `get_paths_info()` 빈 목록 반환 | `huggingface-cli login` 실행 후 재시도 (컨테이너 재시작 후 항상 필요) |
| HDF5 파일이 빈 채로 저장됨 | 추론 중 예외 발생 후 파일 생성 | `--overwrite` 플래그로 재수집 |
| `t0_us` 값 오류 | 히스토리 범위 부족 | `t0_us > 1_600_000` (1.6초 이후 값) |
