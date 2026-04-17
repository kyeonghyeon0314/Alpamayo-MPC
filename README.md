# Alpamayo R1 — VLA Inference & MPC Evaluation

NVIDIA PhysicalAI AV 데이터셋 위에서 **Alpamayo R1** Vision-Language-Action 모델을 추론하고,
그 결과를 MPC 기반 유사 클로즈드 루프(Pseudo-Closed-Loop) 시뮬레이션으로 평가하는 연구 레포지토리.

---

## 전체 구조

```
alpamayo2/
├── alpamayo-R1/          # Alpamayo R1 VLA 추론 패키지 (alpamayo_r1)
├── physical_ai_av/       # NVIDIA PhysicalAI AV 데이터셋 인터페이스 (physical_ai_av)
├── alpamayo_dataset/     # 데이터 수집 · 라벨링 · MLP 학습 스크립트
├── closed-loop-simul/    # Pseudo-Closed-Loop 시뮬레이션 스크립트
├── alpasim/              # AV 시뮬레이션 플랫폼 (독립 프로젝트)
├── collect_dataset.py    # 데이터 수집 CLI 진입점
└── Dockerfile            # GPU 컨테이너 빌드 (alpamayo-R1 기준)
```

---

## 시스템 요구사항

| 항목 | 스펙 |
|------|------|
| GPU | NVIDIA RTX 3090 이상 (VRAM 24GB) |
| OS | Ubuntu 22.04 / Windows 11 (Docker 권장) |
| Python | 3.12 |
| 패키지 관리 | [uv](https://github.com/astral-sh/uv) |
| 모델 가중치 | [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) (~22GB) |
| 데이터셋 | [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |

> 모델 가중치와 데이터셋 모두 HuggingFace 접근 승인이 필요합니다.
> `huggingface-cli login` 으로 토큰을 등록한 뒤 사용하세요.

---

## 빠른 시작 (Docker)

```bash
# 1. 이미지 빌드
docker build -t alpamayo-r1 .

# 2. 컨테이너 실행
docker run \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=16g \
  -v /mnt/alpamayo:/workspace \
  -p 8888:8888 \
  --name alpamayo-r1 -it alpamayo-r1

# 3. 추론 테스트 (컨테이너 내부)
python src/alpamayo_r1/test_inference.py
```

---

## 전체 파이프라인

```
PhysicalAI AV Dataset
        │
        ▼
 collect_dataset.py          ← 클립 단위 추론 실행
        │
        │  Alpamayo R1 추론
        │    - VLM: CoC 텍스트 생성
        │    - Expert + Diffusion: 64-step 궤적 생성
        │    - VLM hidden state (cotend) 추출
        ▼
 alpamayo_dataset/
  ├── label_mpc_weights.py   ← cotend → MPC 가중치 IOC 라벨링
  ├── train.py               ← MLP 학습 (cotend → weights5)
  ├── eval_clip.py           ← Open-loop 클립 평가
  └── data/collected/*.h5    ← HDF5 데이터 저장
        │
        ▼
 closed-loop-simul/
  ├── build_pcl_dataset.py   ← PCL 평가용 h5 빌드
  └── run_pcl_sim.py         ← Pseudo-Closed-Loop 시뮬레이션 실행
```

---

## 모듈별 설명

### `alpamayo-R1/` — VLA 추론 엔진

Cosmos-Reason 기반 VLM과 Flow Matching Diffusion으로 구성된 2단계 파이프라인.

| 단계 | 모듈 | 설명 |
|------|------|------|
| Stage 1 | `models/alpamayo_r1.py` | VLM이 Chain-of-Causation(CoC) 추론 텍스트 생성 |
| Stage 2 | `diffusion/flow_matching.py` | Expert 모델 + Diffusion으로 64개 웨이포인트(6.4초) 궤적 생성 |
| 출력 | `action_space/utils.py` | 3D 궤적 (xyz, rotation) + CoC 텍스트 |

VLM의 KV 캐시를 Stage 2에서 재사용하여 연산 효율을 높입니다.

---

### `alpamayo_dataset/` — 데이터 수집 및 MLP 학습

**주요 스크립트:**

| 파일 | 역할 |
|------|------|
| `label_mpc_weights.py` | 각 클립에 대해 IOC로 최적 MPC 가중치를 역산 |
| `train.py` | cotend → weights5 를 예측하는 MLP 학습 |
| `eval_clip.py` | Open-loop 기준 MLP / Default MPC 비교 평가 |
| `mpc.py` | Bicycle model + OSQP 기반 MPC 구현 |
| `model.py` | CotendMLP 아키텍처 (4096→512→256→5) |

**MPC 가중치 5개:**

| 가중치 | 의미 |
|--------|------|
| `w_long` | 종방향 추종 |
| `w_lat` | 횡방향 추종 |
| `w_hdg` | 헤딩 추종 |
| `w_steer_r` | 조향 변화율 제약 |
| `w_accel_r` | 가속 변화율 제약 |

---

### `closed-loop-simul/` — Pseudo-Closed-Loop 평가

오픈루프 평가의 한계(매 스텝 GT 리셋)를 극복하기 위한 유사 클로즈드 루프 검증 프레임워크.

**핵심 원리:**

- GT 궤적은 MPC가 **추종할 reference 경로**로만 사용
- 차량의 실제 위치는 제어 명령 → Bicycle model 적분으로 갱신
- 100ms마다 cotend → MLP → 새 MPC 가중치 적용 (Receding Horizon)

```
t=0   GT 위치에서 출발
  ↓
cotend[i] → MLP → weights
  ↓
MPC(현재 시뮬 위치, GT reference) → U[0:N]
  ↓
step_dynamics(x, U[0])            ← 첫 번째 입력만 적용
  ↓
t=1   새 시뮬 위치 (GT ≠ 시뮬)
  ↓
반복 ...
```

**평가 지표:**

| 지표 | 설명 |
|------|------|
| ADE | 전 구간 평균 위치 오차 [m] |
| FDE | 마지막 스텝 오차 [m] (발산 여부 핵심) |
| 발산 기준 | FDE > 10m |

**실행:**

```bash
# PCL 데이터셋 빌드
python closed-loop-simul/build_pcl_dataset.py \
    --clip-id <clip_id> --out closed-loop-simul/pcl_clips

# 시뮬레이션 실행
python closed-loop-simul/run_pcl_sim.py \
    --pcl-clip  closed-loop-simul/pcl_clips/<clip_id>.h5 \
    --model-dir closed-loop-simul/mlp_out \
    --out       closed-loop-simul/pcl_sim_out
```

---

## 환경 변수

```bash
PYTHONPATH=/workspace/alpamayo-R1/src:/workspace/physical_ai_av/src
HF_HOME=/workspace/hf_cache
```

---

## 코드 스타일

- 라인 길이: 100자
- 포맷터: `ruff`
- 커밋: `#<Issue> - <Title>`, `git commit -s` (DCO sign-off 필수)

---

## 라이선스

각 서브패키지의 `LICENSE` 파일을 참조하세요.
