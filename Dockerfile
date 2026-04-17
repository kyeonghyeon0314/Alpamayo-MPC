FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── CUDA 환경 변수 ──────────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# flash-attn 빌드 설정
# MAX_JOBS: Docker 빌드 중 병렬 컴파일 수 제한 (OOM 방지)
# TORCH_CUDA_ARCH_LIST: RTX 3090 = Ampere (sm_86)
ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="8.6"

# ── 시스템 패키지 ────────────────────────────────────────────────────────────
# Ubuntu 22.04 기본 Python은 3.10 → deadsnakes PPA로 3.12 설치
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    software-properties-common \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.12 \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12을 기본 python으로 등록
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# pip 설치 (python3.12 대상)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# ── uv 설치 ─────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# UV_SYSTEM_PYTHON=1 → venv 없이 시스템 Python site-packages에 직접 설치
# Docker는 이미 OS 수준으로 격리되어 있으므로 Python venv는 불필요
ENV UV_SYSTEM_PYTHON=1

# ── 서드파티 의존성 설치 ──────────────────────────────────────────────────────
# uv sync 대신 uv pip install --system 사용 → venv 없이 시스템에 직접 설치
# Docker 컨테이너 자체가 격리 환경이므로 venv 불필요
#
# alpamayo-R1/uv.lock 기준으로 아래 패키지들이 설치됨:
#   torch==2.8.0, torchvision
#   transformers==4.57.1, accelerate
#   einops, hydra-core, hydra-colorlog
#   av (PyAV, 번들 FFmpeg 포함), pillow, pandas, pyarrow
#   scipy, numpy, tqdm, huggingface-hub
#   physical_ai_av (PyPI 버전 → 런타임에 PYTHONPATH 로컬 버전이 우선함)
#   flash-attn (torch 설치 후 --no-build-isolation으로 별도 설치)
#   jupyterlab, ipykernel, matplotlib, mediapy (노트북 환경)
WORKDIR /build
COPY alpamayo-R1/pyproject.toml alpamayo-R1/uv.lock ./

# 1단계: uv.lock에서 requirements 추출 (flash-attn 제외 — torch 먼저 필요)
RUN uv export --frozen --no-emit-project --no-hashes \
        --no-dev \
        | grep -v "flash-attn" > /tmp/requirements.txt

# 2단계: torch 포함 일반 의존성 시스템에 직접 설치
RUN uv pip install --system -r /tmp/requirements.txt

# 3단계: flash-attn 설치 (torch가 먼저 설치된 후 no-build-isolation으로 빌드)
RUN uv pip install --system flash-attn --no-build-isolation

# 4단계: 노트북 개발 환경 설치 (jupyterlab, ipykernel, matplotlib, mediapy)
RUN uv pip install --system jupyterlab ipykernel matplotlib mediapy ipywidgets

# ── 런타임 환경 설정 ─────────────────────────────────────────────────────────
# 마운트: docker run -v /mnt/alpamayo:/workspace ...
#
# PYTHONPATH에 두 로컬 패키지 src 경로 추가:
#   - alpamayo_r1   : /workspace/alpamayo-R1/src/alpamayo_r1
#   - physical_ai_av: /workspace/physical_ai_av/src/physical_ai_av
#                     (PyPI 설치 버전보다 PYTHONPATH가 sys.path 우선순위 높음)
ENV PYTHONPATH=/workspace/alpamayo-R1/src:/workspace/physical_ai_av/src

# HuggingFace 모델 캐시를 마운트 볼륨 안에 저장 (재실행 시 재다운로드 불필요)
# 저장 위치: /mnt/alpamayo/hf_cache/ (약 22GB + Qwen 프로세서)
ENV HF_HOME=/workspace/hf_cache

WORKDIR /workspace/alpamayo-R1

# ── 사용법 ────────────────────────────────────────────────────────────────────
# 빌드:
#   docker build -t alpamayo-r1 .
#
# X11 접근 허용 (호스트에서 한 번만 실행):
#   xhost +local:docker
#
# 실행:
#   docker run \
#     --gpus all \                              # GPU 전체 사용
#     --ipc=host \                             # PyTorch 공유 메모리 (DataLoader 필수)
#     --ulimit memlock=-1 \                    # CUDA 메모리 잠금 제한 해제
#     --ulimit stack=67108864 \               # CUDA 스택 크기 (64MB)
#     --shm-size=16g \                        # 공유 메모리 (기본 64MB → Bus error 방지)
#     -v /mnt/alpamayo:/workspace \     # 소스·데이터·모델 캐시 마운트
#     -v /tmp/.X11-unix:/tmp/.X11-unix \      # X11 소켓 (plt.show() 팝업)
#     -e DISPLAY=$DISPLAY \                   # X11 디스플레이 (plt.show() 팝업)
#     -p 8888:8888 \                          # Jupyter 노트북 포트
#     --name alpamayo-r1 \                    # 컨테이너 이름 (재접속 시 편의)
#     -it alpamayo-r1
#실제 실행 명령어 

## docker run --gpus all --shm-size=16g -v /mnt/alpamayo:/workspace -v /tmp.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 8888:8888 --name alpamayo-r1 -it alpamayo-r1
#
# 추론 (컨테이너 안에서):
#   python src/alpamayo_r1/test_inference.py
#
# Jupyter 노트북 (컨테이너 안에서):
## 토큰 고정
#jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --LabApp.token='alpamayo'

## 또는 토큰 없애기 (로컬 개발용으로 편함)
# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --LabApp.token=''
#   → VS Code에서 "Existing Jupyter Server"로 출력된 URL 연결
#
# 중지 후 재접속:
#   docker start alpamayo-r1 && docker attach alpamayo-r1
CMD ["/bin/bash"]
