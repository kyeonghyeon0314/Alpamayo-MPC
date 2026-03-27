import os
from huggingface_hub import snapshot_download

# 모델 ID 및 저장 경로 설정
model_id = "nvidia/Alpamayo-1.5-10B"
local_dir = "./alpamayo_1.5_weights"

print(f"🚀 {model_id} 모델 웨이트 다운로드를 시작합니다 ...")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    token=os.environ.get("HF_TOKEN"),  # huggingface-cli login 또는 환경변수로 설정
    local_dir_use_symlinks=False  # 윈도우에서 우분투로 옮길 계획이라면 False 권장
)

print("다운로드 완료!")