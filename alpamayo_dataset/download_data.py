from physical_ai_av import PhysicalAIAVDatasetInterface
import os

# 1. 인터페이스 초기화
# local_dir을 지정하면 모든 데이터가 해당 폴더에 저장됩니다.
ds = PhysicalAIAVDatasetInterface(
    token=True, # 이미 로그인했다면 True, 아니면 토큰 문자열 입력
    local_dir="./nvidia_test_scenario"
)

# 2. 기본 인덱스 및 메타데이터 다운로드
# 이 과정에서 clip_index.parquet와 features.csv가 자동으로 받아집니다.
ds.download_metadata()

# 3. 다운로드할 대상 클립 ID 선택 (첫 번째 클립 선택)
target_clip_id = ds.clip_index.index[0]
print(f"🎯 선택된 시나리오 클립 ID: {target_clip_id}")

# 4. 해당 클립의 모든 센서 정보(카메라 7개, LiDAR, 레이더, Egomotion 등) 다운로드
# features=None으로 설정하면 해당 클립의 사용 가능한 모든 기능을 가져옵니다.
print("📥 데이터 다운로드 시작 (용량이 크니 네트워크 상태를 확인하세요)...")
ds.download_clip_features(
    clip_id=target_clip_id, 
    features=None,
    max_workers=8 # 병렬 다운로드 속도 향상
)

print(f"✨ 다운로드 완료! 경로: {os.path.abspath('./nvidia_test_scenario')}")