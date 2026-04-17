from physical_ai_av import PhysicalAIAVDatasetInterface
import os
import pandas as pd

# 1. 인터페이스 초기화
output_dir = "./data/nvidia_test_scenario"
ds = PhysicalAIAVDatasetInterface(
    token="hf_", # 이미 로그인했다면 True, 아니면 토큰 문자열 입력
    local_dir=output_dir
)

# 2. 최신 메타데이터 및 인덱스 다운로드 (최신화)
print(f"🔄 '{output_dir}'의 메타데이터를 최신 상태로 업데이트 중...")
ds.download_metadata()

# 3. 데이터 분석 및 통계 계산
print("📊 데이터를 분석하여 보고서를 생성 중입니다...")

dc = ds.metadata.get('data_collection')
fp = ds.metadata.get('feature_presence')

if dc is not None and fp is not None:
    total_clips = len(fp)

    # NuRec(AlpaSim) 호환성 분석 (경고 해결: 인덱스 정렬)
    camera_cols = [c for c in fp.columns if 'camera' in c]
    has_all_cameras = fp[camera_cols].all(axis=1)
    has_lidar = fp['lidar_point_cloud'] if 'lidar_point_cloud' in fp.columns else pd.Series([False]*total_clips, index=fp.index)
    
    # 두 시리즈의 인덱스를 맞춘 후 계산
    nurec_mask = has_all_cameras & has_lidar
    nurec_count = nurec_mask.sum()

    # 분포 및 가용성 계산
    country_dist = dc['country'].value_counts()
    hour_dist = dc['hour_of_day'].value_counts().sort_index()
    sensor_availability = (fp.mean() * 100).sort_values(ascending=False)

    # 주간/야간 구분 (6~18시: Day, 그 외: Night)
    dc['lighting'] = dc['hour_of_day'].apply(lambda x: 'Day' if 6 <= x <= 18 else 'Night')
    lighting_dist = dc['lighting'].value_counts()

    # 4. DATASET_OVERVIEW.md 보고서 생성
    report_path = os.path.join(output_dir, "DATASET_OVERVIEW.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 NVIDIA PhysicalAI AV Dataset Overview\n\n")
        f.write(f"이 보고서는 메타데이터를 기반으로 자동 생성되었습니다. (최종 업데이트: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')})\n\n")

        f.write("## 1. 📈 핵심 요약 (Summary)\n")
        f.write(f"- **전체 시나리오(Clip) 수:** {total_clips:,} 개\n")
        f.write(f"- **NuRec(AlpaSim) 재구성 가능 후보:** {nurec_count:,} 개\n")
        f.write(f"  - **근거:** `feature_presence.parquet`에서 7개 카메라 및 LiDAR가 모두 `True`인 클립\n\n")

        f.write("## 2. 🌍 지역별 시나리오 분포 (Country Distribution)\n")
        f.write("> **근거:** `data_collection.parquet`의 `country` 컬럼 집계\n\n")
        f.write("| 국가 | 클립 수 | 비중 (%) |\n")
        f.write("| :--- | ---: | ---: |\n")
        for country, count in country_dist.items():
            f.write(f"| {country} | {count:,} | {count/total_clips*100:.1f}% |\n")
        f.write("\n")

        f.write("## 3. 💡 주행 환경 구분 (Situational Classification)\n")
        f.write("현재 메타데이터에 공식 'weather' 태그는 없으나, 시간대를 통해 조도 환경을 유추할 수 있습니다.\n\n")
        
        f.write("### 🔆 조도 환경 (Lighting Conditions)\n")
        f.write("> **근거:** `hour_of_day` 기준 (주간: 06~18시, 야간: 19~05시)\n\n")
        for light, count in lighting_dist.items():
            f.write(f"- **{light}:** {count:,} 개 ({count/total_clips*100:.1f}%)\n")
        f.write("\n")

        f.write("### 🕒 시간대별 세부 통계\n")
        f.write("| 시간(Hour) | 클립 수 | 데이터 밀도 |\n")
        f.write("| :--- | ---: | :--- |\n")
        max_hour_val = hour_dist.max()
        for hour, count in hour_dist.items():
            bar = "█" * int(count/max_hour_val * 20)
            f.write(f"| {hour:02d}시 | {count:,} | {bar} |\n")
        f.write("\n")

        f.write("## 4. 🛰️ 센서 및 기능 가용성 (Sensor Availability)\n")
        f.write("> **근거:** `feature_presence.parquet` 내 피처별 존재 확률\n\n")
        f.write("| Feature | 제공 확률 (%) | 설명 |\n")
        f.write("| :--- | ---: | :--- |\n")
        for feat, val in sensor_availability.items():
            f.write(f"| {feat} | {val:.1f}% | |\n")
        f.write("\n")

        f.write("## 5. 📂 참조 파일\n")
        f.write("- `clip_index.parquet` / `metadata/data_collection.parquet` / `metadata/feature_presence.parquet` / `features.csv` \n")

    print(f"📝 보고서가 생성되었습니다: {os.path.abspath(report_path)}")
else:
    print("⚠️ 메타데이터 분석 중 오류가 발생했습니다.")

# 5. 기존 인덱스 요약 파일 저장
clip_list_path = os.path.join(output_dir, "available_clips_summary.csv")
ds.clip_index.to_csv(clip_list_path)

print(f"\n✨ 업데이트 완료! '{output_dir}' 폴더에서 리포트를 확인하세요.")