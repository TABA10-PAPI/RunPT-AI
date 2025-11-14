# data 디렉토리

- `sample_train.csv`  
  - 예시 학습 데이터 파일입니다.
  - 실제 서비스에서는 Health Connect / NRC 데이터에서 추출한 값으로 교체하고,
    LSTM 학습용으로 시계열 윈도우를 구성해 사용해야 합니다.

컬럼 설명:

- `hr`: 평균 심박수 (bpm)
- `hrv`: HRV (예: RMSSD 등)
- `pace`: 러닝 페이스 (분/km)
- `sleep_hours`: 수면 시간 (시간)
- `distance_km`: 일일 러닝 거리 (km)
- `calories`: 활동 칼로리 (kcal)
- `battery`: 라벨 값, 러닝 배터리 점수 (0~100)
