# config/settings.py
from pathlib import Path

# 프로젝트 루트 기준 경로 계산
BASE_DIR = Path(__file__).resolve().parent.parent

# 모델 파일 경로
MODEL_PATH = BASE_DIR / "model" / "battery_lstm.h5"

# 기타 설정 예시
DEBUG = True
