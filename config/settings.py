from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RUNNING_DIR = DATA_DIR / "running"
PROFILE_DIR = DATA_DIR / "profile"
MODEL_DIR = BASE_DIR / "model"

# 디렉토리 생성
RUNNING_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_PATH = MODEL_DIR / "battery_lstm.keras"
