from pathlib import Path
import tensorflow as tf
from functools import lru_cache

# 프로젝트 루트 기준
BASE_DIR = Path(__file__).resolve().parent.parent

# 🔥 Keras3 공식 포맷 (.keras)
BASE_MODEL_PATH = BASE_DIR / "model" / "battery_lstm.keras"


def get_user_model_path(user_id: int) -> Path:
    return BASE_DIR / "model" / f"user_{user_id}_model.keras"


@lru_cache()
def load_base_model():
    print(f"[MODEL] Loading base model at {BASE_MODEL_PATH}")
    return tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False  # 🔥 Keras3에서 더 안정적
    )


def load_user_model(user_id: int):
    user_path = get_user_model_path(user_id)

    if user_path.exists():
        print(f"[MODEL] Loading personalized model: {user_path}")
        return tf.keras.models.load_model(
            user_path,
            compile=False  # 🔥 Keras3 설치 환경에서는 필요
        )

    print("[MODEL] Personalized model not found. Using base model.")
    return load_base_model()
