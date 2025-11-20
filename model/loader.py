from pathlib import Path
import tensorflow as tf
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = BASE_DIR / "model" / "battery_lstm_base.h5"


def get_user_model_path(user_id: int) -> Path:
    return BASE_DIR / "model" / f"user_{user_id}_model.h5"


@lru_cache()
def load_base_model():
    print(f"[MODEL] Loading base model at {BASE_MODEL_PATH}")
    return tf.keras.models.load_model(BASE_MODEL_PATH)


def load_user_model(user_id: int):
    path = get_user_model_path(user_id)
    if path.exists():
        print(f"[MODEL] Load personalized model: {path}")
        return tf.keras.models.load_model(path)

    print("[MODEL] Personalized model not found. Using base model.")
    return load_base_model()
