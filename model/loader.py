import tensorflow as tf
from pathlib import Path
import json
from functools import lru_cache
from config.settings import BASE_MODEL_PATH, MODEL_DIR

def get_user_model_path(user_id: int) -> Path:
    return MODEL_DIR / f"user_{user_id}_model.keras"

@lru_cache()
def load_base_model():
    print(f"[MODEL] Loading base model: {BASE_MODEL_PATH}")
    return tf.keras.models.load_model(BASE_MODEL_PATH)

def load_user_model(user_id: int):
    custom_path = get_user_model_path(user_id)
    if custom_path.exists():
        print(f"[MODEL] Loading user model: {custom_path}")
        return tf.keras.models.load_model(custom_path)
    
    print("[MODEL] Personalized model not found. Using base model.")
    return load_base_model()
