# model/loader.py
import tensorflow as tf
from pathlib import Path
import json
from functools import lru_cache
from config.settings import BASE_MODEL_PATH, MODEL_DIR, RUNNING_DIR


def get_user_model_path(user_id: int) -> Path:
    return MODEL_DIR / f"user_{user_id}_model.keras"


def _get_user_record_count(user_id: int) -> int:
    """
    사용자 러닝 기록 개수를 계산한다.
    fine-tune 조건(>=20)을 판단하는 용도.
    """

    path = RUNNING_DIR / f"user_{user_id}.json"
    if not path.exists():
        return 0

    try:
        with path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return 0

    # history는 list[run_record]
    return len(history)


@lru_cache()
def load_base_model():
    print(f"[MODEL] Loading base model: {BASE_MODEL_PATH}")
    return tf.keras.models.load_model(BASE_MODEL_PATH)


def load_user_model(user_id: int):
    """
    사용자 기록이 20개 이상이면 개인 모델을 사용하고,
    모델이 없거나 로딩 오류가 발생하면 base 모델로 fallback.
    """

    record_count = _get_user_record_count(user_id)

    # 사용자 기록이 부족하면 개인 모델 사용 불가
    if record_count < 20:
        print(f"[MODEL] User {user_id} has only {record_count} records → Using base model.")
        return load_base_model()

    custom_path = get_user_model_path(user_id)

    # 개인 모델 파일 자체가 없으면 base 사용
    if not custom_path.exists():
        print(f"[MODEL] Personal model not found for user {user_id}. Using base model.")
        return load_base_model()

    # 모델 로딩 시도 (오류 시 base fallback)
    try:
        print(f"[MODEL] Loading user model: {custom_path}")
        return tf.keras.models.load_model(custom_path)
    except Exception as e:
        print(f"[MODEL] Failed to load personal model for user {user_id}: {e}")
        print("[MODEL] Falling back to base model.")
        return load_base_model()
