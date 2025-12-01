# model/loader.py
import tensorflow as tf
from pathlib import Path
import json
from functools import lru_cache
from config.settings import BASE_MODEL_PATH, MODEL_DIR, USER_HISTORY_DIR


def get_user_model_path(user_id: int) -> Path:
    return MODEL_DIR / f"user_{user_id}_model.keras"


def _get_user_record_count(user_id: int) -> int:
    """
    사용자 히스토리의 총 기록 개수를 반환.
    fine-tune 조건(>=20)을 판단하기 위해 사용됨.
    """
    path = USER_HISTORY_DIR / f"{user_id}_history.json"
    if not path.exists():
        return 0
    
    try:
        with path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return 0

    # 다차원 구조 → 평탄화
    flat = sum((session for session in history), [])
    return len(flat)


@lru_cache()
def load_base_model():
    print(f"[MODEL] Loading base model: {BASE_MODEL_PATH}")
    return tf.keras.models.load_model(BASE_MODEL_PATH)


def load_user_model(user_id: int):
    """
    - 사용자 기록이 20개 미만이면 base model 사용
    - 개인 모델이 존재하더라도 로딩 오류나 손상된 모델이면 base로 fallback
    - 정상 모델만 load해서 반환
    """

    # 1) 사용자 기록 개수 확인
    record_count = _get_user_record_count(user_id)
    if record_count < 20:
        print(f"[MODEL] User {user_id} has {record_count} records → Using base model.")
        return load_base_model()

    # 2) 개인 모델 경로 체크
    custom_path = get_user_model_path(user_id)
    if not custom_path.exists():
        print(f"[MODEL] Personal model not found for user {user_id}. Using base model.")
        return load_base_model()

    # 3) 모델을 안전하게 로드 (오류 시 base로 fallback)
    try:
        print(f"[MODEL] Loading user model: {custom_path}")
        return tf.keras.models.load_model(custom_path)
    except Exception as e:
        print(f"[MODEL] Failed to load user model for user {user_id}: {e}")
        print("[MODEL] Falling back to base model.")
        return load_base_model()
