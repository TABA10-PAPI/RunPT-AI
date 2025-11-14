# model/loader.py
from functools import lru_cache
import tensorflow as tf
from config.settings import MODEL_PATH


@lru_cache()
def get_model():
    """
    LSTM 모델을 한 번만 로딩하고 캐시.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ 모델 로딩 완료: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None
