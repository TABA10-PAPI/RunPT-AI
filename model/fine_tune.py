import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from schemas.battery import SequenceItem  # 타입 재사용 (편의용)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "users"
BASE_MODEL_PATH = BASE_DIR / "model" / "battery_lstm_base.h5"

WINDOW_SIZE = 7  # LSTM 입력 타임스텝 길이


def get_user_model_path(user_id: int) -> Path:
    return BASE_DIR / "model" / f"user_{user_id}_model.h5"


def _load_flat_history(user_id: int):
    """
    data/users/{userId}_history.json 을 읽어서
    [SequenceItem, SequenceItem, ...] 형태로 평탄화해서 반환.
    """
    path = DATA_DIR / f"{user_id}_history.json"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    flat = []
    for session in history:
        for item in session:
            flat.append(SequenceItem(**item))

    return flat


def finetune_user_model(user_id: int, epochs: int = 1):
    """
    사용자별 전체 히스토리를 기반으로
    - 슬라이딩 윈도우(길이 WINDOW_SIZE)들을 만들고
    - base 모델 또는 기존 user 모델에서 시작해서
    - 짧게라도 추가 학습을 수행한 뒤
    - model/user_{id}_model.h5 로 저장.

    *데이터가 부족하면 그냥 조용히 return 한다.*
    """
    flat = _load_flat_history(user_id)

    # 러닝 데이터가 WINDOW_SIZE 개 미만이면 파인튜닝하지 않음
    if len(flat) < WINDOW_SIZE:
        print(f"[FINETUNE] Not enough data for user {user_id} (len={len(flat)})")
        return

    X_list = []
    y_list = []

    # 슬라이딩 윈도우 생성
    # 예: flat 길이가 10, WINDOW_SIZE=7이면
    # i=0 -> 0~6, i=1 -> 1~7, i=2 -> 2~8, i=3 -> 3~9 총 4개 윈도우
    for i in range(len(flat) - WINDOW_SIZE + 1):
        window = flat[i : i + WINDOW_SIZE]

        x = [
            [
                s.hr,
                s.hrv,
                s.pace,
                s.sleep_hours,
                s.distance_km,
                s.calories,
            ]
            for s in window
        ]
        X_list.append(x)

        # TODO: 실제 레이블(타겟)을 정의하면 여기서 넣으면 됨.
        # 지금은 구조만 잡는 단계라 임시로 50.0 을 사용.
        y_list.append(50.0)

    if not X_list:
        print(f"[FINETUNE] No valid window for user {user_id}")
        return

    X = np.array(X_list, dtype="float32")  # (num_samples, WINDOW_SIZE, 6)
    y = np.array(y_list, dtype="float32")  # (num_samples,)

    user_model_path = get_user_model_path(user_id)

    # 기존 사용자 모델이 있으면 그걸 이어서 학습, 없으면 base 모델에서 시작
    if user_model_path.exists():
        model = tf.keras.models.load_model(user_model_path)
        print(f"[FINETUNE] Continue training user model: {user_model_path}")
    else:
        model = tf.keras.models.load_model(BASE_MODEL_PATH)
        print(f"[FINETUNE] Start finetuning from base model for user {user_id}")

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, batch_size=4, epochs=epochs, verbose=0)

    model.save(user_model_path)
    print(f"[FINETUNE] Saved personalized model to: {user_model_path}")
