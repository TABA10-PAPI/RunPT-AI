# model/fine_tune.py
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from schemas.battery import SequenceItem  # 타입 재사용 (편의용)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "users"
BASE_MODEL_PATH = BASE_DIR / "model" / "battery_lstm_base.h5"

WINDOW_SIZE = 7  # LSTM 입력 타임스텝 길이
MIN_RECORDS_TO_TRAIN = 20  # 최소 20개 기록 있어야 파인튜닝 수행


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


def _get_target_from_window(window):
    """
    윈도우(SequenceItem 리스트)에서 마지막 날의 실제 배터리 값을 타깃으로 사용.
    SequenceItem 안에 battery 또는 battery_score 같은 필드를 찾는다.
    """

    last = window[-1]

    # 실제 사용하는 필드명을 탐색
    val = getattr(last, "battery", None)
    if val is None:
        val = getattr(last, "battery_score", None)
    if val is None:
        val = getattr(last, "battery_raw", None)

    if val is None:
        return None

    val = float(val)

    # 0~100 → 0~1 스케일로 바꾸기
    if val > 1.5:
        val /= 100.0

    # 0~1 범위로 제한
    return max(0.0, min(1.0, val))


def finetune_user_model(user_id: int, epochs: int = 1):
    """
    사용자별 전체 히스토리를 기반으로
    - 기록이 20개 미만이면 파인튜닝하지 않는다.
    - 슬라이딩 윈도우(길이 WINDOW_SIZE)를 만든다.
    - base 모델 또는 user 모델에서 이어서 학습한다.
    - model/user_{id}_model.h5 로 저장.

    *타깃 값은 사용자에게 실제로 제공한 'battery' 값이다.*
    """

    flat = _load_flat_history(user_id)
    num_records = len(flat)

    # ✔ 신규 조건: 20개 이상 기록이 있어야 파인튜닝한다
    if num_records < MIN_RECORDS_TO_TRAIN:
        print(f"[FINETUNE] User {user_id}: Not enough records ({num_records}). Need >= {MIN_RECORDS_TO_TRAIN}.")
        return

    X_list = []
    y_list = []

    # ✔ 슬라이딩 윈도우 생성
    for i in range(num_records - WINDOW_SIZE + 1):
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

        # ✔ 마지막 날의 실제 배터리 값을 타깃으로 사용
        y = _get_target_from_window(window)
        if y is None:
            continue

        X_list.append(x)
        y_list.append(y)

    # ✔ 유효 윈도우가 없으면 종료
    if not X_list:
        print(f"[FINETUNE] User {user_id}: No valid training windows (no target data).")
        return

    X = np.array(X_list, dtype="float32")                       # (num_samples, WINDOW_SIZE, 6)
    y = np.array(y_list, dtype="float32").reshape(-1, 1)        # (num_samples, 1)

    user_model_path = get_user_model_path(user_id)

    # ✔ 기존 사용자 모델이 있으면 이어서 학습한다
    if user_model_path.exists():
        print(f"[FINETUNE] User {user_id}: Continue training existing personal model.")
        model = tf.keras.models.load_model(user_model_path)
    else:
        print(f"[FINETUNE] User {user_id}: Start finetuning from base model.")
        model = tf.keras.models.load_model(BASE_MODEL_PATH)

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, batch_size=4, epochs=epochs, verbose=0)

    model.save(user_model_path)
    print(f"[FINETUNE] User {user_id}: Saved personal model → {user_model_path}")
