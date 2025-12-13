# model/fine_tune.py
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# Adam 옵티마이저와 EarlyStopping을 명시적으로 임포트
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping

from schemas.battery import SequenceItem  # 타입 재사용 (편의용)
from service.skill_service import get_user_static_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "users"
BASE_MODEL_PATH = BASE_DIR / "model" / "battery_lstm_base.h5" 

WINDOW_SIZE = 7  # LSTM 입력 타임스텝 길이
MIN_RECORDS_TO_TRAIN = 20  # 최소 20개 기록 있어야 파인튜닝 수행

# train_lstm.py와 일치하도록 7차원 특성 사용
FEATURE_DIM = 7 

def get_user_model_path(user_id: int) -> Path:
    # 저장 경로 확장자를 .keras로 변경 (TensorFlow 2.x 표준 권장)
    return BASE_DIR / "model" / f"user_{user_id}_model.keras"


def _load_flat_history(user_id: int) -> List[SequenceItem]:
    """
    data/users/{userId}_history.json 을 읽어서
    [SequenceItem, SequenceItem, ...] 형태로 평탄화해서 반환.
    """
    path = DATA_DIR / f"{user_id}_history.json"
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    flat = []
    for session in history:
        if isinstance(session, list):
            for item in session:
                flat.append(SequenceItem(**item))
        elif isinstance(session, dict):
            flat.append(SequenceItem(**session)) 

    return flat


def _get_target_from_window(window: List[SequenceItem]) -> Any:
    """
    윈도우(SequenceItem 리스트)에서 마지막 날의 실제 배터리 값을 타깃으로 사용.
    """
    last = window[-1]

    val = getattr(last, "battery", None)
    if val is None:
        val = getattr(last, "battery_score", None)
    if val is None:
        val = getattr(last, "battery_raw", None)

    if val is None:
        return None

    try:
        val = float(val)
    except (TypeError, ValueError):
        return None

    # 0~100 → 0~1 스케일로 바꾸기
    if val > 1.5:
        val /= 100.0

    # 0~1 범위로 제한 (Sigmoid 출력을 가정)
    return max(0.0, min(1.0, val))


def finetune_user_model(user_id: int, epochs: int = 5):
    """
    사용자별 전체 히스토리를 기반으로 파인튜닝을 수행합니다.
    """

    flat = _load_flat_history(user_id)
    num_records = len(flat)

    # 20개 미만이면 파인튜닝하지 않는다.
    if num_records < MIN_RECORDS_TO_TRAIN:
        print(
            f"[FINETUNE] User {user_id}: Not enough records "
            f"({num_records}). Need >= {MIN_RECORDS_TO_TRAIN}."
        )
        return

    # 유저 정적 특성 로드 (나이/키/몸무게)
    static = get_user_static_features(user_id)
    age = float(static.get("age", 30))
    height = float(static.get("height", 170))
    weight = float(static.get("weight", 65))

    X_list = []
    y_list = []

    # 슬라이딩 윈도우 생성
    for i in range(num_records - WINDOW_SIZE + 1):
        window = flat[i: i + WINDOW_SIZE]

        # ----------------------------------------------------
        # 수정된 7차원 특성 구성 (train_lstm.py와 차원 일치)
        x = [
            [
                float(s.distance_km),  # distance
                float(s.pace),         # pace_sec (근사)
                float(s.sleep_hours),  # time_sec 대신 sleep_hours 사용 (배터리 지표)
                float(s.hr),           # avg_hr (근사)
                age,                   # age
                height,                # height
                weight,                # weight
            ]
            for s in window
        ]
        # ----------------------------------------------------

        # 마지막 날의 실제 배터리 값을 타깃으로 사용
        y = _get_target_from_window(window)
        if y is None:
            continue

        X_list.append(x)
        y_list.append(y)

    if not X_list:
        print(f"[FINETUNE] User {user_id}: No valid training windows (no target data).")
        return

    X = np.array(X_list, dtype="float32")                       # (num_samples, WINDOW_SIZE, 7)
    y = np.array(y_list, dtype="float32").reshape(-1, 1)        # (num_samples, 1)

    user_model_path = get_user_model_path(user_id)

    # 기존 사용자 모델이 있으면 이어서 학습한다
    if user_model_path.exists():
        print(f"[FINETUNE] User {user_id}: Continue training existing personal model.")
        model = load_model(user_model_path)
    else:
        print(f"[FINETUNE] User {user_id}: Start finetuning from base model.")
        # 베이스 모델 로드
        model = load_model(BASE_MODEL_PATH) 

    # ----------------------------------------------------
    # 파인튜닝을 위한 낮은 학습률 적용
    finetune_optimizer = Adam(learning_rate=1e-5) 
    model.compile(optimizer=finetune_optimizer, loss="mse", metrics=['mae'])

    # 조기 종료 (Early Stopping) 적용
    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=2, 
        restore_best_weights=True,
        verbose=1
    )
    # ----------------------------------------------------
    
    history = model.fit(
        X, 
        y, 
        batch_size=4, 
        epochs=epochs,
        validation_split=0.2, # 20%를 검증에 사용
        callbacks=[early_stop],
        verbose=1 
    )

    model.save(user_model_path)
    print(f"[FINETUNE] User {user_id}: Saved personal model → {user_model_path}")
    print(f"[FINETUNE] Final val_loss: {history.history['val_loss'][-1]:.4f}")