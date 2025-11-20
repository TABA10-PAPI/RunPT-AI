import json
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "users"
BASE_MODEL_PATH = BASE_DIR / "model" / "battery_lstm_base.h5"


def get_user_model_path(user_id: int):
    return BASE_DIR / "model" / f"user_{user_id}_model.h5"


def finetune_user_model(user_id: int, epochs: int = 3):
    user_history_path = DATA_DIR / f"{user_id}_history.json"

    if not user_history_path.exists():
        print(f"[FINETUNE] No history for user {user_id}")
        return

    with user_history_path.open("r") as f:
        history = json.load(f)

    x_list, y_list = [], []

    for session in history:
        if len(session) < 7:
            continue

        seq = session[-7:]

        x = [
            [
                s["hr"],
                s["hrv"],
                s["pace"],
                s["sleep_hours"],
                s["distance_km"],
                s["calories"],
            ]
            for s in seq
        ]

        x_list.append(x)
        y_list.append(50.0)  # 실제 target 값은 너가 정의해야 함

    if not x_list:
        print("[FINETUNE] Not enough data.")
        return

    X = np.array(x_list, dtype="float32")
    y = np.array(y_list, dtype="float32")

    user_model_path = get_user_model_path(user_id)

    if user_model_path.exists():
        model = tf.keras.models.load_model(user_model_path)
        print(f"[FINETUNE] Continue training user model: {user_model_path}")
    else:
        model = tf.keras.models.load_model(BASE_MODEL_PATH)
        print(f"[FINETUNE] Start finetuning base model")

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, batch_size=4, epochs=epochs)

    model.save(user_model_path)
    print(f"[FINETUNE] Saved personalized model to: {user_model_path}")
