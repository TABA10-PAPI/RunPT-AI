import json
from pathlib import Path
import numpy as np
from config.settings import RUNNING_DIR
from model.loader import load_user_model

def get_running_path(user_id: int) -> Path:
    return RUNNING_DIR / f"user_{user_id}.json"

def load_running_data(user_id: int):
    path = get_running_path(user_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())

def extract_features(record):
    pace_min, pace_sec = map(int, record["pace"].split(":"))
    pace_total = pace_min * 60 + pace_sec

    return [
        record["distance"],
        pace_total,
        record["time_sec"],
        record["avg_hr"]
    ]

def predict_battery(user_id: int):
    data = load_running_data(user_id)
    recent = data[-7:]

    if len(recent) < 7:
        recent = ([recent[-1]] * (7 - len(recent))) + recent

    features = np.array([extract_features(r) for r in recent])
    features = features.reshape(1, 7, 4)

    model = load_user_model(user_id)
    score = model.predict(features)[0][0]
    return round(score * 100, 2)
