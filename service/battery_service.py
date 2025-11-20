import json
import numpy as np
from pathlib import Path
from typing import List

from model.loader import load_user_model
from schemas.battery import SequenceItem, BatteryResponse, Recommendation

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "users"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 7


def save_user_history(user_id: int, sequence: List[SequenceItem]):
    """각 사용자의 러닝 기록을 파일(data/users/userid_history.json)에 저장"""
    path = DATA_DIR / f"{user_id}_history.json"

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    history.append([item.dict() for item in sequence])

    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _predict_battery(user_id: int, sequence: List[SequenceItem]) -> float:
    model = load_user_model(user_id)

    # 시퀀스 길이 보정
    if len(sequence) < WINDOW_SIZE:
        last = sequence[-1]
        sequence = list(sequence) + [last] * (WINDOW_SIZE - len(sequence))
    else:
        sequence = sequence[-WINDOW_SIZE:]

    arr = np.array(
        [
            [
                s.hr,
                s.hrv,
                s.pace,
                s.sleep_hours,
                s.distance_km,
                s.calories,
            ]
            for s in sequence
        ],
        dtype="float32",
    )
    arr = np.expand_dims(arr, axis=0)  # (1,7,6)

    pred = model.predict(arr)
    value = float(pred[0][0])
    return max(0, min(100, value))


def _make_recommendations(battery_value: float) -> List[Recommendation]:
    recs = []
    if battery_value > 80:
        recs.append(Recommendation(message="에너지가 충분해요! 훈련 강도를 올려보세요."))
    elif battery_value > 50:
        recs.append(Recommendation(message="적당한 강도로 러닝하기 좋은 상태에요."))
    else:
        recs.append(Recommendation(message="충분한 휴식이 필요해요."))
    return recs


def handle_battery_request(user_id: int, sequence: List[SequenceItem]) -> BatteryResponse:
    save_user_history(user_id, sequence)  # 사용자 기록 저장

    battery = _predict_battery(user_id, sequence)

    recs = _make_recommendations(battery)

    return BatteryResponse(
        userId=user_id,
        battery=battery,
        recommendations=recs
    )
