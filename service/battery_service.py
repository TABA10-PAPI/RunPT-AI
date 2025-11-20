import json
import numpy as np
from pathlib import Path
from typing import List

from model.loader import load_user_model
from model.fine_tune import finetune_user_model
from schemas.battery import SequenceItem, BatteryResponse, Recommendation

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "users"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# LSTM이 학습된 시퀀스 길이
WINDOW_SIZE = 7


def save_user_history(user_id: int, sequence: List[SequenceItem]):
    """
    각 사용자의 러닝 기록을 파일(data/users/{userId}_history.json)에 세션 단위로 저장.
    한 번 요청이 들어올 때마다 sequence 전체를 하나의 세션으로 쌓는다.
    """
    path = DATA_DIR / f"{user_id}_history.json"

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    # pydantic BaseModel -> dict로 변환해서 저장
    history.append([item.dict() for item in sequence])

    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _load_flat_history(user_id: int) -> List[SequenceItem]:
    """
    data/users/{userId}_history.json 을 읽어서
    [SequenceItem, SequenceItem, ...] 형태로 평탄화해서 반환.
    """
    path = DATA_DIR / f"{user_id}_history.json"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)  # [ [ {..}, {..} ], [ {..}, ... ], ... ]

    flat: List[SequenceItem] = []
    for session in history:
        for item in session:
            flat.append(SequenceItem(**item))

    return flat


def _build_window_from_history(user_id: int, latest_sequence: List[SequenceItem]) -> List[SequenceItem]:
    """
    사용자별 전체 히스토리를 평탄화해서,
    최근 WINDOW_SIZE 개의 러닝 데이터로 윈도우를 만든다.
    - 히스토리가 부족하면 마지막 값을 복제해서 패딩한다.
    - 히스토리가 아예 없으면 latest_sequence를 기반으로 만든다.
    """
    flat = _load_flat_history(user_id)

    # 히스토리가 전혀 없는 극단적인 경우 (이론상 거의 없겠지만 안전하게 처리)
    if not flat:
        flat = list(latest_sequence)

    if len(flat) >= WINDOW_SIZE:
        window = flat[-WINDOW_SIZE:]
    else:
        # 부족하면 마지막 값을 복제해서 채움
        last = flat[-1]
        window = flat + [last] * (WINDOW_SIZE - len(flat))

    return window


def _predict_battery(user_id: int, latest_sequence: List[SequenceItem]) -> float:
    """
    - 사용자별 히스토리를 읽어서 최근 WINDOW_SIZE 개로 윈도우를 만든 뒤
    - 사용자 전용 모델(없으면 base 모델)을 로딩해서 예측.
    """
    model = load_user_model(user_id)
    window = _build_window_from_history(user_id, latest_sequence)

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
            for s in window
        ],
        dtype="float32",
    )
    # (1, WINDOW_SIZE, 6)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    value = float(pred[0][0])
    # 0~100 사이로 클램핑
    return max(0.0, min(100.0, value))


def _make_recommendations(battery_value: float) -> List[Recommendation]:
    recs: List[Recommendation] = []

    if battery_value > 80:
        recs.append(
            Recommendation(
                type="HIGH",
                note="에너지가 충분해요! 훈련 강도를 올려보세요."
            )
        )
    elif battery_value > 50:
        recs.append(
            Recommendation(
                type="MEDIUM",
                note="적당한 강도로 러닝하기 좋은 상태에요."
            )
        )
    else:
        recs.append(
            Recommendation(
                type="LOW",
                note="충분한 휴식이 필요해요. 무리하지 마세요."
            )
        )

    return recs



def handle_battery_request(user_id: int, sequence: List[SequenceItem]) -> BatteryResponse:
    """
    /predict/battery 진입점에서 호출되는 함수.
    - 들어온 sequence(한 개든 여러 개든)를 히스토리에 저장
    - 히스토리 전체를 기반으로 예측
    - 예측 후, 백그라운드처럼 사용자별 파인튜닝 시도
    """
    if not sequence:
        # 극단적으로 빈 배열이 들어오는 상황 방어 (실제로는 안 그럴 거지만)
        default_battery = 50.0
        recs = _make_recommendations(default_battery)
        return BatteryResponse(userId=user_id, battery=default_battery, recommendations=recs)

    # 1) 히스토리에 저장 (세션 단위)
    save_user_history(user_id, sequence)

    # 2) 현재까지의 히스토리(최근 WINDOW_SIZE)를 기반으로 예측
    battery = _predict_battery(user_id, sequence)

    # 3) 예측 후, 사용자별 파인튜닝 시도
    #    데이터가 부족하면 finetune_user_model 내부에서 그냥 return 하므로 에러 없음.
    try:
        finetune_user_model(user_id, epochs=1)
    except Exception as e:
        print(f"[FINETUNE] Error while finetuning user {user_id}: {e}")

    # 4) 추천 생성 및 응답
    recs = _make_recommendations(battery)

    # stress는 일단 예시로 "배터리 낮을수록 스트레스 높다" 가정해서 100 - battery 로 설정
    stress = max(0.0, min(100.0, 100.0 - battery))

    return BatteryResponse(
        battery=battery,
        stress=stress,
        recommendations=recs,
    )

