# services/battery_service.py
from typing import List
import numpy as np

from schemas.battery import (
    SequenceItem,
    BatteryResponse,
    Recommendation,
)
from model.loader import get_model


def _predict_battery(sequence: List[SequenceItem]) -> float:
    """
    시퀀스를 LSTM에 넣어서 러닝 배터리(0~100)를 예측.
    """
    model = get_model()
    if model is None:
        # 모델이 없으면 개발 단계에서는 기본값 반환
        return 50.0

    arr = np.array(
        [
            [
                item.hr,
                item.hrv,
                item.pace,
                item.sleep_hours,
                item.distance_km,
                item.calories,
            ]
            for item in sequence
        ],
        dtype="float32",
    )
    # (time_steps, feature_dim) -> (1, time_steps, feature_dim)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    battery = float(pred[0][0])
    return max(0.0, min(100.0, battery))


def _make_recommendations(battery: float) -> List[Recommendation]:
    """
    배터리 값 기반으로 러닝 추천 생성 (규칙 기반, 나중에 ML로 교체 가능).
    """
    recs: List[Recommendation] = []

    if battery < 30:
        recs.append(
            Recommendation(
                type="REST",
                note="오늘은 컨디션이 낮아요. 가벼운 스트레칭과 휴식을 추천합니다.",
            )
        )
        recs.append(
            Recommendation(
                type="RECOVERY_WALK",
                distance_km=3.0,
                pace_min_per_km=10.0,
                note="걷기 위주로 아주 가볍게 몸만 풀어보세요.",
            )
        )
    elif battery < 60:
        recs.append(
            Recommendation(
                type="EASY_RUN",
                distance_km=5.0,
                pace_min_per_km=6.5,
                note="대화가 가능한 이지 페이스로 5km 러닝을 추천합니다.",
            )
        )
        recs.append(
            Recommendation(
                type="RECOVERY_RUN",
                distance_km=3.0,
                pace_min_per_km=7.0,
                note="짧은 거리로 가볍게 러닝하면서 몸만 풀어도 충분해요.",
            )
        )
        recs.append(
            Recommendation(
                type="REST",
                note="피곤하다면 과감히 쉬어도 괜찮습니다.",
            )
        )
    else:
        recs.append(
            Recommendation(
                type="TEMPO_RUN",
                distance_km=8.0,
                pace_min_per_km=5.5,
                note="오늘은 기록 갱신을 노려볼만 한 날이에요. 템포런으로 도전해보세요.",
            )
        )
        recs.append(
            Recommendation(
                type="INTERVAL",
                distance_km=5.0,
                pace_min_per_km=5.0,
                note="인터벌 트레이닝으로 스피드와 지구력을 같이 올려봅시다.",
            )
        )
        recs.append(
            Recommendation(
                type="EASY_RUN",
                distance_km=6.0,
                pace_min_per_km=6.0,
                note="컨디션은 좋지만, 이지런으로 조절해서 과훈련을 막는 것도 좋아요.",
            )
        )

    return recs


def handle_battery_request(
    user_id: int, sequence: List[SequenceItem]
) -> BatteryResponse:
    """
    컨트롤러에서 직접 호출하는 엔트리 포인트.
    """
    if not sequence:
        # 데이터가 없을 때 기본 응답
        return BatteryResponse(
            battery=50.0,
            stress=50.0,
            recommendations=[
                Recommendation(
                    type="REST", note="데이터가 부족하여 기본값을 반환했습니다."
                )
            ],
        )

    battery = _predict_battery(sequence)
    stress = 100.0 - battery
    recommendations = _make_recommendations(battery)

    return BatteryResponse(
        battery=battery,
        stress=stress,
        recommendations=recommendations,
    )
