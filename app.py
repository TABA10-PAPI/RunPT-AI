# app.py
from fastapi import FastAPI
from schemas.battery_request import BatteryRequest
from schemas.battery_response import (
    BatteryResponse,
    RecommendedRun,
    BatteryScoreResponse,
    RecommendationResponse,
)
from schemas.running_save_request import RunningSaveRequest
from service.battery_service import predict_battery, load_running_data, get_running_path
from service.skill_service import load_user_skill, update_user_skill
from service.recommendation_engine import (
    generate_recommendations,
    generate_beginner_recommendations,
)
import json

app = FastAPI(title="RunPT-AI Prototype")


def _compute_battery_and_recommendations(req: BatteryRequest):
    """
    공통 로직:
    - 러닝 기록 로드
    - 배터리 점수 계산
    - 스킬 로드
    - 러닝 추천 생성
    """
    user_id = req.user_id

    # 1) 러닝 기록 로드
    runs = load_running_data(user_id)

    # 2) 러닝 기록이 전혀 없는 사용자
    if not runs:
        # 충분히 휴식을 취했다고 가정 → 배터리 100
        battery_score = 100.0
        rec_dicts = generate_beginner_recommendations()
        recommendations = [RecommendedRun(**r) for r in rec_dicts]
        return battery_score, recommendations

    # 3) 러닝 기록이 있는 사용자: LSTM + 스킬 기반 추천
    battery_score = predict_battery(user_id)
    skill = load_user_skill(user_id)
    # acute_fatigue를 쓰는 버전이면 여기서 계산해서 같이 넘기면 됨
    rec_dicts = generate_recommendations(skill, battery_score)
    recommendations = [RecommendedRun(**r) for r in rec_dicts]

    return battery_score, recommendations


@app.post("/battery", response_model=BatteryResponse)
def get_battery_info(req: BatteryRequest):
    """
    기존 엔드포인트:
    - 배터리 점수 + 러닝 추천 3개를 한 번에 반환
    """
    battery_score, recommendations = _compute_battery_and_recommendations(req)

    return BatteryResponse(
        battery_score=battery_score,
        recommendations=recommendations,
    )


@app.post("/battery/score", response_model=BatteryScoreResponse)
def get_battery_score(req: BatteryRequest):
    """
    새 엔드포인트:
    - 배터리 점수만 반환
    """
    battery_score, _ = _compute_battery_and_recommendations(req)
    return BatteryScoreResponse(battery_score=battery_score)


@app.post("/battery/recommendations", response_model=RecommendationResponse)
def get_battery_recommendations(req: BatteryRequest):
    """
    새 엔드포인트:
    - 러닝 추천만 반환
    """
    _, recommendations = _compute_battery_and_recommendations(req)
    return RecommendationResponse(recommendations=recommendations)


@app.post("/running/save")
def save_running(req: RunningSaveRequest):
    path = get_running_path(req.user_id)
    data = load_running_data(req.user_id)

    new_record = req.dict()
    data.append(new_record)

    path.write_text(json.dumps(data, indent=2))

    # 스킬 업데이트
    skill = update_user_skill(req.user_id, new_record)

    return {
        "message": "Running saved",
        "updated_skill": skill
    }


@app.get("/")
def root():
    return {"message": "RunPT-AI API running"}
