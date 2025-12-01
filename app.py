from fastapi import FastAPI
from schemas.battery_request import BatteryRequest
from schemas.battery_response import (
    BatteryResponse,
    RecommendedRun,
    BatteryScoreResponse,
    RecommendationResponse,
)
from schemas.running_save_request import RunningSaveRequest

from service.battery_service import (
    explain_battery_score,
    predict_battery,
    load_running_data,
    get_running_path
)

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
    - 최근 7일(day summary) 기반 배터리 계산
    - 러닝 스킬 로드
    - 러닝 추천 생성
    """

    user_id = req.user_id
    date_str = req.date   # ← 반드시 날짜 전달

    runs = load_running_data(user_id)

    # 러닝 기록이 전혀 없는 사용자
    if not runs:
        battery_score = 100.0
        rec_dicts = generate_beginner_recommendations()
        recommendations = [RecommendedRun(**r) for r in rec_dicts]
        return battery_score, recommendations

    # 기록이 있는 사용자 → 날짜 기반 LSTM 배터리 계산
    battery_score = predict_battery(user_id, date_str)

    # 스킬 로드
    skill = load_user_skill(user_id)

    # 추천 생성
    rec_dicts = generate_recommendations(skill, battery_score)
    recommendations = [RecommendedRun(**r) for r in rec_dicts]

    return battery_score, recommendations



# ---------------------------
# API: 배터리 + 추천 함께 반환
# ---------------------------
@app.post("/battery", response_model=BatteryResponse)
def get_battery_info(req: BatteryRequest):
    battery_score, recommendations = _compute_battery_and_recommendations(req)
    return BatteryResponse(
        battery_score=battery_score,
        recommendations=recommendations,
    )


# ---------------------------
# API: 배터리 점수만 반환
# ---------------------------
@app.post("/battery/score", response_model=BatteryScoreResponse)
def get_battery_score(req: BatteryRequest):
    battery_score, rest_days, fatigue, had_hard_run = predict_battery(req.user_id, req.date)
    reason, feedback = explain_battery_score(battery_score, rest_days, fatigue, had_hard_run)

    return BatteryScoreResponse(
        battery_score=battery_score,
        reason=reason,
        feedback=feedback
    )


# ---------------------------
# API: 추천만 반환
# ---------------------------
@app.post("/battery/recommendations", response_model=RecommendationResponse)
def get_battery_recommendations(req: BatteryRequest):
    _, recommendations = _compute_battery_and_recommendations(req)
    return RecommendationResponse(
        recommendations=recommendations
    )


# ---------------------------
# API: 러닝 기록 저장
# ---------------------------
@app.post("/running/save")
def save_running(req: RunningSaveRequest):
    path = get_running_path(req.user_id)
    data = load_running_data(req.user_id)

    new_record = req.dict()  # date 포함됨
    data.append(new_record)

    path.write_text(json.dumps(data, indent=2))

    skill = update_user_skill(req.user_id, new_record)

    return {
        "message": "Running saved",
        "updated_skill": skill
    }


@app.get("/")
def root():
    return {"message": "RunPT-AI API running"}
