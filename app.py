from fastapi import FastAPI
from pydantic import BaseModel

from schemas.add_user_request import AddUserRequest
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
    get_running_path,
    compute_acute_fatigue  # ← 전날 피로도 기반 acute fatigue
)

from service.skill_service import (
    load_user_skill, 
    update_user_skill,
    save_user_skill,
    save_user_profile
)

from service.recommendation_engine import (
    generate_recommendations,
    generate_beginner_recommendations,
    
)

import json
from datetime import datetime, timedelta

app = FastAPI(title="RunPT-AI Prototype")


# --------------------------------------------------------------------
# 공통 로직: 배터리 + 추천 생성
# --------------------------------------------------------------------
def _compute_battery_and_recommendations(req: BatteryRequest):

    user_id = req.user_id
    date_str = req.date
    runs = load_running_data(user_id)

    # 기록 없음 → 초보자 추천
    if not runs:
        battery_score = 100.0
        rec_dicts = generate_beginner_recommendations()
        recommendations = [RecommendedRun(**r) for r in rec_dicts]
        return battery_score, recommendations

    # 기록 있음 → 배터리 계산
    battery_score, rest_days, fatigue, had_hard_run = predict_battery(user_id, date_str)

    # 스킬 로드
    skill = load_user_skill(user_id)

    # ----------------------------
    # 🔥 전날 러닝 기록 가져오기
    # ----------------------------
    yesterday = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_session = None
    for r in reversed(runs):
        if r.get("date", "")[:10] == yesterday:
            yesterday_session = r
            break

    # ----------------------------
    # 🔥 acute fatigue 계산
    # ----------------------------
    acute_fatigue = compute_acute_fatigue(yesterday_session)

    # ----------------------------
    # 추천 생성
    # ----------------------------
    rec_dicts = generate_recommendations(
        skill=skill,
        battery_score=battery_score,
        acute_fatigue=acute_fatigue
    )
    recommendations = [RecommendedRun(**r) for r in rec_dicts]

    return battery_score, recommendations


# --------------------------------------------------------------------
# API: 배터리 + 추천 같이 반환
# --------------------------------------------------------------------
@app.post("/battery", response_model=BatteryResponse)
def get_battery_info(req: BatteryRequest):
    battery_score, recommendations = _compute_battery_and_recommendations(req)
    return BatteryResponse(
        battery_score=battery_score,
        recommendations=recommendations,
    )


# --------------------------------------------------------------------
# API: 배터리 점수만 반환 (+ 이유 & 피드백 포함)
# --------------------------------------------------------------------
@app.post("/battery/score", response_model=BatteryScoreResponse)
def get_battery_score(req: BatteryRequest):

    battery_score, rest_days, fatigue, had_hard_run = predict_battery(req.user_id, req.date)

    reason, feedback = explain_battery_score(
        battery=battery_score,
        rest_days=rest_days,
        fatigue=fatigue,
        had_hard_run=had_hard_run
    )

    return BatteryScoreResponse(
        battery_score=battery_score,
        reason=reason,
        feedback=feedback
    )


# --------------------------------------------------------------------
# API: 추천만 반환
# --------------------------------------------------------------------
@app.post("/battery/recommendations", response_model=RecommendationResponse)
def get_battery_recommendations(req: BatteryRequest):
    _, recommendations = _compute_battery_and_recommendations(req)
    return RecommendationResponse(
        recommendations=recommendations
    )


# --------------------------------------------------------------------
# API: 러닝 기록 저장
# --------------------------------------------------------------------
@app.post("/running/save")
def save_running(req: RunningSaveRequest):
    path = get_running_path(req.user_id)
    data = load_running_data(req.user_id)

    new_record = req.dict()
    data.append(new_record)

    path.write_text(json.dumps(data, indent=2))

    skill = update_user_skill(req.user_id, new_record)

    return {
        "message": "Running saved",
        "updated_skill": skill
    }

# --------------------------------------------------------------------
# API: 신규 유저 등록 (/add-user)
# --------------------------------------------------------------------
@app.post("/add-user")
def add_user(req: AddUserRequest):
    user_id = req.uid

    # 1) 프로필 파일 저장
    profile = {
        "user_id": user_id,
        "age": req.age,
        "height": req.height,
        "weight": req.weight,
    }
    save_user_profile(user_id, profile)

    # 2) 스킬 파일에도 동일 정보 반영
    skill = load_user_skill(user_id)
    skill["age"] = req.age
    skill["height"] = req.height
    skill["weight"] = req.weight
    save_user_skill(user_id, skill)

    return {
        "message": "User profile & skill initialized",
        "user_id": user_id,
        "profile": profile,
    }


@app.get("/")
def root():
    return {"message": "RunPT-AI API running"}
