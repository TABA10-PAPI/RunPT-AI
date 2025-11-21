from fastapi import FastAPI
from schemas.battery_request import BatteryRequest
from schemas.battery_response import BatteryResponse, RecommendedRun
from schemas.running_save_request import RunningSaveRequest
from service.battery_service import predict_battery, load_running_data, get_running_path
from service.skill_service import load_user_skill, update_user_skill
from service.recommendation_engine import (
    generate_recommendations,
    generate_beginner_recommendations,
)
import json

app = FastAPI(title="RunPT-AI Prototype")

@app.post("/battery", response_model=BatteryResponse)
def get_battery_info(req: BatteryRequest):
    # 1) 먼저 러닝 기록이 있는지 확인
    runs = load_running_data(req.user_id)

    # 2) 러닝 기록이 전혀 없는 사용자
    if not runs:
        # 충분히 휴식을 취했다고 가정 → 배터리 100
        battery_score = 100.0
        rec_dicts = generate_beginner_recommendations()

    # 3) 러닝 기록이 있는 사용자 : LSTM + 실력 기반 추천
    else:
        battery_score = predict_battery(req.user_id)
        skill = load_user_skill(req.user_id)
        rec_dicts = generate_recommendations(skill, battery_score)

    return BatteryResponse(
        battery_score=battery_score,
        recommendations=[RecommendedRun(**r) for r in rec_dicts]
    )


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

@app.get("/")
def root():
    return {"message": "RunPT-AI API running"}
