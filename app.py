from fastapi import FastAPI
from schemas.battery import BatteryRequest, BatteryResponse
from services.battery_service import handle_battery_request

app = FastAPI(title="RunPT-AI Service")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/battery", response_model=BatteryResponse)
def predict_battery(req: BatteryRequest):
    """
    러닝 배터리 + 스트레스 + 러닝 추천 반환 엔드포인트
    """
    return handle_battery_request(req.userId, req.sequence)
