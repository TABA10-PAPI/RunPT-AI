from pydantic import BaseModel
from typing import List, Dict, Any

class RecommendedRun(BaseModel):
    type: str
    distance_km: float
    target_pace: str
    reason: str

class BatteryResponse(BaseModel):
    battery_score: float
    recommendations: List[RecommendedRun]
