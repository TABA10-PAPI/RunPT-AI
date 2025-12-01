# schemas/battery_response.py
from pydantic import BaseModel
from typing import List

class RecommendedRun(BaseModel):
    type: str
    distance_km: float
    target_pace: str
    reason: str

class BatteryResponse(BaseModel):
    battery_score: float
    recommendations: List[RecommendedRun]

class BatteryScoreResponse(BaseModel):
    battery_score: float
    reason: str
    feedback: str


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedRun]
