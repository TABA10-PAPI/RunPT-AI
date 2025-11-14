# schemas/battery.py
from pydantic import BaseModel, Field
from typing import List, Optional


class SequenceItem(BaseModel):
    hr: float = Field(..., description="심박수 (bpm)")
    hrv: float = Field(..., description="HRV (예: RMSSD)")
    pace: float = Field(..., description="페이스 (분/km)")
    sleep_hours: float = Field(..., description="수면 시간 (시간)")
    distance_km: float = Field(..., description="일일 러닝 거리 (km)")
    calories: float = Field(..., description="활동 칼로리 (kcal)")


class BatteryRequest(BaseModel):
    userId: int
    sequence: List[SequenceItem]


class Recommendation(BaseModel):
    type: str
    distance_km: Optional[float] = None
    pace_min_per_km: Optional[float] = None
    note: Optional[str] = None


class BatteryResponse(BaseModel):
    battery: float
    stress: float
    recommendations: List[Recommendation]
