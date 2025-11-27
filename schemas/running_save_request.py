from pydantic import BaseModel, validator

class RunningSaveRequest(BaseModel):
    user_id: int
    date: str
    distance: float
    pace_sec: int
    time_sec: int
    avg_hr: int

    @validator("pace_sec")
    def pace_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("pace_sec must be positive")
        return v