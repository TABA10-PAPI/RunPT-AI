from pydantic import BaseModel

class RunningSaveRequest(BaseModel):
    user_id: int
    date: str
    distance: float
    pace: str      # "5:25" 같은 페이스
    time_sec: int
    avg_hr: int
