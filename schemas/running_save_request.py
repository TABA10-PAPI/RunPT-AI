from pydantic import BaseModel, validator

class RunningSaveRequest(BaseModel):
    user_id: int
    date: str
    distance: float  # 입력은 m 또는 km 둘 다 가능
    pace_sec: int
    time_sec: int
    avg_hr: int

    # ------------------------
    # distance를 km 단위로 변환
    # ------------------------
    @validator("distance", pre=True)
    def convert_distance_to_km(cls, v):
        """
        Spring이 5000(m) 같은 값을 보내면 km로 자동 변환한다.
        이미 5.0처럼 km이면 그대로 둔다.
        기준: 100 이상이면 m 단위로 간주.
        """
        if v is None:
            return v

        try:
            val = float(v)
        except:
            raise ValueError("distance must be numeric")

        return val / 1000.0 if val > 200 else val

    # ------------------------
    # pace_sec validation
    # ------------------------
    @validator("pace_sec")
    def pace_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("pace_sec must be positive")
        return v
