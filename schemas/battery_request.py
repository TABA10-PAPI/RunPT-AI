from pydantic import BaseModel

class BatteryRequest(BaseModel):
    user_id: int
    date: str
