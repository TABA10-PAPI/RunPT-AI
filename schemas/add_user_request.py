from pydantic import BaseModel

class AddUserRequest(BaseModel):
    uid: int           # 내부적으로 user_id로 사용
    age: int           # 나이
    height: int        # 키 (cm)
    weight: int        # 몸무게 (kg)
