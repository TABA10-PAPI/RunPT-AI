# tests/test_battery_api.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_battery():
    payload = {
        "userId": 1,
        "sequence": [
            {
                "hr": 70,
                "hrv": 50,
                "pace": 6.0,
                "sleep_hours": 7.0,
                "distance_km": 5.0,
                "calories": 400,
            }
            for _ in range(7)  # 7일치라고 가정
        ],
    }

    resp = client.post("/predict/battery", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "battery" in data
    assert "stress" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
