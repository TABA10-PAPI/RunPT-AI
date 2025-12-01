# service/battery_service.py
import json
from pathlib import Path
import numpy as np
from config.settings import RUNNING_DIR
from model.loader import load_user_model


def get_running_path(user_id: int) -> Path:
    return RUNNING_DIR / f"user_{user_id}.json"


def load_running_data(user_id: int):
    path = get_running_path(user_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(record):
    return [
        record["distance"],
        record["pace_sec"],   # 초 단위 페이스
        record["time_sec"],
        record["avg_hr"]
    ]

# ---------------------------
# Domain Logic (규칙 기반)
# ---------------------------

def is_hard_run(record):
    """전날 고강도 운동 여부 판단 (기본 규칙 버전)"""
    if not record:
        return False

    # ❶ 인터벌 기록
    if record.get("is_interval", False):
        return True

    # ❷ 레이스 페이스 수준 달림
    if record.get("is_race", False):
        return True

    # ❸ 신기록 갱신
    if record.get("new_record", False):
        return True

    # ❹ LSD (거리 기반)
    if record["distance"] >= 18:
        return True

    return False


def compute_rest_days(data):
    """최근 연속 휴식일 계산"""
    cnt = 0
    for r in reversed(data):
        if r["distance"] == 0:
            cnt += 1
        else:
            break
    return cnt


def compute_accumulated_fatigue(data):
    """최근 러닝 강도 기반 피로도 (0~1 간단 계산)"""
    if not data:
        return 0

    loads = []
    for r in data[-7:]:
        # 거리 + 페이스 + 심박 등을 활용한 간단한 로드
        load = (r["distance"] * 0.4) + (r["avg_hr"] / 200 * 0.6)
        loads.append(load)

    max_val = max(loads) if max(loads) > 0 else 1
    fatigue = sum(loads) / (len(loads) * max_val)
    return min(1, max(0, fatigue))


def adjust_battery(raw, had_hard_run, rest_days, fatigue):
    """AI raw 예측 → 규칙 기반 보정 → 최종 배터리"""
    battery = raw

    # ---------------------
    # 1) 전날 고강도 운동 없으면 40 이하 금지
    # ---------------------
    if not had_hard_run and battery < 40:
        battery = 40.0

    # ---------------------
    # 2) 하루 이상 쉬었다면 웬만하면 70 이상
    # ---------------------
    if rest_days >= 1:
        # 피로도가 낮으면 강하게 보정
        if fatigue < 0.7 and battery < 70:
            battery = 70.0
        # 피로도가 조금 높아도 60 아래는 가기 어렵도록
        elif fatigue < 0.85 and battery < 60:
            battery = 60.0

    # ---------------------
    # 3) 전반적으로 평균 70 근처가 되게 소폭 상향 보정
    # ---------------------
    battery += 5

    # ---------------------
    # 4) 0~100 범위 안으로
    # ---------------------
    battery = max(0, min(100, battery))

    return round(battery, 2)


# ---------------------------
# Predict Battery
# ---------------------------
def predict_battery(user_id: int):
    data = load_running_data(user_id)
    if not data:
        return 75.0  # 기본값

    recent = data[-7:]

    # 7일 미만 → 가장 최근 기록으로 채우기
    if len(recent) < 7:
        recent = ([recent[-1]] * (7 - len(recent))) + recent

    # LSTM 입력 구성
    features = np.array([extract_features(r) for r in recent])
    features = features.reshape(1, 7, 4)

    # 모델 불러오기
    model = load_user_model(user_id)
    raw_score = model.predict(features)[0][0]
    raw_battery = raw_score * 100

    # ---------------------
    # 후처리 보정 로직
    # ---------------------
    yesterday = recent[-1]
    had_hard_run = is_hard_run(yesterday)
    rest_days = compute_rest_days(data)
    fatigue = compute_accumulated_fatigue(data)

    final_battery = adjust_battery(
        raw=raw_battery,
        had_hard_run=had_hard_run,
        rest_days=rest_days,
        fatigue=fatigue
    )

    return final_battery
