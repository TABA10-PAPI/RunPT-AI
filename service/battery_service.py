# service/battery_service.py
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from config.settings import RUNNING_DIR
from model.loader import load_user_model
from service.skill_service import get_user_static_features

WINDOW_SIZE = 7
FEATURE_DIM = 7  # distance, pace_sec, time_sec, avg_hr, age, height, weight


# ---------------------------
# 날짜 정규화 함수
# ---------------------------
def clean_date(date_str: str) -> str:
    """
    YYYY-MM-DD만 추출 (ISO8601 포함 전체 형식 대응)
    예: '2025-11-05T09:25:00Z' → '2025-11-05'
    """
    return date_str.split("T")[0]


# ---------------------------
# 파일 로드
# ---------------------------
def get_running_path(user_id: int) -> Path:
    return RUNNING_DIR / f"user_{user_id}.json"


def load_running_data(user_id: int):
    path = get_running_path(user_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ---------------------------
# 날짜 기반 Daily Summary 생성
# ---------------------------
def build_daily_records(data: list[dict], today: datetime, days: int = 7):
    parsed = []

    for r in data:
        if "date" not in r:
            continue

        # 날짜 정규화 적용
        date_str = clean_date(r["date"])
        d = datetime.strptime(date_str, "%Y-%m-%d")

        parsed.append((d.date(), r))

    # 날짜 기준 정렬
    parsed.sort(key=lambda x: x[0])

    # 날짜별 세션 묶기
    daily_map: dict = {}
    for d, r in parsed:
        daily_map.setdefault(d, []).append(r)

    # 오늘 기준 days일 생성
    result: list[dict] = []
    for i in range(days):
        day = today.date() - timedelta(days=(days - 1 - i))
        sessions = daily_map.get(day, [])

        if sessions:
            total_dist = sum(s.get("distance", 0.0) for s in sessions)
            total_time = sum(s.get("time_sec", 0.0) for s in sessions)
            avg_hr = sum(s.get("avg_hr", 60.0) for s in sessions) / len(sessions)
            pace_sec = total_time / total_dist if total_dist > 0 else 0.0
        else:
            total_dist = 0.0
            total_time = 0.0
            avg_hr = 60.0
            pace_sec = 0.0

        result.append({
            "date": day.strftime("%Y-%m-%d"),
            "distance": float(total_dist),
            "time_sec": float(total_time),
            "avg_hr": float(avg_hr),
            "pace_sec": float(pace_sec),
        })

    return result


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(record: dict, age: float, height: float, weight: float):
    return [
        float(record["distance"]),
        float(record["pace_sec"]),
        float(record["time_sec"]),
        float(record["avg_hr"]),
        float(age),
        float(height),
        float(weight),
    ]


# ---------------------------
# Domain Logic (규칙 기반)
# ---------------------------
def is_hard_run(record: dict | None):
    if not record:
        return False

    if record.get("is_interval", False):
        return True
    if record.get("is_race", False):
        return True
    if record.get("new_record", False):
        return True
    if record.get("distance", 0.0) >= 18:
        return True

    return False


def compute_rest_days_daily(daily: list[dict]) -> int:
    cnt = 0
    for r in reversed(daily):
        if r["distance"] == 0:
            cnt += 1
        else:
            break
    return cnt


def compute_daily_fatigue(daily: list[dict]) -> float:
    loads = []
    for r in daily:
        load = (r["distance"] * 0.4) + ((r["avg_hr"] / 200.0) * 0.6)
        loads.append(load)

    max_val = max(loads) if loads and max(loads) > 0 else 1.0
    fatigue = sum(loads) / (len(loads) * max_val)

    return max(0.0, min(1.0, fatigue))


def adjust_battery(raw: float, had_hard_run: bool, rest_days: int, fatigue: float) -> float:
    battery = raw

    # 모델이 너무 낮게 준 경우, 최근에 빡센 런이 없으면 최소 방어선
    if not had_hard_run and battery < 40:
        battery = 40.0

    # 휴식 1일 이상
    if rest_days >= 1:
        if fatigue < 0.7 and battery < 70:
            battery = 70.0
        elif fatigue < 0.85 and battery < 60:
            battery = 60.0

    # 휴식 2일 이상
    if rest_days >= 2:
        if fatigue < 0.5:
            battery = max(battery, 95.0)
        else:
            battery = max(battery, 90.0)

    # 최종 보정
    battery += 5.0
    battery = max(0.0, min(100.0, battery))
    return round(battery, 2)


# ---------------------------
# 배터리 예측
# ---------------------------
def predict_battery(user_id: int, date_str: str):
    data = load_running_data(user_id)
    if not data:
        return 75.0, 0, 0.0, False

    today = datetime.strptime(clean_date(date_str), "%Y-%m-%d")

    # 최근 7일 요약 생성
    daily = build_daily_records(data, today, days=WINDOW_SIZE)

    # 🔥 유저 정적 특성(나이/키/몸무게) 로드
    static = get_user_static_features(user_id)
    age = static["age"]
    height = static["height"]
    weight = static["weight"]

    # LSTM 입력 (1, 7, 7) = (batch, time, feature_dim)
    features = np.array(
        [extract_features(r, age, height, weight) for r in daily],
        dtype="float32"
    ).reshape(1, WINDOW_SIZE, FEATURE_DIM)

    model = load_user_model(user_id)
    raw_score = float(model.predict(features)[0][0])
    raw_battery = raw_score * 100.0

    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # 날짜 정규화 후 비교
    yesterday_session = None
    for r in reversed(data):
        if clean_date(r.get("date", "")) == yesterday_str:
            yesterday_session = r
            break

    had_hard_run = is_hard_run(yesterday_session)
    rest_days = compute_rest_days_daily(daily)
    fatigue = compute_daily_fatigue(daily)
    fatigue = apply_rest_decay_weighted(daily, fatigue)

    final = adjust_battery(
        raw=raw_battery,
        had_hard_run=had_hard_run,
        rest_days=rest_days,
        fatigue=fatigue
    )

    return final, rest_days, fatigue, had_hard_run


# ---------------------------
# 배터리 설명 생성
# ---------------------------
def explain_battery_score(battery: float, rest_days: int, fatigue: float, had_hard_run: bool):

    reasons = []

    if rest_days >= 3:
        reasons.append("최근 3일 이상 충분한 휴식을 취했습니다.")
    elif rest_days == 2:
        reasons.append("최근 2일 동안 휴식을 취하며 회복이 잘 이루어졌습니다.")
    elif rest_days == 1:
        reasons.append("전날 휴식을 취해 회복이 어느 정도 이루어졌습니다.")
    else:
        reasons.append("최근 며칠간 꾸준히 러닝을 수행했습니다.")

    if had_hard_run:
        reasons.append("전날 고강도 운동을 수행하여 피로가 누적되었습니다.")

    if fatigue >= 0.8:
        reasons.append("최근 러닝 강도와 심박 수준이 높아 피로도가 높은 상태입니다.")
    elif fatigue >= 0.5:
        reasons.append("최근 러닝 강도가 중간 수준으로 피로가 약간 누적되었습니다.")
    else:
        reasons.append("러닝 강도가 낮아 피로도가 낮은 상태입니다.")

    reason_text = " ".join(reasons)

    if battery >= 85:
        feedback = "오늘은 상태가 매우 좋습니다! 템포런이나 인터벌 같은 고강도 훈련도 가능합니다."
    elif battery >= 70:
        feedback = "상태가 양호합니다. 스테디런 또는 중강도 훈련을 추천합니다."
    elif battery >= 50:
        feedback = "무리하지 않는 것이 좋습니다. 가벼운 이지런 또는 조깅 정도로 훈련하세요."
    elif battery >= 30:
        feedback = "피로가 누적된 상태입니다. 회복 위주의 조깅 또는 휴식을 추천합니다."
    else:
        feedback = "매우 피곤한 상태입니다. 오늘은 완전 휴식을 취하는 것이 좋습니다."

    return reason_text, feedback


def compute_acute_fatigue(latest_run: dict | None) -> float:
    """전날 러닝 기반 단기 피로도 계산"""
    if latest_run is None:
        return 0.1  # 휴식일 → 피로도 매우 낮음

    dist = latest_run.get("distance", 0.0)
    hr = latest_run.get("avg_hr", 120.0)
    pace = latest_run.get("pace_sec", 360.0)

    # 기본 피로도
    fatigue = 0.1

    # 거리 기반
    if dist >= 15:
        fatigue += 0.5
    elif dist >= 10:
        fatigue += 0.3
    elif dist >= 5:
        fatigue += 0.1

    # 심박 기반
    if hr >= 165:
        fatigue += 0.4
    elif hr >= 150:
        fatigue += 0.2

    # interval / race 플래그
    if latest_run.get("is_interval", False) or latest_run.get("is_race", False):
        fatigue = max(fatigue, 0.8)

    return min(1.0, fatigue)


def apply_rest_decay_weighted(daily: list[dict], fatigue: float) -> float:
    """
    각 휴식일의 '최근일수 가중치' 기반 피로도 감소
    daily: 최근 7일 (0: 가장 오래전, 6: 오늘)
    """
    rest_effect = 0.0

    # daily[-1] = 오늘
    # daily[-2] = 전날
    # daily[-3] = 2일 전 ...
    for idx in range(1, len(daily) + 1):
        day_ago = idx
        day_record = daily[-day_ago]

        # 휴식일 판정
        if day_record["distance"] == 0:
            weight = 0.5 ** (day_ago - 1)   # 전날=1→0.5^0=1.0, 2일 전=0.5, 3일 전=0.25...
            rest_effect += weight

    # 최대 영향도 제한 (너무 많이 깎이지 않도록)
    rest_effect = min(rest_effect, 0.9)

    new_fatigue = fatigue * (1 - rest_effect)
    return round(new_fatigue, 4)
