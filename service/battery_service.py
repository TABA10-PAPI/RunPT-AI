# service/battery_service.py
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from config.settings import RUNNING_DIR
from model.loader import load_user_model


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
    """
    최근 7일치 날짜 기반 day-summary 기록 만들기.
    러닝이 없는 날은 휴식일(distance=0)로 자동 생성.
    """
    # 날짜 파싱
    parsed = []
    for r in data:
        if "date" not in r:
            continue
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        parsed.append((d.date(), r))

    # 날짜 기준 정렬
    parsed.sort(key=lambda x: x[0])

    # 날짜별 세션 묶기
    daily_map = {}
    for d, r in parsed:
        daily_map.setdefault(d, []).append(r)

    # 오늘 기준 최근 7일 날짜
    result = []
    for i in range(days):
        day = today.date() - timedelta(days=(days - 1 - i))
        sessions = daily_map.get(day, [])

        if sessions:
            # 하루 여러 세션 → 합산 처리
            total_dist = sum(s["distance"] for s in sessions)
            total_time = sum(s["time_sec"] for s in sessions)
            avg_hr = sum(s["avg_hr"] for s in sessions) / len(sessions)
            pace_sec = total_time / total_dist if total_dist > 0 else 0.0
        else:
            # 휴식일
            total_dist = 0.0
            total_time = 0.0
            avg_hr = 60.0
            pace_sec = 0.0

        result.append({
            "date": day.strftime("%Y-%m-%d"),
            "distance": total_dist,
            "time_sec": total_time,
            "avg_hr": avg_hr,
            "pace_sec": pace_sec
        })

    return result


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(record):
    return [
        record["distance"],
        record["pace_sec"],
        record["time_sec"],
        record["avg_hr"],
    ]


# ---------------------------
# Domain Logic (규칙 기반)
# ---------------------------

def is_hard_run(record):
    """러닝 세션 고강도 여부 판단"""
    if not record:
        return False

    if record.get("is_interval", False):
        return True
    if record.get("is_race", False):
        return True
    if record.get("new_record", False):
        return True
    if record["distance"] >= 18:
        return True

    return False


def compute_rest_days_daily(daily):
    """Daily Summary 기반 휴식일 계산"""
    cnt = 0
    for r in reversed(daily):
        if r["distance"] == 0:
            cnt += 1
        else:
            break
    return cnt


def compute_daily_fatigue(daily):
    """Daily Summary 기반 피로도 계산"""
    loads = []
    for r in daily:
        load = (r["distance"] * 0.4) + ((r["avg_hr"] / 200) * 0.6)
        loads.append(load)

    max_val = max(loads) if max(loads) > 0 else 1
    fatigue = sum(loads) / (len(loads) * max_val)

    return max(0.0, min(1.0, fatigue))


def adjust_battery(raw, had_hard_run, rest_days, fatigue):
    """AI raw 예측 → 규칙 보정"""

    battery = raw

    # 1) 전날 고강도 운동 없으면 40 이하 금지
    if not had_hard_run and battery < 40:
        battery = 40.0

    # 2) 하루 이상 쉬었다면 → 70 이상 추천
    if rest_days >= 1:
        if fatigue < 0.7 and battery < 70:
            battery = 70.0
        elif fatigue < 0.85 and battery < 60:
            battery = 60.0

    # ⭐ 3) 2일 이상 휴식 → 90~100 자연 상승
    if rest_days >= 2:
        if fatigue < 0.5:
            battery = max(battery, 95)
        else:
            battery = max(battery, 90)

    # 4) 평상시 미세 상향
    battery += 5

    # 5) 0~100 범위로 제한
    battery = max(0, min(100, battery))
    return round(battery, 2)


def predict_battery(user_id: int, date_str: str):
    data = load_running_data(user_id)
    if not data:
        return 75.0, 0, 0.0, False  # battery, rest_days, fatigue, hard_run

    today = datetime.strptime(date_str, "%Y-%m-%d")

    # 일단 day summary 생성
    daily = build_daily_records(data, today, days=7)

    # LSTM 입력 구성
    features = np.array([extract_features(r) for r in daily])
    features = features.reshape(1, 7, 4)

    # AI raw score
    model = load_user_model(user_id)
    raw_score = model.predict(features)[0][0]
    raw_battery = raw_score * 100

    # Domain logic 계산
    yesterday_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    yesterday_session = None
    for r in reversed(data):
        if r.get("date") == yesterday_date:
            yesterday_session = r
            break

    had_hard_run = is_hard_run(yesterday_session)
    rest_days = compute_rest_days_daily(daily)
    fatigue = compute_daily_fatigue(daily)

    final = adjust_battery(
        raw=raw_battery,
        had_hard_run=had_hard_run,
        rest_days=rest_days,
        fatigue=fatigue
    )

    return final, rest_days, fatigue, had_hard_run

def explain_battery_score(battery: float, rest_days: int, fatigue: float, had_hard_run: bool):
    """
    배터리 점수 + 휴식일 + 피로도 + 하드런 여부 기반 설명 생성
    """

    # ---------------------------
    # Reason 생성
    # ---------------------------
    reasons = []

    # 휴식일 기반
    if rest_days >= 3:
        reasons.append("최근 3일 이상 충분한 휴식을 취했습니다.")
    elif rest_days == 2:
        reasons.append("최근 2일 동안 휴식을 취하며 회복이 잘 이루어졌습니다.")
    elif rest_days == 1:
        reasons.append("전날 휴식을 취해 회복이 어느 정도 이루어졌습니다.")
    else:
        reasons.append("최근 며칠간 꾸준히 러닝을 수행했습니다.")

    # 하드런 기반
    if had_hard_run:
        reasons.append("전날 고강도 운동을 수행하여 피로가 누적되었습니다.")

    # 피로도 기반
    if fatigue >= 0.8:
        reasons.append("최근 러닝 강도와 심박 수준이 높아 피로도가 높은 상태입니다.")
    elif fatigue >= 0.5:
        reasons.append("최근 러닝 강도가 중간 수준으로 피로가 약간 누적되었습니다.")
    else:
        reasons.append("러닝 강도가 낮아 피로도가 낮은 상태입니다.")

    reason_text = " ".join(reasons)

    # ---------------------------
    # Feedback 생성
    # ---------------------------
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
