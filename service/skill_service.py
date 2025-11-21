import json
from pathlib import Path
from config.settings import PROFILE_DIR


def get_skill_path(user_id: int) -> Path:
    return PROFILE_DIR / f"user_{user_id}_skill.json"


def load_user_skill(user_id: int):
    path = get_skill_path(user_id)
    if not path.exists():
        # 초기 기본 실력값
        return {
            "avg_pace_sec": 360,      # 6:00 페이스
            "max_distance": 5.0,      # 기본 최대거리
            "weekly_distance": 0.0,   # 최근 주간 거리
            "fatigue_level": 0.3,     # 기본 피로도
            "consistency_score": 0.5  # 기본 꾸준함 점수
        }
    return json.loads(path.read_text())


def save_user_skill(user_id: int, skill: dict):
    path = get_skill_path(user_id)
    path.write_text(json.dumps(skill, indent=2))


def update_user_skill(user_id: int, new_run: dict):
    skill = load_user_skill(user_id)

    # ------------------------------
    # 1) Pace 파싱
    # ------------------------------
    pace_min, pace_sec = map(int, new_run["pace"].split(":"))
    pace_total = pace_min * 60 + pace_sec
    distance = new_run["distance"]

    # ------------------------------
    # 2) 평균 페이스 업데이트
    # ------------------------------
    skill["avg_pace_sec"] = (skill["avg_pace_sec"] * 0.7) + (pace_total * 0.3)

    # ------------------------------
    # 3) 최대 거리 업데이트
    # ------------------------------
    skill["max_distance"] = max(skill["max_distance"], distance)

    # ------------------------------
    # 4) 최근 주간 거리 (지수 평활)
    # ------------------------------
    skill["weekly_distance"] = skill["weekly_distance"] * 0.8 + distance * 0.2

    # ------------------------------
    # 5) 피로도 업데이트 (거리 기반)
    # ------------------------------
    fatigue_increase = (distance / 20) * 0.1  # 거리 비례 피로 증가
    skill["fatigue_level"] = skill["fatigue_level"] * 0.9 + fatigue_increase
    skill["fatigue_level"] = min(1.0, max(0.0, skill["fatigue_level"]))  # 0~1 범위 제한

    # ------------------------------
    # 6) 개선된 Consistency Score
