import json
from pathlib import Path
from config.settings import PROFILE_DIR


def get_skill_path(user_id: int) -> Path:
    return PROFILE_DIR / f"user_{user_id}_skill.json"


def load_user_skill(user_id: int):
    path = get_skill_path(user_id)
    if not path.exists():
        return {
            "avg_pace_sec": 360,
            "max_distance": 5.0,
            "weekly_distance": 0.0,
            "fatigue_level": 0.3,
            "consistency_score": 0.5
        }
    return json.loads(path.read_text())


def save_user_skill(user_id: int, skill: dict):
    path = get_skill_path(user_id)
    path.write_text(json.dumps(skill, indent=2))


def update_user_skill(user_id: int, new_run: dict):
    skill = load_user_skill(user_id)

    # pace_total → 이미 초 단위로 입력됨
    pace_total = new_run["pace_sec"]
    distance = new_run["distance"]

    # 1) avg pace 업데이트
    skill["avg_pace_sec"] = (skill["avg_pace_sec"] * 0.7) + (pace_total * 0.3)

    # 2) 최대 거리
    skill["max_distance"] = max(skill["max_distance"], distance)

    # 3) 주간 거리 (지수 평활)
    skill["weekly_distance"] = skill["weekly_distance"] * 0.8 + distance * 0.2

    # 4) 피로도
    fatigue_increase = (distance / 20) * 0.1
    skill["fatigue_level"] = skill["fatigue_level"] * 0.9 + fatigue_increase
    skill["fatigue_level"] = min(1.0, max(0.0, skill["fatigue_level"]))

    # 5) 꾸준함 점수 개선된 버전
    consistency = skill["consistency_score"]

    consistency *= 0.99  # 자연 감소

    if distance >= 1.0:
        consistency += 0.01
    else:
        consistency += 0.003

    if skill["fatigue_level"] > 0.8:
        consistency -= 0.02

    consistency = min(1.0, max(0.0, consistency))
    skill["consistency_score"] = consistency

    save_user_skill(user_id, skill)
    return skill
