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

    pace_min, pace_sec = map(int, new_run["pace"].split(":"))
    pace_total = pace_min * 60 + pace_sec

    skill["avg_pace_sec"] = (skill["avg_pace_sec"] * 0.7) + (pace_total * 0.3)
    skill["max_distance"] = max(skill["max_distance"], new_run["distance"])
    skill["weekly_distance"] = skill["weekly_distance"] * 0.8 + new_run["distance"] * 0.2
    skill["fatigue_level"] = min(1.0, skill["fatigue_level"] * 0.9 + (new_run["distance"] / 20) * 0.1)
    skill["consistency_score"] = min(1.0, skill["consistency_score"] + 0.01)

    save_user_skill(user_id, skill)
    return skill
