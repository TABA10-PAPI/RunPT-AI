# service/skill_service.py
import json
from pathlib import Path
from copy import deepcopy
from config.settings import PROFILE_DIR

"""
유저 러닝 스킬 프로필 관리 모듈.

이 모듈의 책임:
- 사용자별 장기 실력 및 습관에 대한 요약 지표 관리
  - avg_pace_sec: 장기 평균 페이스(초/키로)
  - max_distance: 한 번에 뛴 최대 거리(km)
  - weekly_distance: 최근 1주일 수준의 러닝 볼륨 (지수평활 근사)
  - training_load: 장기적인 훈련 부하(0~1), acute 피로가 아니라 chronic 패턴
  - fatigue_level: training_load의 별칭(기존 데이터 호환용)
  - consistency_score: 얼마나 꾸준히 뛰는지(0~1)

주의:
- "피로도"는 배터리 계산에서 사용하는 최근 7일 acute 피로와는 별개로,
  여기서는 "장기적인 훈련 패턴"을 나타내는 chronic load 개념으로만 사용한다.
"""

DEFAULT_SKILL = {
    "avg_pace_sec": 360.0,        # 6:00 /km
    "max_distance": 5.0,          # 5km
    "weekly_distance": 0.0,       # km 단위, 지수평활된 값
    "training_load": 0.2,         # 0~1 (chronic load)
    "fatigue_level": 0.2,         # training_load와 동일(호환용)
    "consistency_score": 0.5,     # 0~1

    # 🔥 새로 추가: 정적 특성
    # None이면 아직 /add-user를 안 탄 상태라고 보고, 모델 입력시 기본값으로 치환
    "age": None,                  # 나이 (int)
    "height": None,               # 키 (cm)
    "weight": None,               # 몸무게 (kg)
}

# 유저 기본 프로필 (정적 특성)
DEFAULT_PROFILE = {
    "user_id": None,
    "age": None,
    "height": None,
    "weight": None,
}


def get_profile_path(user_id: int) -> Path:
    """user_{id}_profile.json 위치"""
    return PROFILE_DIR / f"user_{user_id}_profile.json"


def load_user_profile(user_id: int) -> dict:
    """
    유저 프로필 로드.
    없으면 기본값 + user_id 세팅해서 반환.
    """
    path = get_profile_path(user_id)
    if not path.exists():
        prof = deepcopy(DEFAULT_PROFILE)
        prof["user_id"] = user_id
        return prof

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        prof = deepcopy(DEFAULT_PROFILE)
        prof["user_id"] = user_id
        return prof

    prof = deepcopy(DEFAULT_PROFILE)
    if isinstance(data, dict):
        prof.update(data)

    prof["user_id"] = user_id
    return prof


def save_user_profile(user_id: int, profile: dict):
    """
    유저 프로필 저장 (/add-user 에서 사용).
    age/height/weight는 int 또는 None 으로 정리해서 저장.
    """
    base = deepcopy(DEFAULT_PROFILE)
    base["user_id"] = user_id
    if profile:
        base.update(profile)

    for key in ("age", "height", "weight"):
        val = base.get(key)
        try:
            base[key] = int(val) if val is not None else None
        except (TypeError, ValueError):
            base[key] = None

    path = get_profile_path(user_id)
    path.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")


def get_skill_path(user_id: int) -> Path:
    return PROFILE_DIR / f"user_{user_id}_skill.json"


def _merge_with_default(data: dict) -> dict:
    """
    저장된 JSON에 키가 빠져있거나 오래된 형식이어도
    DEFAULT_SKILL 기반으로 안전하게 merge 한다.
    """
    skill = deepcopy(DEFAULT_SKILL)
    if data:
        skill.update(data)

    # training_load / fatigue_level 호환 처리
    if "training_load" in data:
        skill["training_load"] = float(data["training_load"])
        skill["fatigue_level"] = skill["training_load"]
    elif "fatigue_level" in data:
        # 기존 데이터가 fatigue_level만 가지고 있었던 경우
        skill["training_load"] = float(data["fatigue_level"])
        skill["fatigue_level"] = skill["training_load"]

    # 범위 클램프
    skill["training_load"] = max(0.0, min(1.0, float(skill["training_load"])))
    skill["fatigue_level"] = skill["training_load"]
    skill["consistency_score"] = max(0.0, min(1.0, float(skill["consistency_score"])))

    return skill


def load_user_skill(user_id: int) -> dict:
    """
    유저 스킬 프로필 로드.
    없으면 DEFAULT_SKILL 기반으로 초기화된 dict 반환.
    """
    path = get_skill_path(user_id)
    if not path.exists():
        return deepcopy(DEFAULT_SKILL)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # 파일이 깨졌거나 포맷이 이상하면 기본값으로 리셋
        return deepcopy(DEFAULT_SKILL)

    return _merge_with_default(data)


def save_user_skill(user_id: int, skill: dict):
    """
    유저 스킬 프로필 저장.
    training_load와 fatigue_level은 항상 동기화해서 저장한다.
    """
    path = get_skill_path(user_id)

    # 저장 전에 기본 스키마와 merge + 정리
    merged = _merge_with_default(skill)

    # training_load와 fatigue_level 동기화
    merged["fatigue_level"] = merged["training_load"]

    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


def update_user_skill(user_id: int, new_run: dict) -> dict:
    """
    새 러닝 기록(new_run)을 반영해 유저 스킬을 업데이트한다.

    new_run 예시(일부만 사용):
    {
        "distance": 10.0,       # km
        "pace_sec": 300,        # 초/키로
        "time_sec": 3000,       # 총 시간(선택적)
        ...
    }

    업데이트 로직 요약:
    - avg_pace_sec:
        거리 가중치를 둔 지수평활 평균 (롱런은 평균에 더 영향)
    - max_distance:
        한 번에 뛴 최대 거리 갱신
    - weekly_distance:
        대략 최근 1주일 거리 수준을 지수평활로 근사
    - training_load (chronic load):
        거리 + 강도(페이스)를 기반으로 천천히 변하는 0~1 값
    - consistency_score:
        자연 감소 + 러닝할수록 증가 (롱런/강도 훈련일수록 더 보상)
    """
    skill = load_user_skill(user_id)

    # --- 입력 값 파싱 ---
    distance = float(new_run.get("distance", 0.0))
    pace_total = float(new_run.get("pace_sec", skill["avg_pace_sec"]))  # 없으면 기존 평균으로
    time_sec = float(new_run.get("time_sec", distance * pace_total))    # 대충 근사 (옵션)

    # ---------------------------
    # 1) avg pace 업데이트 (거리 가중 지수평활)
    # ---------------------------
    # 기본 alpha: 0.2, 하지만 거리 5km 이상이면 영향을 더 키운다.
    base_alpha = 0.2
    distance_factor = min(1.5, max(0.3, distance / 5.0))  # 5km 기준, 0.3~1.5 배
    alpha = base_alpha * distance_factor

    prev_pace = skill["avg_pace_sec"]
    new_avg_pace = prev_pace * (1.0 - alpha) + pace_total * alpha
    skill["avg_pace_sec"] = new_avg_pace

    # ---------------------------
    # 2) 최대 거리 갱신
    # ---------------------------
    skill["max_distance"] = max(skill["max_distance"], distance)

    # ---------------------------
    # 3) weekly_distance (최근 1주 볼륨 근사)
    # ---------------------------
    # 지수평활: 주로 1~2주 정도의 러닝량을 반영하는 느낌으로
    # decay = 0.75 -> 새로운 거리 비중 25%
    weekly_decay = 0.75
    skill["weekly_distance"] = skill["weekly_distance"] * weekly_decay + distance * (1.0 - weekly_decay)

    # ---------------------------
    # 4) training_load (chronic load, 0~1)
    # ---------------------------
    # 거리 + 강도 기반 단순 세션 로드 계산
    # - 기준 페이스: 현재 avg_pace_sec
    # - avg보다 빠르면 강도↑, 느리면 강도↓
    if prev_pace > 0:
        intensity_ratio = prev_pace / pace_total  # 1보다 크면 평소보다 강한 강도
    else:
        intensity_ratio = 1.0

    # 강도는 0.6 ~ 1.4 사이로 클램프
    intensity_factor = max(0.6, min(1.4, intensity_ratio))

    # 세션 로드: (거리 / 10km) * intensity_factor 를 0~1로 클램프
    session_load = (distance / 10.0) * intensity_factor
    session_load = max(0.0, min(1.0, session_load))

    # training_load는 천천히 변하는 chronic load
    # 이전 값 90%, 이번 세션 10%
    load_decay = 0.9
    training_load = skill["training_load"] * load_decay + session_load * (1.0 - load_decay)
    training_load = max(0.0, min(1.0, training_load))
    skill["training_load"] = training_load
    skill["fatigue_level"] = training_load  # 호환용 alias

    # ---------------------------
    # 5) consistency_score (0~1)
    # ---------------------------
    # 자연 감소 + 러닝하면 증가.
    # - 기본적으로 매 호출마다 조금씩 감소 (쉬는 날 가정)
    # - distance >= 1km 이면 의미 있는 러닝으로 보고 보상
    # - weekly_distance와 training_load가 너무 낮으면 약간 패널티
    consistency = float(skill["consistency_score"])

    # 하루 지나면서 자연 감소 (exponential decay)
    consistency *= 0.97  # 이전 97% 유지

    if distance >= 1.0:
        # 의미 있는 러닝: 기본 보상
        consistency += 0.03

        # 롱런(현재 max_distance의 60% 이상)일 경우 추가 보상
        if distance >= max(3.0, skill["max_distance"] * 0.6):
            consistency += 0.02
    else:
        # 스트레칭, 짧은 조깅 등이라고 가정하고 아주 약간만 보상
        if distance > 0:
            consistency += 0.005

    # 주간 볼륨이 너무 낮고(training_load도 낮으면) 습관이 떨어지는 중이라고 판단
    if skill["weekly_distance"] < 5.0 and training_load < 0.2:
        consistency -= 0.01

    # 클램프
    consistency = max(0.0, min(1.0, consistency))
    skill["consistency_score"] = consistency

    # ---------------------------
    # 저장 및 반환
    # ---------------------------
    save_user_skill(user_id, skill)
    return skill

def get_user_static_features(user_id: int) -> dict:
    """
    LSTM 입력용 유저 정적 특성(나이, 키, 몸무게)을 반환.
    - profile 에 값이 있으면 우선
    - 없으면 skill 파일에 있는 값 사용
    - 둘 다 없으면 안전한 기본값 (30/170/65) 사용
    """
    profile = load_user_profile(user_id)
    skill = load_user_skill(user_id)

    def _pick(key: str):
        v = profile.get(key)
        if v in (None, 0, "0", ""):
            v = skill.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    age = _pick("age")
    height = _pick("height")
    weight = _pick("weight")

    if age is None:
        age = 30
    if height is None:
        height = 170
    if weight is None:
        weight = 65

    return {
        "age": age,
        "height": height,
        "weight": weight,
    }
