# service/recommendation_engine.py
def format_pace(sec: float) -> str:
    """초 단위 페이스를 'M:SS' 형식으로 변환."""
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_distance(base_km: float, weekly_distance: float,
                     training_load: float, battery_score: float,
                     acute_fatigue: float, max_distance: float):
    """
    랜덤 없이 '결정적(deterministic)' 방식으로 거리 계산.
    스포츠 과학 기반:
      - acute_fatigue ↑ → 거리 감소
      - chronic load(training_load) ↑ → 거리 증가
      - battery_score ↑ → 거리 증가
      - 주간 거리 증가율 제한(부상 방지)
    """

    # 1) scaling factor 계산
    fatigue_factor = 1.0 - 0.5 * acute_fatigue            # 0.5 ~ 1.0
    battery_factor = 0.7 + 0.6 * (battery_score / 100.0)  # 0.7 ~ 1.3
    chronic_factor = 0.7 + 0.6 * training_load            # 0.7 ~ 1.3

    distance_factor = fatigue_factor * battery_factor * chronic_factor
    distance_factor = _clamp(distance_factor, 0.4, 1.6)

    # 2) base_km × factor
    dist = base_km * distance_factor

    # 3) 주간 부하 관리 (최대 증가율 25%)
    max_increase = max(2.0, weekly_distance * 0.25)  # 최소 2km 증가 허용
    dist = min(dist, weekly_distance + max_increase)

    # 4) 최대거리 기반 제한
    dist = min(dist, max_distance * 1.3)

    return round(max(3.0, dist), 1)


def generate_recommendations(skill: dict, battery_score: float, acute_fatigue: float):
    """
    랜덤 없는 최종 추천 엔진:
    - 거리 결정은 compute_distance()로 deterministic하게 계산
    - 페이스는 skill 기반 offset
    - 강도 구간은 배터리 + acute 피로 + consistency 기반 결정
    """

    avg_pace = float(skill.get("avg_pace_sec", 360))
    max_distance = float(skill.get("max_distance", 5.0))
    weekly_distance = float(skill.get("weekly_distance", 0.0))
    training_load = float(skill.get("training_load", 0.2))
    consistency = float(skill.get("consistency_score", 0.5))

    battery_score = _clamp(battery_score, 0, 100)
    acute_fatigue = _clamp(acute_fatigue, 0, 1)
    training_load = _clamp(training_load, 0, 1)
    consistency = _clamp(consistency, 0, 1)

    # ---------------------------
    # 1) 레벨 분류 (chronic load 중심)
    # ---------------------------
    if training_load < 0.15:
        level = "beginner"
    elif training_load < 0.30:
        level = "novice"
    elif training_load < 0.55:
        level = "intermediate"
    else:
        level = "advanced"

    # 보조 조절
    if level == "advanced" and max_distance < 15 and avg_pace > 5 * 60:
        level = "intermediate"

    # ---------------------------
    # 2) level별 base 거리 테이블
    # ---------------------------
    level_base = {
        "beginner":      {"easy": 4, "steady": 5, "quality": 5},
        "novice":        {"easy": 6, "steady": 7, "quality": 7},
        "intermediate":  {"easy": 8, "steady": 10, "quality": 10},
        "advanced":      {"easy": 12, "steady": 14, "quality": 14},
    }

    easy_base = level_base[level]["easy"]
    steady_base = level_base[level]["steady"]
    quality_base = level_base[level]["quality"]

    # ---------------------------
    # 3) 페이스 오프셋
    # ---------------------------
    if level in ["beginner", "novice"]:
        rec_offset = 90
        easy_offset = 60
        steady_offset = 30
        hard_offset = -10
    elif level == "intermediate":
        rec_offset = 75
        easy_offset = 45
        steady_offset = 20
        hard_offset = -15
    else:
        rec_offset = 60
        easy_offset = 30
        steady_offset = 10
        hard_offset = -20

    # 피로 반영
    fatigue_scale = 1.0 + 0.5 * acute_fatigue
    rec_offset *= fatigue_scale
    easy_offset *= fatigue_scale
    steady_offset *= fatigue_scale
    hard_offset /= fatigue_scale

    def clamp_pace(sec):
        return _clamp(sec, 3*60+30, 10*60)

    # ---------------------------
    # 4) intensity zone 결정
    # ---------------------------
    if battery_score < 40:
        zone = "low"
    elif battery_score < 70:
        zone = "medium"
    else:
        zone = "high"

    # consistency 낮으면 zone 한 단계 다운
    if zone == "high" and consistency < 0.3:
        zone = "medium"

    # acute fatigue 큰 날은 강도 제한
    if zone == "high" and acute_fatigue > 0.6:
        zone = "medium"
    if zone == "medium" and acute_fatigue > 0.8:
        zone = "low"

    recs = []

    # ---------------------------
    # 5) zone별 추천 생성
    # ---------------------------

    # (1) 회복 위주
    if zone == "low":
        easy_dist = compute_distance(
            easy_base, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
        )
        recs.append({
            "type": "Recovery Run",
            "distance_km": easy_dist,
            "target_pace": format_pace(clamp_pace(avg_pace + rec_offset)),
            "reason": f"피로도 {acute_fatigue:.2f}, 배터리 {battery_score:.0f} → 회복 우선"
        })

        easy_dist2 = compute_distance(
            easy_base * 0.9, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
        )
        recs.append({
            "type": "Easy Run",
            "distance_km": easy_dist2,
            "target_pace": format_pace(clamp_pace(avg_pace + easy_offset)),
            "reason": "기초 체력 유지"
        })

        if consistency < 0.4:  # 습관이 약한 러너에게 보조 추천
            easy_dist3 = compute_distance(
                easy_base * 0.8, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
            )
            recs.append({
                "type": "Easy Run (Short)",
                "distance_km": easy_dist3,
                "target_pace": format_pace(clamp_pace(avg_pace + easy_offset)),
                "reason": "러닝 습관 형성을 위한 짧은 러닝"
            })

    # (2) 중간 강도
    elif zone == "medium":
        easy_dist = compute_distance(
            easy_base, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
        )
        recs.append({
            "type": "Easy Run",
            "distance_km": easy_dist,
            "target_pace": format_pace(clamp_pace(avg_pace + easy_offset)),
            "reason": "컨디션 중간 → 부담 없는 거리 확보"
        })

        steady_dist = compute_distance(
            steady_base, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
        )
        recs.append({
            "type": "Steady Run",
            "distance_km": steady_dist,
            "target_pace": format_pace(clamp_pace(avg_pace + steady_offset)),
            "reason": "지구력 강화"
        })

        # 고레벨 + 단기 피로 낮음 → build up 추천
        if training_load > 0.25 and acute_fatigue < 0.6:
            build_dist = compute_distance(
                steady_base, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
            )
            recs.append({
                "type": "Build-up Run",
                "distance_km": build_dist,
                "target_pace": format_pace(clamp_pace(avg_pace)),
                "reason": "후반에 페이스를 올리는 빌드업"
            })

    # (3) 고강도 허용
    else:  # zone == "high"
        tempo_dist = compute_distance(
            quality_base, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
        )
        recs.append({
            "type": "Tempo Run",
            "distance_km": tempo_dist,
            "target_pace": format_pace(clamp_pace(avg_pace + hard_offset)),
            "reason": "젖산역치 향상"
        })

        # 인터벌 (단기 피로가 낮을 때만)
        if acute_fatigue < 0.5:
            interval_km = compute_distance(
                quality_base * 0.6, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
            )
            recs.append({
                "type": "Interval",
                "distance_km": interval_km,
                "target_pace": format_pace(clamp_pace(avg_pace + hard_offset - 10)),
                "reason": "VO2max 향상"
            })

        # LSD (장거리; 장기 부하 충분 + 최근 피로 낮음)
        if training_load > 0.35 and weekly_distance > 15 and acute_fatigue < 0.6:
            base_lsd = max(12.0, quality_base + 4)
            lsd_dist = compute_distance(
                base_lsd, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
            )
            recs.append({
                "type": "LSD (Long Slow Distance)",
                "distance_km": lsd_dist,
                "target_pace": format_pace(clamp_pace(avg_pace + rec_offset)),
                "reason": "장거리 지구력 향상"
            })

    # ---------------------------
    # 6) 추천 최소 3개 보장
    # ---------------------------
    while len(recs) < 3:
        extra_dist = compute_distance(
            easy_base, weekly_distance, training_load, battery_score, acute_fatigue, max_distance
        )
        recs.append({
            "type": "Easy Run",
            "distance_km": extra_dist,
            "target_pace": format_pace(clamp_pace(avg_pace + easy_offset)),
            "reason": "훈련 밸런스를 위한 추가 러닝"
        })

    return recs[:3]

def generate_beginner_recommendations():
    """러닝 기록이 전혀 없는 사용자를 위한 기본 추천"""
    return [
        {
            "type": "Easy Run",
            "distance_km": 3.0,
            "target_pace": "7:00",
            "reason": "러닝이 처음인 사용자를 위한 가벼운 3km 러닝"
        },
        {
            "type": "Easy Run",
            "distance_km": 5.0,
            "target_pace": "8:00",
            "reason": "기초 체력 형성을 위한 여유로운 5km 러닝"
        }
    ]
