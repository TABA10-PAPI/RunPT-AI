import random

def format_pace(sec):
    return f"{int(sec//60)}:{int(sec%60):02d}"

def generate_recommendations(skill: dict, battery_score: float):
    recs = []

    avg_pace = skill.get("avg_pace_sec", 360)
    max_distance = skill.get("max_distance", 5.0)
    weekly_distance = skill.get("weekly_distance", 0.0)
    fatigue = skill.get("fatigue_level", 0.3)
    consistency = skill.get("consistency_score", 0.5)

    # 1) 러너 레벨 분류 (대략적인 예시)
    #   - 평균 페이스 + 최대 거리 기반
    if max_distance < 5 and avg_pace > 7 * 60:
        level = "beginner"
    elif max_distance < 10 and avg_pace > 6 * 60:
        level = "novice"
    elif max_distance < 15 or avg_pace > 5 * 60:
        level = "intermediate"
    else:
        level = "advanced"

    # 2) 레벨별 기본 거리 범위 설정
    if level == "beginner":
        easy_min, easy_max = 3, 6
        quality_min, quality_max = 3, 6
    elif level == "novice":
        easy_min, easy_max = 5, 8
        quality_min, quality_max = 5, 8
    elif level == "intermediate":
        easy_min, easy_max = 7, 12
        quality_min, quality_max = 6, 10
    else:  # advanced
        easy_min, easy_max = 10, 18
        quality_min, quality_max = 8, 14

    # 3) 피로도/배터리 기반 거리 조정 factor
    #   - 피로도 ↑ → 거리 감소, 배터리 ↑ → 거리 증가
    fatigue = max(0.0, min(1.0, fatigue))
    battery_score = max(0.0, min(100.0, battery_score))

    fatigue_factor = 1.0 - 0.4 * fatigue          # 0.6 ~ 1.0
    battery_factor = 0.6 + (battery_score / 100) * 0.6  # 0.6 ~ 1.2
    distance_factor = fatigue_factor * battery_factor   # 대략 0.36 ~ 1.2

    # 4) 페이스 오프셋 (레벨별 여유)
    if level in ["beginner", "novice"]:
        recovery_offset = 90   # +1분30초
        easy_offset = 60       # +1분
        steady_offset = 30     # +30초
        hard_offset = -10      # -10초
    elif level == "intermediate":
        recovery_offset = 75
        easy_offset = 45
        steady_offset = 20
        hard_offset = -15
    else:  # advanced
        recovery_offset = 60
        easy_offset = 30
        steady_offset = 10
        hard_offset = -20

    def clamp_pace(sec):
        # 너무 빠르거나 느린 비정상 값 방지 (3:30 ~ 10:00 사이로 제한)
        return max(3*60 + 30, min(10*60, sec))

    # -------------------------
    # 5) 배터리 상태에 따른 추천 패턴
    # -------------------------
    # (1) 배터리 낮음 → 회복 위주
    if battery_score < 40:
        easy_dist = round(random.uniform(easy_min, easy_max) * distance_factor, 1)
        recs.append({
            "type": "Recovery Run",
            "distance_km": max(3.0, easy_dist),
            "target_pace": format_pace(clamp_pace(avg_pace + recovery_offset)),
            "reason": f"피로도 {fatigue:.2f}, 배터리 {battery_score:.0f} → 회복이 우선"
        })

        easy_dist2 = round(random.uniform(easy_min, easy_max) * distance_factor, 1)
        recs.append({
            "type": "Easy Run",
            "distance_km": max(3.0, easy_dist2),
            "target_pace": format_pace(clamp_pace(avg_pace + easy_offset)),
            "reason": "가벼운 페이스로 기초 체력 유지"
        })

    # (2) 배터리 중간 → 이지 + 조깅/빌드업
    elif battery_score < 70:
        easy_dist = round(random.uniform(easy_min, easy_max) * distance_factor, 1)
        recs.append({
            "type": "Easy Run",
            "distance_km": max(4.0, easy_dist),
            "target_pace": format_pace(clamp_pace(avg_pace + easy_offset)),
            "reason": "컨디션 중간 수준 → 무리하지 않는 범위에서 거리 확보"
        })

        steady_dist = round(random.uniform(quality_min, quality_max) * distance_factor, 1)
        recs.append({
            "type": "Jogging / Steady Run",
            "distance_km": max(5.0, steady_dist),
            "target_pace": format_pace(clamp_pace(avg_pace + steady_offset)),
            "reason": "지구력 강화 + 현재 페이스 유지"
        })

        # 레벨이 intermediate 이상이면 빌드업도 제안
        if level in ["intermediate", "advanced"]:
            build_dist = round(random.uniform(quality_min, quality_max) * distance_factor, 1)
            recs.append({
                "type": "Build-up Run",
                "distance_km": max(6.0, build_dist),
                "target_pace": format_pace(clamp_pace(avg_pace)),  # 후반에만 빠르게
                "reason": "후반으로 갈수록 페이스를 올리는 빌드업 훈련"
            })

    # (3) 배터리 높음 → 템포/인터벌/LSD
    else:
        tempo_dist = round(random.uniform(quality_min, quality_max) * distance_factor, 1)
        recs.append({
            "type": "Tempo Run",
            "distance_km": max(5.0, tempo_dist),
            "target_pace": format_pace(clamp_pace(avg_pace + hard_offset)),
            "reason": "상태 매우 좋음 → 젖산역치 향상을 위한 템포런 적합"
        })

        # 인터벌 (고강도 but 전체 거리는 짧게)
        interval_total_dist = max(4.0, min(8.0, quality_min * distance_factor))
        recs.append({
            "type": "Interval",
            "distance_km": round(interval_total_dist, 1),
            "target_pace": format_pace(clamp_pace(avg_pace + hard_offset - 10)),
            "reason": "VO2max 향상을 위한 고강도 인터벌 세션"
        })

        # LSD는 주간 거리가 어느 정도 되는 러너에게만
        if weekly_distance > 20 and level in ["intermediate", "advanced"]:
            lsd_base = max(12.0, min(24.0, max_distance * 1.2))
            lsd_dist = round(lsd_base * distance_factor, 1)
            recs.append({
                "type": "LSD (Long Slow Distance)",
                "distance_km": lsd_dist,
                "target_pace": format_pace(clamp_pace(avg_pace + recovery_offset)),
                "reason": "장거리 지구력 강화를 위한 느리고 긴 거리 러닝"
            })

    # 6) 최소 3개까지만 전달 (너무 많으면 잘라냄)
    if len(recs) < 3:
        # 부족하면 Easy Run을 채워서 3개는 맞춰준다
        while len(recs) < 3:
            extra_dist = round(random.uniform(easy_min, easy_max) * distance_factor, 1)
            recs.append({
                "type": "Easy Run",
                "distance_km": max(3.0, extra_dist),
                "target_pace": format_pace(clamp_pace(avg_pace + easy_offset)),
                "reason": "전체 밸런스를 맞추기 위한 추가 이지런"
            })

    return recs[:3]
