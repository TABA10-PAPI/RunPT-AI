# train_lstm.py
import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from config.settings import BASE_MODEL_PATH, RUNNING_DIR
from service.skill_service import get_user_static_features


WINDOW_SIZE = 7  # LSTM 입력창 크기(최근 7회 러닝 또는 7일)
FEATURE_DIM = 7  # distance, pace_sec, time_sec, avg_hr, age, height, weight

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV_DIR = BASE_DIR / "data" / "csv"


# ---------------------------
# 공통 Feature 추출
# ---------------------------
def _extract_features(record: Dict, static: Dict[str, float]) -> List[float]:
    """
    features = [distance, pace_sec, time_sec, avg_hr, age, height, weight]
    """
    distance = float(record.get("distance", 0.0))
    pace_sec = float(record.get("pace_sec", 360.0))
    time_sec = float(record.get("time_sec", distance * pace_sec))
    avg_hr = float(record.get("avg_hr", 140.0))

    age = float(static.get("age", 30))
    height = float(static.get("height", 170))
    weight = float(static.get("weight", 65))

    return [distance, pace_sec, time_sec, avg_hr, age, height, weight]


def _heuristic_battery_target(window: List[Dict]) -> float:
    """
    규칙 기반 dummy target. 베이스 모델 사전학습용.
    """
    total_dist_7d = sum(float(r.get("distance", 0.0)) for r in window)

    last = window[-1]
    last_dist = float(last.get("distance", 0.0))
    last_hr = float(last.get("avg_hr", 140.0))

    fatigue_7d = min(1.0, total_dist_7d / 50.0)
    fatigue_last = min(1.0, last_dist / 20.0)
    fatigue_hr = min(1.0, max(0.0, (last_hr - 120.0) / 60.0))

    fatigue = 0.4 * fatigue_7d + 0.4 * fatigue_last + 0.2 * fatigue_hr
    fatigue = max(0.0, min(1.0, fatigue))

    battery = 0.9 - 0.7 * fatigue
    battery = max(0.1, min(0.95, battery))
    return float(battery)


# ---------------------------
# JSON 러닝 데이터 로딩
# ---------------------------
def _load_all_running_sessions() -> List[Tuple[Optional[int], List[Dict]]]:
    """
    RUNNING_DIR/user_*.json 파일 모두 읽어서
    (user_id, 세션 리스트) 형태로 반환
    """
    running_dir = Path(RUNNING_DIR)
    if not running_dir.exists():
        return []

    sessions: List[Tuple[Optional[int], List[Dict]]] = []
    for path in running_dir.glob("user_*.json"):
        try:
            stem = path.stem  # 예: "user_123"
            user_id: Optional[int] = None
            if "_" in stem:
                uid_part = stem.split("_", 1)[1]
                try:
                    user_id = int(uid_part)
                except ValueError:
                    user_id = None

            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                sessions.append((user_id, data))
        except Exception as e:
            print(f"[TRAIN] Failed reading {path}: {e}")
    return sessions


def build_dataset_from_running(window_size: int = WINDOW_SIZE):
    sessions = _load_all_running_sessions()
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for user_id, session in sessions:
        if len(session) < window_size:
            continue

        # 유저 정적 특성 (age/height/weight) 로드
        if user_id is not None:
            static = get_user_static_features(user_id)
        else:
            static = {"age": 30, "height": 170, "weight": 65}

        for i in range(len(session) - window_size + 1):
            window = session[i: i + window_size]
            feat_window = [_extract_features(r, static) for r in window]
            X_list.append(np.array(feat_window, dtype="float32"))
            y_list.append(np.array([_heuristic_battery_target(window)], dtype="float32"))

    if not X_list:
        print("[TRAIN] No valid windows from JSON.")
        return None, None

    X = np.stack(X_list, axis=0)          # (N, WINDOW_SIZE, FEATURE_DIM)
    y = np.stack(y_list, axis=0)          # (N, 1)
    print(f"[TRAIN] JSON dataset: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------
# CSV 데이터 로딩
# ---------------------------
def _load_csv_sequences() -> List[Tuple[Optional[int], List[Dict]]]:
    """
    CSV에서 user_id별로 레코드를 모아서
    (user_id, seq) 리스트로 반환
    """
    sequences_by_user: Dict[str, List[Dict]] = {}

    if not TRAIN_CSV_DIR.exists():
        return []

    for path in TRAIN_CSV_DIR.glob("*.csv"):
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row.get("user_id", "default")
                    sequences_by_user.setdefault(user_id, []).append(row)
        except Exception as e:
            print(f"[TRAIN] CSV load error {path}: {e}")

    result: List[Tuple[Optional[int], List[Dict]]] = []
    for uid_str, seq in sequences_by_user.items():
        # 날짜 정렬
        if seq and "date" in seq[0]:
            seq.sort(key=lambda r: r["date"])

        try:
            user_id: Optional[int] = int(uid_str)
        except (TypeError, ValueError):
            user_id = None

        result.append((user_id, seq))

    return result


def _get_csv_target(window: List[Dict]) -> Optional[float]:
    """
    battery_label → float normalized (0~1)
    """
    last = window[-1]
    val = (
        last.get("battery_label")
        or last.get("battery")
        or last.get("target_battery")
    )

    if val is None:
        return None

    try:
        v = float(val)
    except Exception:
        return None

    if v > 1.5:  # 0~100 scale
        v = v / 100.0

    return max(0.0, min(1.0, v))


def build_dataset_from_csv(window_size: int = WINDOW_SIZE):
    sequences = _load_csv_sequences()
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for user_id, seq in sequences:
        if len(seq) < window_size:
            continue

        # 유저 정적 특성 (age/height/weight) 로드
        if user_id is not None:
            static = get_user_static_features(user_id)
        else:
            static = {"age": 30, "height": 170, "weight": 65}

        for i in range(len(seq) - window_size + 1):
            window = seq[i: i + window_size]

            records = []
            for row in window:
                rec = {
                    "distance": float(row.get("distance", 0.0)),
                    "pace_sec": float(row.get("pace_sec", 360.0)),
                    "time_sec": float(
                        row.get(
                            "time_sec",
                            float(row.get("distance", 0.0)) * float(row.get("pace_sec", 360.0))
                        )
                    ),
                    "avg_hr": float(row.get("avg_hr", 140.0)),
                }
                records.append(rec)

            target = _get_csv_target(window)
            if target is None:
                continue

            feat_window = [_extract_features(r, static) for r in records]
            X_list.append(np.array(feat_window, dtype="float32"))
            y_list.append(np.array([target], dtype="float32"))

    if not X_list:
        print("[TRAIN] No CSV windows found.")
        return None, None

    X = np.stack(X_list, axis=0)      # (N, WINDOW_SIZE, FEATURE_DIM)
    y = np.stack(y_list, axis=0)      # (N, 1)
    print(f"[TRAIN] CSV dataset: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------
# 메인 학습 로직
# ---------------------------
def train_sample_lstm():
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    # JSON 데이터 우선 사용
    X_run, y_run = build_dataset_from_running()
    if X_run is not None:
        X_list.append(X_run)
        y_list.append(y_run)

    # CSV 부족하면 보충
    X_csv, y_csv = build_dataset_from_csv()
    if X_csv is not None:
        # JSON 데이터가 없거나, 샘플 수가 너무 적으면 CSV도 추가
        if not X_list or X_run.shape[0] < 200:
            X_list.append(X_csv)
            y_list.append(y_csv)

    # 최종 dataset 구성
    if X_list:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        print(f"[TRAIN] Final dataset X={X.shape}, y={y.shape}")
    else:
        print("[TRAIN] No real data → fallback to random")
        X = np.random.rand(200, WINDOW_SIZE, FEATURE_DIM).astype("float32")
        y = np.random.rand(200, 1).astype("float32")

    # ---------------------------
    # 모델 정의 (확장)
    # ---------------------------
    model = tf.keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, FEATURE_DIM)),
        layers.LSTM(64, return_sequences=False),   # 32 → 64로 확대
        layers.Dense(32, activation="relu"),       # 16 → 32로 확대
        layers.Dense(1, activation="sigmoid")
    ])

    # MAE 메트릭 + MSE 손실
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # EarlyStopping 콜백 설정
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # 학습 (validation_split + EarlyStopping 적용)
    history = model.fit(
        X,
        y,
        epochs=100,             # 5 → 50으로 증가
        batch_size=16,
        validation_split=0.2,  # 8:2로 train/valid 분리
        callbacks=[early_stop],
        verbose=1,
    )

    # 마지막 loss/val_loss 로그 찍어주기 (디버깅용)
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    print(f"[TRAIN] Final loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")

    model.save(BASE_MODEL_PATH)
    print(f"[TRAIN] Saved model to {BASE_MODEL_PATH}")


if __name__ == "__main__":
    train_sample_lstm()
