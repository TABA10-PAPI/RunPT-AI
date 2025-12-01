# train_lstm.py
import json
import csv
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from config.settings import BASE_MODEL_PATH, RUNNING_DIR

WINDOW_SIZE = 7  # LSTM 입력창 크기(최근 7회 러닝 또는 7일)
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV_DIR = BASE_DIR / "data" / "csv"


# ---------------------------
# 공통 Feature 추출
# ---------------------------
def _extract_features(record: Dict) -> List[float]:
    """
    features = [distance, pace_sec, time_sec, avg_hr]
    """
    distance = float(record.get("distance", 0.0))
    pace_sec = float(record.get("pace_sec", 360.0))
    time_sec = float(record.get("time_sec", distance * pace_sec))
    avg_hr = float(record.get("avg_hr", 140.0))

    return [distance, pace_sec, time_sec, avg_hr]


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
def _load_all_running_sessions() -> List[List[Dict]]:
    """
    RUNNING_DIR/user_*.json 파일 모두 읽어서 세션 리스트 반환
    """
    running_dir = Path(RUNNING_DIR)
    if not running_dir.exists():
        return []

    sessions = []
    for path in running_dir.glob("user_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                sessions.append(data)
        except Exception as e:
            print(f"[TRAIN] Failed reading {path}: {e}")
    return sessions


def build_dataset_from_running(window_size: int = WINDOW_SIZE):
    sessions = _load_all_running_sessions()
    X_list = []
    y_list = []

    for session in sessions:
        if len(session) < window_size:
            continue

        for i in range(len(session) - window_size + 1):
            window = session[i: i + window_size]
            X_list.append([_extract_features(r) for r in window])
            y_list.append(_heuristic_battery_target(window))

    if not X_list:
        print("[TRAIN] No valid windows from JSON.")
        return None, None

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32").reshape(-1, 1)
    print(f"[TRAIN] JSON dataset: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------
# CSV 데이터 로딩
# ---------------------------
def _load_csv_sequences() -> List[List[Dict]]:
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

    # 날짜 정렬
    for uid, seq in sequences_by_user.items():
        if seq and "date" in seq[0]:
            seq.sort(key=lambda r: r["date"])

    return list(sequences_by_user.values())


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
    except:
        return None

    if v > 1.5:  # 0~100 scale
        v = v / 100.0

    return max(0.0, min(1.0, v))


def build_dataset_from_csv(window_size: int = WINDOW_SIZE):
    sequences = _load_csv_sequences()
    X_list = []
    y_list = []

    for seq in sequences:
        if len(seq) < window_size:
            continue

        for i in range(len(seq) - window_size + 1):
            window = seq[i: i + window_size]

            records = []
            for row in window:
                rec = {
                    "distance": float(row.get("distance", 0.0)),
                    "pace_sec": float(row.get("pace_sec", 360.0)),
                    "time_sec": float(
                        row.get("time_sec",
                                float(row.get("distance", 0.0)) * float(row.get("pace_sec", 360.0)))
                    ),
                    "avg_hr": float(row.get("avg_hr", 140.0)),
                }
                records.append(rec)

            target = _get_csv_target(window)
            if target is None:
                continue

            X_list.append([_extract_features(r) for r in records])
            y_list.append(target)

    if not X_list:
        print("[TRAIN] No CSV windows found.")
        return None, None

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32").reshape(-1, 1)
    print(f"[TRAIN] CSV dataset: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------
# 메인 학습 로직
# ---------------------------
def train_sample_lstm():
    X_list = []
    y_list = []

    # JSON 데이터 우선 사용
    X_run, y_run = build_dataset_from_running()
    if X_run is not None:
        X_list.append(X_run)
        y_list.append(y_run)

    # CSV 부족하면 보충
    X_csv, y_csv = build_dataset_from_csv()
    if X_csv is not None:
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
        X = np.random.rand(200, WINDOW_SIZE, 4).astype("float32")
        y = np.random.rand(200, 1).astype("float32")

    # 모델 정의
    model = tf.keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, 4)),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16)
    model.save(BASE_MODEL_PATH)

    print(f"[TRAIN] Saved model to {BASE_MODEL_PATH}")


if __name__ == "__main__":
    train_sample_lstm()
