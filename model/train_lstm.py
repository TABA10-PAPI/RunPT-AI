# train_lstm.py
import json
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from config.settings import BASE_MODEL_PATH, RUNNING_DIR

WINDOW_SIZE = 7  # LSTM 입력으로 사용할 일수 (7일 연속)
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV_DIR = BASE_DIR / "data" / "csv"   # CSV 학습 데이터 디렉토리


# ---------------------------
# 공통: feature & target 생성
# ---------------------------

def _extract_features(record: dict) -> list[float]:
    """
    예측에 사용할 feature를 추출.
    predict_battery에서 사용하는 것과 동일한 순서/구조로 맞춰야 함.

    features = [distance, pace_sec, time_sec, avg_hr]
    """
    distance = float(record.get("distance", 0.0))
    pace_sec = float(record.get("pace_sec", 360.0))  # 없으면 6분 페이스
    time_sec = float(record.get("time_sec", distance * pace_sec))
    avg_hr = float(record.get("avg_hr", 140.0))

    return [distance, pace_sec, time_sec, avg_hr]


def _heuristic_battery_target(window: list[dict]) -> float:
    """
    7일치 러닝 window를 보고, 규칙 기반으로 '가짜 배터리 점수(0~1)'를 만든다.
    - 7일 거리 합이 많을수록 피로 ↑ → 배터리 ↓
    - 마지막 날 거리가 길수록 배터리 ↓
    - 평균 심박이 높으면 배터리 ↓

    베이스 모델 pretrain 용 rough target.
    """
    total_dist_7d = sum(float(r.get("distance", 0.0)) for r in window)
    last = window[-1]
    last_dist = float(last.get("distance", 0.0))
    last_hr = float(last.get("avg_hr", 140.0))

    # 대략적인 피로도 점수 (0~1)
    fatigue_7d = min(1.0, total_dist_7d / 50.0)       # 7일 50km 이상이면 높은 편
    fatigue_last = min(1.0, last_dist / 20.0)         # 하루 20km 기준
    fatigue_hr = min(1.0, max(0.0, (last_hr - 120.0) / 60.0))  # 120~180 구간 정규화

    fatigue = 0.4 * fatigue_7d + 0.4 * fatigue_last + 0.2 * fatigue_hr
    fatigue = max(0.0, min(1.0, fatigue))

    # 피로도가 0일 때 ~0.9, 피로도가 1일 때 ~0.2 정도로 매핑
    battery = 0.9 - 0.7 * fatigue
    battery = max(0.1, min(0.95, battery))

    return float(battery)


# ---------------------------
# 1단계: JSON 러닝 기록으로부터 dataset 생성
# ---------------------------

def _load_all_running_sessions() -> list[list[dict]]:
    """
    RUNNING_DIR 아래의 user_*.json 파일들을 모두 읽어서
    각 유저별 러닝 기록 리스트를 반환.
    형태: [ [record, record, ...], [record, ...], ... ]
    """
    all_sessions = []

    running_dir = Path(RUNNING_DIR)
    if not running_dir.exists():
        return []

    for path in running_dir.glob("user_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                all_sessions.append(data)
        except Exception as e:
            print(f"[TRAIN] Failed to load {path}: {e}")
            continue

    return all_sessions


def build_dataset_from_running(window_size: int = WINDOW_SIZE):
    """
    실제 서비스에서 쌓인 JSON 러닝 기록으로부터
    (num_samples, window_size, 4) X, (num_samples, 1) y를 생성.
    """
    all_sessions = _load_all_running_sessions()
    X_list: list[list[list[float]]] = []
    y_list: list[float] = []

    for session in all_sessions:
        if len(session) < window_size:
            continue

        for i in range(len(session) - window_size + 1):
            window = session[i : i + window_size]
            x = [_extract_features(r) for r in window]
            y = _heuristic_battery_target(window)

            X_list.append(x)
            y_list.append(y)

    if not X_list:
        print("[TRAIN] No valid running windows found from JSON.")
        return None, None

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32").reshape(-1, 1)
    print(f"[TRAIN] Built dataset from JSON running logs: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------
# 2단계: CSV 파일로부터 dataset 생성
# ---------------------------

def _load_csv_sequences() -> list[list[dict]]:
    """
    TRAIN_CSV_DIR 안의 모든 CSV 파일을 읽어서
    user_id 기준으로 시퀀스를 만들어 반환.
    CSV 형식 예시:
    user_id,date,distance,pace_sec,time_sec,avg_hr,battery_label
    """
    sequences_by_user: dict[str, list[dict]] = {}

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
            print(f"[TRAIN] Failed to load CSV {path}: {e}")
            continue

    # date 칼럼 기준 정렬 (있을 경우)
    for uid, seq in sequences_by_user.items():
        if seq and "date" in seq[0]:
            seq.sort(key=lambda r: r["date"])

    return list(sequences_by_user.values())


def _get_csv_target(window: list[dict]) -> float | None:
    """
    CSV 윈도우에서 타깃 배터리 값을 추출.
    - battery_label (0~1 또는 0~100) 기준
    """
    last = window[-1]
    val = last.get("battery_label") or last.get("battery") or last.get("target_battery")
    if val is None:
        return None

    try:
        v = float(val)
    except ValueError:
        return None

    if v > 1.5:  # 0~100 스케일이라고 가정
        v = v / 100.0

    v = max(0.0, min(1.0, v))
    return v


def build_dataset_from_csv(window_size: int = WINDOW_SIZE):
    """
    TRAIN_CSV_DIR 아래 CSV 파일들로부터
    (num_samples, window_size, 4) X, (num_samples, 1) y를 생성.
    """
    sequences = _load_csv_sequences()
    X_list: list[list[list[float]]] = []
    y_list: list[float] = []

    for seq in sequences:
        if len(seq) < window_size:
            continue

        for i in range(len(seq) - window_size + 1):
            window = seq[i : i + window_size]

            # feature 구성
            records = []
            for row in window:
                rec = {
                    "distance": float(row.get("distance", 0.0)),
                    "pace_sec": float(row.get("pace_sec", 360.0)),
                    "time_sec": float(row.get("time_sec", float(row.get("distance", 0.0)) * float(row.get("pace_sec", 360.0)))),
                    "avg_hr": float(row.get("avg_hr", 140.0)),
                }
                records.append(rec)

            target = _get_csv_target(window)
            if target is None:
                continue

            x = [_extract_features(r) for r in records]
            X_list.append(x)
            y_list.append(target)

    if not X_list:
        print("[TRAIN] No valid training windows from CSV.")
        return None, None

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32").reshape(-1, 1)
    print(f"[TRAIN] Built dataset from CSV files: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------
# 메인 학습 함수
# ---------------------------

def train_sample_lstm():
    """
    학습 우선순위:
    1) 실제 JSON 러닝 데이터 (RUNNING_DIR/user_*.json)
    2) 부족하면 data/csv/*.csv 사용
    3) 그래도 없으면 랜덤 데이터로 최소한의 베이스 모델 생성
    """
    X_list = []
    y_list = []

    # 1) JSON 러닝 데이터
    X_run, y_run = build_dataset_from_running(window_size=WINDOW_SIZE)
    if X_run is not None and y_run is not None:
        X_list.append(X_run)
        y_list.append(y_run)

    # 2) CSV 데이터 (JSON 데이터가 없거나 적을 때 보충)
    X_csv, y_csv = build_dataset_from_csv(window_size=WINDOW_SIZE)
    if X_csv is not None and y_csv is not None:
        # JSON 데이터가 너무 적으면 같이 합쳐서 사용
        if not X_list or X_run.shape[0] < 200:
            X_list.append(X_csv)
            y_list.append(y_csv)

    # 3) 최종 X, y 구성
    if X_list:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        print(f"[TRAIN] Final training dataset: X={X.shape}, y={y.shape}")
    else:
        print("[TRAIN] No real training data found. Fallback to random data training.")
        X = np.random.rand(200, WINDOW_SIZE, 4).astype("float32")
        y = np.random.rand(200, 1).astype("float32")

    # 모델 정의 및 학습
    model = tf.keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, 4)),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # 0~1 → 배터리 비율
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16)
    model.save(BASE_MODEL_PATH)
    print(f"[TRAIN] Base LSTM model trained & saved at {BASE_MODEL_PATH}")


if __name__ == "__main__":
    train_sample_lstm()
