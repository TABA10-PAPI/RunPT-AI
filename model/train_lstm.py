# model/train_lstm.py
import numpy as np
import tensorflow as tf
from pathlib import Path

# 윈도우/피처 설정
WINDOW_SIZE = 7
FEATURE_DIM = 6  # hr, hrv, pace, sleep_hours, distance_km, calories
NUM_SAMPLES = 1000


def generate_dummy_data():
    X = np.random.rand(NUM_SAMPLES, WINDOW_SIZE, FEATURE_DIM).astype("float32")
    y = (np.random.rand(NUM_SAMPLES, 1) * 100.0).astype("float32")  # 0~100
    return X, y


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_SIZE, FEATURE_DIM)),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear", name="battery_score"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    X, y = generate_dummy_data()
    model = build_model()
    model.summary()

    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

    # model/ 디렉토리에 저장
    model_dir = Path(__file__).resolve().parent
    save_path = model_dir / "battery_lstm.h5"
    model.save(save_path)
    print(f"✅ 모델 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
