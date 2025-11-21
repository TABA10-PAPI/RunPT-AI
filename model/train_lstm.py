# 실제 훈련 시 네가 CSV 기반으로 데이터 넣으면 됨
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from config.settings import BASE_MODEL_PATH

def train_sample_lstm():
    X = np.random.rand(200, 7, 4)   # 7일 × 4개 feature
    y = np.random.rand(200, 1)

    model = tf.keras.Sequential([
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # 0~1 → 배터리 비율
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16)
    model.save(BASE_MODEL_PATH)
    print("Base LSTM model trained & saved.")

if __name__ == "__main__":
    train_sample_lstm()
