# =========================================================
# 微電網 EMS 系統 - 負載預測模型訓練腳本 (train.py)
# =========================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def build_training_pipeline():
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / 'model_config_v2_fixed.json'
    model_output_path = base_dir / 'lstm_model_brain_v2_fixed.h5'
    scaler_output_path = base_dir / 'data_scaler_v2_fixed.pkl'

    # 1. 讀取設定檔
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到設定檔: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    lookback_steps = config.get('lookback_steps', 192)
    steps_ahead = config.get('steps_ahead', 96)
    
    print(f"[*] 讀取設定成功：Lookback={lookback_steps}, Steps Ahead={steps_ahead}")

    # 2. 準備示範或載入歷史電力數據
    # 註：這裡使用專案中的 sample_history_kw.json 或建立模擬數據示範
    sample_history_path = base_dir / 'sample_history_kw.json'
    if os.path.exists(sample_history_path):
        with open(sample_history_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            # 假設內容為 list 或 dict，抓出數值陣列
            if isinstance(raw_data, dict) and 'load_kw' in raw_data:
                series_data = raw_data['load_kw']
            elif isinstance(raw_data, list):
                series_data = raw_data
            else:
                series_data = list(raw_data.values())[0]
    else:
        # 若無歷史檔則生成模擬正弦波電力負載數據供測試用
        print("[!] 找不到 sample_history_kw.json，正在生成模擬電力負載資料...")
        time_idx = np.arange(2000)
        series_data = (50 + 20 * np.sin(2 * np.pi * time_idx / 96) + np.random.normal(0, 2, size=2000)).tolist()

    raw_array = np.array(series_data, dtype=np.float64).reshape(-1, 1)

    # 3. 數據正規化 (MinMaxScaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_array = scaler.fit_transform(raw_array)

    # 4. 建立滑動視窗 (Rolling Window) X 與 Y (支援多步輸出 96 步)
    X, y = [], []
    for i in range(len(scaled_array) - lookback_steps - steps_ahead + 1):
        X.append(scaled_array[i : i + lookback_steps, 0])
        y.append(scaled_array[i + lookback_steps : i + lookback_steps + steps_ahead, 0])

    X = np.array(X)
    y = np.array(y)

    # 調整 LSTM 輸入維度 (Samples, Timesteps, Features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print(f"[*] 資料切片完成：X shape = {X.shape}, y shape = {y.shape}")

    # 5. 依時序切分法 (Chronological Data Split: 80% 訓練, 20% 驗證)
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # 6. 建構 LSTM 模型架構 (輸出層改為 96 步)
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(lookback_steps, 1), return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(steps_ahead)  # 輸出 96 個節點，對應未來 24 小時
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # 7. 模型訓練 (配合 EarlyStopping 防止過擬合)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    print("[*] 開始執行模型訓練...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # 8. 儲存新模型與 Scaler
    model.save(model_output_path)
    joblib.dump(scaler, scaler_output_path)

    print(f"[✔] 訓練完成！")
    print(f" - 新模型已儲存至: {model_output_path}")
    print(f" - 新 Scaler 已儲存至: {scaler_output_path}")

if __name__ == '__main__':
    build_training_pipeline()