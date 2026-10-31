# =========================================================
# 微電網 EMS 系統 - 智慧校園用電分析與 AI 預測平台 (app.py)
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

import json
from datetime import datetime
from pathlib import Path

# =========================================================
# 月度用電量預測：計算引擎
# =========================================================
ASSET_DIR = Path(__file__).resolve().parent  # 模型與設定檔存放位置

LOOKBACK_MONTHS = 12


# ---------------------------------------------------------
# 1. 資料整理與 5:1:1 切分
# ---------------------------------------------------------
def build_monthly_series(df, year_col, month_col, value_col, roc_year=True):
    d = df[[year_col, month_col, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    year = pd.to_numeric(d[year_col], errors="coerce")
    month = pd.to_numeric(d[month_col], errors="coerce")
    if roc_year:
        year = year + 1911
    d["_y"], d["_m"] = year, month
    d = d.dropna(subset=["_y", "_m", value_col])
    if d.empty:
        raise ValueError("沒有可用的「年／月／數值」資料，請確認欄位選擇是否正確")

    idx = pd.to_datetime(
        {"year": d["_y"].astype(int), "month": d["_m"].astype(int), "day": 1}
    )
    s = pd.Series(d[value_col].to_numpy(dtype=float), index=idx).sort_index()

    if s.index.duplicated().any():
        raise ValueError(f"有重複的月份：{list(s.index[s.index.duplicated()].strftime('%Y-%m'))}")
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    missing = full.difference(s.index)
    if len(missing) > 0:
        raise ValueError(f"缺少月份：{list(missing.strftime('%Y-%m'))}")
    if (s < 0).any():
        raise ValueError("資料中含有負值")
    s.index.freq = "MS"
    return s


def split_5_1_1(series):
    n = len(series)
    n_val = n // 7
    n_test = n // 7
    n_train = n - n_val - n_test
    if n_val < 1 or n_train <= LOOKBACK_MONTHS + 1:
        raise ValueError(f"資料只有 {n} 個月，不足以做 5:1:1 切分與 12 個月回看")
    return (
        series.iloc[:n_train],
        series.iloc[n_train:n_train + n_val],
        series.iloc[n_train + n_val:],
    )


# ---------------------------------------------------------
# 2. 縮放與特徵
# ---------------------------------------------------------
def _fit_scaler(train_values):
    lo, hi = float(np.min(train_values)), float(np.max(train_values))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return {"lo": lo, "hi": hi}


def _scale(x, sc):
    return (np.asarray(x, dtype=float) - sc["lo"]) / (sc["hi"] - sc["lo"])


def _unscale(x, sc):
    return np.asarray(x, dtype=float) * (sc["hi"] - sc["lo"]) + sc["lo"]


def _month_feats(index):
    ang = 2 * np.pi * np.asarray(index.month) / 12.0
    return np.column_stack([np.sin(ang), np.cos(ang)])


def _window_features(scaled_window, window_index):
    return np.column_stack([scaled_window, _month_feats(window_index)])


def make_samples(scaled, index, start, end, lookback=LOOKBACK_MONTHS):
    X, y = [], []
    for t in range(max(start, lookback), end):
        X.append(_window_features(scaled[t - lookback:t], index[t - lookback:t]))
        y.append(scaled[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------
# 3. 評估指標
# ---------------------------------------------------------
def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true != 0
    return {
        "MAPE_%": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
    }


# ---------------------------------------------------------
# 4. 模型：季節性基準、Holt-Winters、小型 LSTM
# ---------------------------------------------------------
def seasonal_naive_predict(values, positions):
    return np.array([values[t - 12] for t in positions])


def holt_winters_predict(values, positions):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    out = []
    for t in positions:
        fit = ExponentialSmoothing(
            values[:t], trend="add", seasonal="add",
            seasonal_periods=12, initialization_method="estimated",
        ).fit()
        out.append(float(fit.forecast(1)[0]))
    return np.array(out)


def build_lstm(lookback=LOOKBACK_MONTHS, n_features=3, units=16, dropout=0.2):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, n_features)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_lstm(X_tr, y_tr, X_va, y_va, epochs=500, seed=42):
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = build_lstm(lookback=X_tr.shape[1], n_features=X_tr.shape[2])
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=8, shuffle=True, callbacks=[stop], verbose=0)
    return model


# ---------------------------------------------------------
# 5. 完整實驗流程
# ---------------------------------------------------------
def run_experiment(series, out_dir=".", target_name="度數", unit="kWh",
                   use_holt_winters=True, use_lstm=True, epochs=500, seed=42,
                   save=True):
    n = len(series)
    train, val, test = split_5_1_1(series)
    n_train, n_val = len(train), len(val)
    values, index = series.to_numpy(dtype=float), series.index

    sc = _fit_scaler(train.to_numpy())
    scaled = _scale(values, sc)
    ranges = {"val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}

    preds = {"seasonal_naive": {}}
    for name, (a, b) in ranges.items():
        preds["seasonal_naive"][name] = seasonal_naive_predict(values, range(a, b))

    if use_holt_winters:
        try:
            preds["holt_winters"] = {
                name: holt_winters_predict(values, range(a, b))