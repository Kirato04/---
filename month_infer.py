# =========================================================
# 微電網 EMS 系統 - 月度用電量／電費預測模組 (monthly_forecast.py)
# 版本 B：以「月」為單位，回看 12 個月，預測下個月（可滾動至多 12 個月）
# 使用真實月度資料，不使用正弦波還原
# =========================================================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK_MONTHS = 12


# ---------------------------------------------------------
# 1. 資料整理與 5:1:1 切分
# ---------------------------------------------------------
def build_monthly_series(df, year_col, month_col, value_col, roc_year=True):
    """
    把「年、月、數值」三欄整理成以月初為索引的 pd.Series。
    roc_year=True 代表年份是民國年（107 -> 2018）。
    缺月、重複月份、負值都會直接報錯，避免悄悄產生錯誤資料。
    """
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
    """依時間順序切成 訓練:驗證:測試 = 5:1:1（84 個月 -> 60 / 12 / 12）。"""
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
# 2. 縮放與特徵（縮放只用訓練集，避免資料洩漏）
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
    """月份的 sin/cos 編碼：這是日曆特徵，不是造假資料。"""
    ang = 2 * np.pi * np.asarray(index.month) / 12.0
    return np.column_stack([np.sin(ang), np.cos(ang)])


def _window_features(scaled_window, window_index):
    return np.column_stack([scaled_window, _month_feats(window_index)])


def make_samples(scaled, index, start, end, lookback=LOOKBACK_MONTHS):
    """目標位置 t 屬於 [start, end)，輸入是 t 之前真實的 12 個月。"""
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
# 4. 三種模型：季節性基準、Holt-Winters、小型 LSTM
# ---------------------------------------------------------
def seasonal_naive_predict(values, positions):
    """預測值 = 去年同月。這是 LSTM 至少要贏過的最低標準。"""
    return np.array([values[t - 12] for t in positions])


def holt_winters_predict(values, positions):
    """逐月滾動：每個目標月只用它之前的真實資料重新擬合。"""
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
# 5. 完整實驗流程：切分 -> 訓練 -> 驗證/測試 -> 存檔
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
                for name, (a, b) in ranges.items()
            }
        except Exception as e:
            print(f"[略過 Holt-Winters] {e}")

    model = None
    if use_lstm:
        try:
            X_tr, y_tr = make_samples(scaled, index, 0, n_train)
            X_va, y_va = make_samples(scaled, index, *ranges["val"])
            model = train_lstm(X_tr, y_tr, X_va, y_va, epochs=epochs, seed=seed)
            preds["lstm"] = {}
            for name, (a, b) in ranges.items():
                X, _ = make_samples(scaled, index, a, b)
                p = model.predict(X, verbose=0).reshape(-1)
                preds["lstm"][name] = _unscale(p, sc)
        except ImportError as e:
            print(f"[略過 LSTM] 沒有安裝 tensorflow：{e}")

    metrics = {
        m: {name: evaluate(values[slice(*ranges[name])], p[name]) for name in ranges}
        for m, p in preds.items()
    }
    best = min(metrics, key=lambda m: metrics[m]["val"]["MAPE_%"])

    frame = pd.DataFrame({"actual": values[n_train:]}, index=index[n_train:])
    frame["split"] = ["val"] * n_val + ["test"] * (n - n_train - n_val)
    for m, p in preds.items():
        frame[m] = np.concatenate([p["val"], p["test"]])

    config = {
        "model_version": "monthly_v1",
        "lookback_months": LOOKBACK_MONTHS,
        "target_name": target_name,
        "unit": unit,
        "scaler": sc,
        "split_months": {"train": n_train, "val": n_val, "test": n - n_train - n_val},
        "best_model_on_val": best,
        "metrics": metrics,
    }
    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "monthly_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if model is not None:
            model.save(str(out / "monthly_model.keras"))

    return {"metrics": metrics, "best_model": best, "predictions": frame, "config": config}


# ---------------------------------------------------------
# 6. 推論與防禦（月版）
# ---------------------------------------------------------
class MonthlyEMSPredictor:
    def __init__(self, model_path="monthly_model.keras", config_path="monthly_config.json"):
        base = Path(__file__).resolve().parent
        self.model_path = base / model_path
        self.config_path = base / config_path
        self.model = None
        self.scaler = None
        self.config = {}
        self.lookback = LOOKBACK_MONTHS
        self.unit = "kWh"
        self.model_version = "unknown"
        self.is_ready = False
        self.init_error = None
        self.stats = {
            "total_requests": 0,
            "errors": {
                "timestamp_error": 0, "horizon_error": 0, "dimension_error": 0,
                "length_insufficient": 0, "invalid_input": 0,
                "negative_value_blocked": 0, "model_not_ready": 0,
                "model_output_invalid": 0, "unexpected_exception": 0,
            },
            "fallback_used": {"seasonal_naive": 0, "none": 0},
        }
        self._load()

    def _load(self):
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"找不到設定檔: {self.config_path}")
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.lookback = int(self.config.get("lookback_months", LOOKBACK_MONTHS))
            self.scaler = self.config["scaler"]
            self.unit = self.config.get("unit", "kWh")
            self.model_version = self.config.get("model_version", "monthly_v1")
            if not self.model_path.exists():
                raise FileNotFoundError(f"找不到模型檔: {self.model_path}")
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.is_ready = True
        except Exception as e:
            self.is_ready = False
            self.init_error = f"元件載入失敗: {e}"

    @staticmethod
    def _parse_month(x):
        if x is None:
            raise ValueError("latest_month 不得為 None")
        if not isinstance(x, (str, datetime, pd.Timestamp, pd.Period)):
            raise ValueError(f"不支援的型別 {type(x)}")
        ts = x.to_timestamp() if isinstance(x, pd.Period) else pd.Timestamp(x)
        if pd.isna(ts):
            raise ValueError("無效的月份 (NaT)")
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.to_period("M").to_timestamp()  # 一律對齊月初

    def forecast_monthly(self, recent_history, latest_month, forecast_months=1):
        """
        recent_history：依時間排序的月度數值，最後一筆必須是 latest_month 當月。
        latest_month：最後一筆資料的月份，例如 "2024-12"。
        forecast_months：往後預測幾個月（1~12）。
        """
        self.stats["total_requests"] += 1

        # 防線 1：月份解析
        try:
            latest = self._parse_month(latest_month)
        except Exception as e:
            return self._fail("timestamp_error", f"latest_month 無效：{e}", None, None)

        # 防線 2：預測月數
        if (isinstance(forecast_months, bool)
                or not isinstance(forecast_months, (int, np.integer))
                or not 1 <= forecast_months <= 12):
            return self._fail("horizon_error", "forecast_months 必須是 1~12 的整數", None, None)
        months = [latest + pd.DateOffset(months=i) for i in range(1, int(forecast_months) + 1)]

        # 防線 3：型別與維度
        if not isinstance(recent_history, (list, tuple, np.ndarray, pd.Series)):
            return self._fail("dimension_error", "recent_history 必須為 list 或 array", months, None)
        try:
            arr = np.array(recent_history, dtype=np.float64)
        except (ValueError, TypeError):
            return self._fail("dimension_error", "輸入含非數值內容", months, None)
        if arr.ndim != 1:
            return self._fail("dimension_error", f"必須為 1D，目前為 {arr.ndim}D", months, None)

        # 防線 4：長度
        if len(arr) < self.lookback:
            return self._fail("length_insufficient",
                              f"至少需要 {self.lookback} 個月，目前 {len(arr)} 個月", months, None)

        # 防線 5：物理邊界（NaN/Inf、負值）
        window = arr[-self.lookback:]
        if not np.isfinite(window).all():
            return self._fail("invalid_input", "輸入含 NaN 或 Inf", months, None)
        if (window < 0).any():
            return self._fail("negative_value_blocked", "輸入含負值", months, None)

        # 輸入合格：此後若失敗，可用「去年同月」當備援
        if not self.is_ready:
            return self._fail("model_not_ready", self.init_error or "模型未就緒", months, window)

        # 防線 6：推論與輸出檢核（有限、非負、不超過訓練最大值的 3 倍）
        try:
            upper = self.scaler["hi"] * 3
            win = list(window)
            preds = []
            for i in range(len(months)):
                end = latest + pd.DateOffset(months=i)
                w_idx = pd.date_range(end=end, periods=self.lookback, freq="MS")
                x = _window_features(_scale(win, self.scaler), w_idx)[None, ...]
                p_scaled = float(np.asarray(
                    self.model.predict(x.astype(np.float32), verbose=0)).reshape(-1)[0])
                p = float(_unscale(p_scaled, self.scaler))
                if not np.isfinite(p) or p < 0 or p > upper:
                    return self._fail("model_output_invalid",
                                      f"第 {i + 1} 個月預測值異常：{p}", months, window)
                preds.append(p)
                win = win[1:] + [p]
        except Exception as e:
            return self._fail("unexpected_exception", f"推論例外：{e}", months, window)

        return {
            "ai_valid": 1,
            "forecast_values": [round(v, 2) for v in preds],
            "fallback_values": None,
            "prediction_months": [m.strftime("%Y-%m") for m in months],
            "unit": self.unit,
            "model_version": self.model_version,
            "error_reason": None,
            "fallback_strategy": None,
        }

    def _fail(self, error_type, msg, months, history):
        """記錄錯誤；若輸入合格，直接算出『去年同月』備援值。"""
        self.stats["errors"][error_type] += 1
        fallback, strategy = None, "none"
        if history is not None and months:
            last12 = np.asarray(history)[-12:]
            fallback = [round(float(last12[i % 12]), 2) for i in range(len(months))]
            strategy = "seasonal_naive"
        self.stats["fallback_used"][strategy] += 1
        return {
            "ai_valid": 0,
            "forecast_values": None,
            "fallback_values": fallback,
            "prediction_months": [m.strftime("%Y-%m") for m in months] if months else None,
            "unit": self.unit,
            "model_version": self.model_version,
            "error_reason": msg,
            "fallback_strategy": strategy,
        }

    def get_system_audit_report(self):
        return self.stats


_global_predictor = None


def forecast_monthly(recent_history, latest_month, forecast_months=1):
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = MonthlyEMSPredictor()
    return _global_predictor.forecast_monthly(recent_history, latest_month, forecast_months)


# ---------------------------------------------------------
# 命令列：python monthly_forecast.py data.csv --year-col 年 --month-col 月 --value-col 度數
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="月度用電預測：訓練並評估")
    ap.add_argument("csv")
    ap.add_argument("--year-col", required=True)
    ap.add_argument("--month-col", required=True)
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--ad", action="store_true", help="年份為西元年（預設為民國年）")
    ap.add_argument("--unit", default="kWh")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    s = build_monthly_series(df, args.year_col, args.month_col, args.value_col,
                             roc_year=not args.ad)
    res = run_experiment(s, out_dir=args.out_dir, target_name=args.value_col, unit=args.unit)
    print(json.dumps(res["metrics"], ensure_ascii=False, indent=2))
    print("驗證集表現最佳的模型：", res["best_model"])