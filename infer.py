# =========================================================
# 微電網 EMS 系統 - 負載預測推論與防禦模組 (infer.py - 完整修正支援多天滾動版)
# =========================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

class EMSLoadPredictor:
    def __init__(self, 
                 model_path='lstm_model_brain_v2_fixed.h5', 
                 scaler_path='data_scaler_v2_fixed.pkl', 
                 config_path='model_config_v2_fixed.json'):
        """
        初始化預測器：具備完整安全防禦、LSTM 模型載入與錯誤稽核計數器
        """
        base_dir = Path(__file__).resolve().parent
        self.model_path = str(base_dir / model_path)
        self.scaler_path = str(base_dir / scaler_path)
        self.config_path = str(base_dir / config_path)
        
        # 安全預設值
        self.model = None
        self.scaler = None
        self.config = {}
        self.is_ready = False
        self.model_version = "unknown"
        self.horizon_min = None
        self.lookback_steps = None
        self.target_unit = "kW"
        self.init_error = None

        # 系統防禦與錯誤稽核計數器 (System Audit & Error Logging)
        self.stats = {
            "total_requests": 0,
            "errors": {
                "timestamp_error": 0,
                "dimension_error": 0,
                "length_insufficient": 0,
                "negative_load_blocked": 0,
                "model_output_nan": 0,
                "model_not_ready": 0,
                "unexpected_exception": 0
            },
            "fallback_used": {
                "historical_matching": 0,
                "moving_average": 0
            }
        }

        # 嘗試載入資產
        self._load_components()

    def _load_components(self):
        try:
            # 1. 讀取 Config
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"找不到設定檔: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            self.lookback_steps = self.config.get('lookback_steps', 192)
            self.model_version = self.config.get('model_version', 'v2_public_data_fixed')
            self.target_unit = self.config.get('target_unit', 'kW')

            # Horizon 動態取得
            if "forecast_horizon_minutes" in self.config:
                self.horizon_min = int(self.config["forecast_horizon_minutes"])
            elif "steps_ahead" in self.config and "interval_minutes" in self.config:
                self.horizon_min = int(self.config["steps_ahead"] * self.config["interval_minutes"])
            else:
                self.horizon_min = 1440 # 預設 24 小時

            # 2. 載入 Scaler
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"找不到 Scaler: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)

            # 3. 載入 Keras Model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"找不到模型檔: {self.model_path}")
            
            class CustomDense(tf.keras.layers.Dense):
                def __init__(self, **kwargs):
                    kwargs.pop('quantization_config', None)
                    super().__init__(**kwargs)

            self.model = load_model(
                self.model_path, 
                compile=False,
                custom_objects={'Dense': CustomDense} 
            )

            self.is_ready = True

        except Exception as e:
            self.is_ready = False
            self.init_error = f"元件載入失敗: {str(e)}"

    def forecast_load_kw(self, recent_history_kw, latest_data_timestamp, forecast_days=1):
        """
        正式推論介面：支援動態預測天數 (forecast_days) 與多步滾動預測
        """
        self.stats["total_requests"] += 1
        target_timestamp_str = None

        # ==================== 防線 1：時間戳與時區對齊驗證 ====================
        if latest_data_timestamp is None:
            self.stats["errors"]["timestamp_error"] += 1
            return self._trigger_fallback("timestamp_error", "latest_data_timestamp 為必填參數，不得為 None")

        try:
            if isinstance(latest_data_timestamp, str):
                parsed_dt = pd.to_datetime(latest_data_timestamp)
            elif isinstance(latest_data_timestamp, (datetime, pd.Timestamp)):
                parsed_dt = pd.Timestamp(latest_data_timestamp)
            else:
                self.stats["errors"]["timestamp_error"] += 1
                return self._trigger_fallback("timestamp_error", f"不支援的時間格式類型: {type(latest_data_timestamp)}")
            
            if pd.isna(parsed_dt):
                self.stats["errors"]["timestamp_error"] += 1
                return self._trigger_fallback("timestamp_error", "無效的 timestamp 內容 (NaT)")

            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.tz_localize('Asia/Taipei')

            total_horizon_min = int(forecast_days * 1440)
            horizon_delta = timedelta(minutes=total_horizon_min)
            target_dt = parsed_dt + horizon_delta
            target_timestamp_str = target_dt.isoformat() 
            
        except Exception as e:
            self.stats["errors"]["timestamp_error"] += 1
            return self._trigger_fallback("timestamp_error", f"Timestamp 解析錯誤: {str(e)}")

        # 檢查模型是否就緒
        if not self.is_ready:
            self.stats["errors"]["model_not_ready"] += 1
            return self._trigger_fallback("model_not_ready", self.init_error or "模型元件未就緒", target_timestamp_str)

        # ==================== 防線 2 & 3：型別、維度與長度檢查 ====================
        if not isinstance(recent_history_kw, (list, tuple, np.ndarray, pd.Series)):
            self.stats["errors"]["dimension_error"] += 1
            return self._trigger_fallback("dimension_error", "recent_history_kw 必須為 list 或 array 格式", target_timestamp_str)

        try:
            arr = np.array(recent_history_kw, dtype=np.float64)
            if arr.ndim != 1:
                self.stats["errors"]["dimension_error"] += 1
                return self._trigger_fallback("dimension_error", f"輸入資料維度錯誤，必須為 1D array，目前維度為 {arr.ndim}", target_timestamp_str)
        except (ValueError, TypeError):
            self.stats["errors"]["dimension_error"] += 1
            return self._trigger_fallback("dimension_error", "輸入資料包含非數值型別 (Non-numeric values)", target_timestamp_str)
            
        if len(arr) < self.lookback_steps:
            self.stats["errors"]["length_insufficient"] += 1
            return self._trigger_fallback("length_insufficient", f"輸入序列長度不足！需要至少 {self.lookback_steps} 筆，目前為 {len(arr)} 筆", target_timestamp_str)

        recent_window = arr[-self.lookback_steps:]

        # ==================== 防線 4：物理邊界防禦（NaN/Inf 與負值阻斷） ====================
        if not np.isfinite(recent_window).all():
            self.stats["errors"]["model_output_nan"] += 1
            return self._trigger_fallback("model_output_nan", "輸入資料包含 NaN 或 Inf 異常值", target_timestamp_str)

        if (recent_window < 0).any():
            self.stats["errors"]["negative_load_blocked"] += 1
            return self._trigger_fallback("negative_load_blocked", "輸入資料包含負值負載 (Negative Load)", target_timestamp_str)

        # ==================== 執行正規化與多步滾動推論 ====================
        try:
            total_steps_needed = int(forecast_days * 96) # 每天 96 步 (15分鐘一筆)
            current_window = recent_window.copy()
            all_preds = []

            while len(all_preds) < total_steps_needed:
                raw_series = current_window.reshape(-1, 1)
                scaled_series = self.scaler.transform(raw_series)
                input_tensor = scaled_series.reshape(1, self.lookback_steps, 1)

                scaled_pred = self.model.predict(input_tensor, verbose=0)
                
                if not np.isfinite(scaled_pred).all():
                    break

                pred_rescaled = self.scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()
                
                all_preds.extend(pred_rescaled)
                current_window = np.concatenate([current_window[len(pred_rescaled):], pred_rescaled])

            pred_rescaled_final = np.array(all_preds[:total_steps_needed])

            # ==================== 防線 5：輸出正當性檢核 ====================
            if not np.isfinite(pred_rescaled_final).all():
                self.stats["errors"]["model_output_nan"] += 1
                return self._trigger_fallback("model_output_nan", "模型推論輸出異常 (包含 NaN/Inf)", target_timestamp_str)

            # 產生未來時間點的 Timestamp 串列
            interval_delta = timedelta(minutes=15)
            prediction_timestamps = [
                (parsed_dt + interval_delta * (i + 1)).isoformat() 
                for i in range(len(pred_rescaled_final))
            ]

            # ==================== 正常輸出 (ai_valid: 1) ====================
            return {
                "ai_valid": 1,
                "forecast_load_kw": [round(float(val), 2) for val in pred_rescaled_final],
                "prediction_timestamp": prediction_timestamps[0],
                "prediction_timestamps": prediction_timestamps,
                "prediction_horizon_min": total_horizon_min,
                "model_version": self.model_version,
                "error_reason": None,
                "fallback_strategy": None
            }

        except Exception as e:
            self.stats["errors"]["unexpected_exception"] += 1
            return self._trigger_fallback("unexpected_exception", f"推論過程發生未預期例外: {str(e)}", target_timestamp_str)

    def _trigger_fallback(self, error_type, error_msg, target_timestamp_str=None):
        """當觸發防禦時，決定要用哪種 Fallback 策略並記錄統計"""
        if error_type in ["length_insufficient", "timestamp_error", "model_not_ready"]:
            strategy = "historical_matching"
        else:
            strategy = "moving_average"
            
        self.stats["fallback_used"][strategy] += 1
        
        return {
            "ai_valid": 0,
            "forecast_load_kw": None,
            "prediction_timestamp": target_timestamp_str,
            "prediction_horizon_min": self.horizon_min,
            "model_version": self.model_version,
            "error_reason": error_msg,
            "fallback_strategy": strategy
        }

    def get_system_audit_report(self):
        """提供給網頁或系統呼叫的即時稽核報表"""
        return self.stats

# 頂層簡便函式
_global_predictor = None

def forecast_load_kw(recent_history_kw, latest_data_timestamp, forecast_days=1):
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = EMSLoadPredictor()
    return _global_predictor.forecast_load_kw(recent_history_kw, latest_data_timestamp, forecast_days)