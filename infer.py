def forecast_load_kw(self, recent_history_kw, latest_data_timestamp, forecast_days=1):
        """
        推論介面：支援動態預測天數 (forecast_days)
        - 當 forecast_days = 1 時：輸出標準 96 步（24小時）預測。
        - 當 forecast_days > 1 時：啟動滾動預測機制，往後延伸預測多天。
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

            # 總預測分鐘數 = 天數 * 1440分鐘
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
                return self._trigger_fallback("dimension_error", f"輸入資料維度錯誤，目前維度為 {arr.ndim}", target_timestamp_str)
        except (ValueError, TypeError):
            self.stats["errors"]["dimension_error"] += 1
            return self._trigger_fallback("dimension_error", "輸入資料包含非數值型別", target_timestamp_str)
            
        if len(arr) < self.lookback_steps:
            self.stats["errors"]["length_insufficient"] += 1
            return self._trigger_fallback("length_insufficient", f"輸入序列長度不足！需要至少 {self.lookback_steps} 筆", target_timestamp_str)

        recent_window = arr[-self.lookback_steps:]

        # ==================== 防線 4：物理邊界防禦 ====================
        if not np.isfinite(recent_window).all():
            self.stats["errors"]["model_output_nan"] += 1
            return self._trigger_fallback("model_output_nan", "輸入資料包含 NaN 或 Inf 異常值", target_timestamp_str)

        if (recent_window < 0).any():
            self.stats["errors"]["negative_load_blocked"] += 1
            return self._trigger_fallback("negative_load_blocked", "輸入資料包含負值負載", target_timestamp_str)

        # ==================== 執行滾動推論 (支援 1 天或多天) ====================
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
                
                # 將本次預測的結果加入總清單，並滾動更新視窗
                all_preds.extend(pred_rescaled)
                current_window = np.concatenate([current_window[len(pred_rescaled):], pred_rescaled])

            pred_rescaled_final = np.array(all_preds[:total_steps_needed])

            # ==================== 防線 5：輸出正當性檢核 ====================
            if not np.isfinite(pred_rescaled_final).all():
                self.stats["errors"]["model_output_nan"] += 1
                return self._trigger_fallback("model_output_nan", "模型推論輸出異常", target_timestamp_str)

            # 產生未來時間點的 Timestamp 串列
            interval_delta = timedelta(minutes=15)
            prediction_timestamps = [
                (parsed_dt + interval_delta * (i + 1)).isoformat() 
                for i in range(len(pred_rescaled_final))
            ]

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