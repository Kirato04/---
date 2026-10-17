# ⚡ 微電網 EMS 系統 - 負載預測 AI 與 MQTT 整合服務 (4A)

本模組為微電網專題中負責預測建築未來負載 (kW) 的 AI 核心引擎，並透過合約驗證的 MQTT 通訊管道與中央 EMS 進行端到端（End-to-End）安全對接。

## 1. 模組架構與檔案清單
* `infer.py`: 推論主程式 (包含 `EMSLoadPredictor` 類別與對外 API `forecast_load_kw`)[cite: 3]
* `mqtt_contract.py`: 通訊合約安檢員 (`MQTTContractValidator`)，嚴格執行型態、時區與數值防禦
* `mqtt_publisher.py`: 資料發布端，結合 AI 推論與合約安檢發布至 MQTT Broker
* `mqtt_subscriber.py`: 資料接收端 (Mock EMS)，內含三種狀態分流及 Stale/Ordering/Duplicate 檢查
* `mqtt_config.json`: 集中化 MQTT 連線設定檔
* `lstm_model_brain_v2_fixed.h5`: 凍結版 LSTM 模型權重 (禁止修改)[cite: 3]
* `data_scaler_v2_fixed.pkl`: 僅在訓練集擬合之特徵還原器 (禁止修改)[cite: 3]
* `model_config_v2_fixed.json`: 模型規格與物理單位定義 (禁止修改)[cite: 3]
* `test_mqtt_pipeline.py`: Step 5 異常情境與壓力測試腳本
* `test_consistency.py`: Step 6 數值一致性比對腳本
* `test_regression.py`: Step 7 迴歸固定檔測試腳本
* `requirements.txt`: 系統套件依賴清單[cite: 3]
* `4A_ACCEPTANCE_REPORT.md`: 最終 4A 驗收與評估報告

## 2. 模型規格與效能指標
* **輸入與輸出**: 讀取過去 24 小時 (96 步)，預測未來 30 分鐘的平均有功功率[cite: 3]。
* **物理單位**: kW[cite: 3]。
 * **[極度重要]**：本 inference 模組之輸入必須為 `kW`。`infer.py` **不負責**處理 `kWh -> kW` 的轉換[cite: 3]。
* **模型評估成績**: 凍結模型於未參與本次模型訓練的 held-out Test 區段進行評估；公開資料結果定位為方法與流程驗證。

## 3. MQTT 通訊合約與安全政策
* **Schema Version**: 嚴格強制為 `"1.0"`。
* **Timestamp**: 統一採用帶有時區 (`+08:00`) 的 ISO 8601 格式。
* **Type Safety**: 嚴格阻擋將 Python `bool`（如 `True`）當作 `int` 或數值傳入的攻擊。
* **Subscriber 三種狀態分流**:
  1. `CONTRACT REJECT`: 格式或 JSON 結構錯誤。
  2. `AI INVALID / FALLBACK`: 格式正確但 `ai_valid=0`（不採信預測數值）。
  3. `FORECAST ACCEPT`: 格式合法且 `ai_valid=1`（中央系統正式採用）。
* **Freshness Policy**: 內建 `STALE_TIMEOUT_SECONDS = 120.0`（DEVELOPMENT_VALUE_NOT_FINAL）、亂序防禦及重複訊息過濾。

## 4. 快速啟動與測試 (Quick Start)

### 安裝依賴環境
```bash
pip install -r requirements.txt