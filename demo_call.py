import numpy as np
from datetime import datetime
from infer import forecast_load_kw

print("=== 微電網 EMS 系統 - AI 預測呼叫測試 ===")

# 1. 模擬 EMS 系統收集到的近期歷史負載資料 (過去 24 小時，每 15 分鐘一筆，共 96 筆)
# 這裡用 numpy 產生隨機數值做為 demo (真實情況下由現場電表提供)
simulated_history_kw = np.random.uniform(low=200.0, high=600.0, size=96).tolist()

# 2. 模擬當下的最新資料時間
current_time = datetime.now()
print(f"📥 接收到最新資料，時間點: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 3. 呼叫推論模組 (這就是未來 EMS 系統真正會執行的動作)
print("⏳ AI 預測引擎運算中...\n")
result = forecast_load_kw(
    recent_history_kw=simulated_history_kw,
    latest_data_timestamp=current_time
)

# 4. 輸出並應用結果
if result.get("ai_valid") == 1:
    print("✅ 預測成功！")
    print(f"👉 目標時間 ({result['prediction_horizon_min']} 分鐘後): {result['prediction_timestamp']}")
    print(f"👉 預測負載: {result['forecast_load_kw']} kW")
    print(f"🤖 模型版本: {result['model_version']}")
else:
    print("⚠️ 預測失敗！")
    print(f"錯誤原因: {result.get('error', '未知錯誤')}")