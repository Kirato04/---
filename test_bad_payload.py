import json
import time
import paho.mqtt.client as mqtt

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "eams/dev/a83f41c2/ai/load_forecast/v1"  # 請確保與接收端 Topic 一致

client = mqtt.Client()
client.connect(BROKER, PORT, 60)
client.loop_start()
time.sleep(1)

print("=== 開始展示 AI Invalid 攔截測試 ==-\n")

# 模擬 AI 模型出錯或無法預測時，強制把 ai_valid 設為 0
ai_invalid_payload = {
    "schema_version":" 1.0",
    "forecast_load_kw": 0.0,
    "latest_data_timestamp": "2026-09-01T19:00:00+08:00",
    "prediction_timestamp": "2026-09-01T19:30:00+08:00",
    "prediction_horizon_min": 30,
    "model_version": "lstm_v1",
    "ai_valid": 0,  # <--- 關鍵！觸發接收端攔截
    "error": "Model computation timeout or NaN detected",
    "generated_at": "2026-09-01T19:05:00+08:00"
}

print("[測試] 正在發送 ai_valid = 0 的異常預測封包...")
client.publish(TOPIC, json.dumps(ai_invalid_payload))
time.sleep(2)

client.loop_stop()
client.disconnect()
print("\n=== 測試完畢 ===")