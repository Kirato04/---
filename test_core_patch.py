# test_core_patch.py
import sys
from datetime import datetime, timedelta
from mqtt_contract import MQTTContractValidator
from mqtt_subscriber import MQTTSubscriber

def run_patch_tests():
    print("=" * 60)
    print("⚡ 開始執行 4A_CORE_PATCH 驗證測試")
    print("=" * 60)
    
    validator = MQTTContractValidator()
    base_time = datetime.now().astimezone()
    valid_latest = base_time.isoformat()
    valid_pred = (base_time + timedelta(minutes=30)).isoformat()

    # 1. 正常 ai_valid=1 → ACCEPT
    p1 = {
        "schema_version": "1.0", "forecast_load_kw": 350.0,
        "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
        "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
        "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
    }
    v1, _ = validator.validate_payload(p1)
    assert v1 == True, "Test 1 failed"
    print("[1. 正常 ai_valid=1] ➔ 【PASS】")

    # 2. 合法 ai_valid=0 + null prediction timestamp → FALLBACK
    p2 = {
        "schema_version": "1.0", "forecast_load_kw": None,
        "latest_data_timestamp": valid_latest, "prediction_timestamp": None,
        "prediction_horizon_min": 30, "model_version": "unknown",
        "ai_valid": 0, "generated_at": base_time.isoformat(), "error": "Model initialization exception"
    }
    v2, _ = validator.validate_payload(p2)
    assert v2 == True, "Test 2 failed"
    print("[2. 合法 ai_valid=0 + null prediction timestamp] ➔ 【PASS】")

    # 3. JSON null → CONTRACT REJECT
    v3, _ = validator.validate_payload(None)
    assert v3 == False, "Test 3 failed"
    print("[3. JSON null 頂層防呆] ➔ 【PASS】")

    # 4. JSON array → CONTRACT REJECT
    v4, _ = validator.validate_payload([1, 2, 3])
    assert v4 == False, "Test 4 failed"
    print("[4. JSON array 頂層防呆] ➔ 【PASS】")

    # 5 & 6. validation_mode 合法測試 (live / replay)
    try:
        sub_live = MQTTSubscriber(config_data={"validation_mode": "live", "broker_host": "broker.emqx.io", "broker_port": 1883, "topic": "test"})
        sub_replay = MQTTSubscriber(config_data={"validation_mode": "replay", "broker_host": "broker.emqx.io", "broker_port": 1883, "topic": "test"})
        print("[5 & 6. validation_mode live/replay] ➔ 【PASS】")
    except Exception as e:
        print(f"[5 & 6] ➔ 【FAIL】: {str(e)}")
        sys.exit(1)

    # 7. validation_mode 拼錯字 ("liv") → 必須拋出 ValueError
    try:
        sub_invalid = MQTTSubscriber(config_data={"validation_mode": "liv", "broker_host": "broker.emqx.io", "broker_port": 1883, "topic": "test"})
        print("[7. validation_mode=\"liv\"] ➔ 【FAIL】 (應攔截卻未報錯)")
        sys.exit(1)
    except ValueError:
        print("[7. validation_mode=\"liv\" 嚴格拒絕] ➔ 【PASS】")

    print("=" * 60)
    print("4A_CORE_PATCH = PASS")
    print("=" * 60)

if __name__ == "__main__":
    run_patch_tests()