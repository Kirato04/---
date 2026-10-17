# test_mqtt_pipeline.py
import json
import sys
from datetime import datetime, timedelta
from mqtt_contract import MQTTContractValidator

def run_robustness_tests():
    validator = MQTTContractValidator()
    
    base_time = datetime.now().astimezone()
    valid_latest = base_time.isoformat()
    valid_pred = (base_time + timedelta(minutes=30)).isoformat()
    
    # 定義 15 個嚴格的傳輸與解析測試案例
    test_cases = [
        {
            "name": "1. 正常 Happy Path (ai_valid=1)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": 350.5,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": True
        },
        {
            "name": "2. 合法 Fallback (ai_valid=0)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": None,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 0, "generated_at": base_time.isoformat(), "error": "Insufficient history"
            },
            "expected_valid": True
        },
        {
            "name": "3. 殘缺的 JSON bytes 格式",
            "payload_raw": b'{"forecast_load_kw": 350.5, "schema_version": "1.0"',
            "expected_valid": False
        },
        {
            "name": "4. 空的 payload bytes",
            "payload_raw": b'',
            "expected_valid": False
        },
        {
            "name": "5. 缺少必填欄位 (missing forecast_load_kw)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": None,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "6. 錯誤的 schema_version (2.0)",
            "payload": {
                "schema_version": "2.0", "forecast_load_kw": 350.5,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "7. 錯誤的 horizon (60 min)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": 350.5,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 60, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "8. forecast_load_kw 為布林值 (True)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": True,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "9. ai_valid 為布林值 (True)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": 350.5,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": True, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "10. 數值型態字串攻擊 (\"343.4\")",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": "343.4",
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "11. NaN 數值攻擊",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": float('nan'),
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "12. 負數負載攻擊 (-10.5)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": -10.5,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "13. 無效的 generated_at 格式",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": 350.5,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": "not-a-timestamp", "error": None
            },
            "expected_valid": False
        },
        {
            "name": "14. 缺少時區的 latest_data_timestamp (Naive)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": 350.5,
                "latest_data_timestamp": "2026-08-23T10:00:00", "prediction_timestamp": valid_pred,
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        },
        {
            "name": "15. 缺少時區的 prediction_timestamp (Naive)",
            "payload": {
                "schema_version": "1.0", "forecast_load_kw": 350.5,
                "latest_data_timestamp": valid_latest, "prediction_timestamp": "2026-08-23T10:30:00",
                "prediction_horizon_min": 30, "model_version": "v2_public_data_fixed",
                "ai_valid": 1, "generated_at": base_time.isoformat(), "error": None
            },
            "expected_valid": False
        }
    ]

    print("=" * 60)
    print("⚡ 開始執行 Gate F3：真正 Robustness & Parser 壓力測試")
    print("=" * 60)
    
    passed_count = 0
    failed_count = 0

    for case in test_cases:
        name = case["name"]
        expected = case["expected_valid"]
        
        try:
            if "payload_raw" in case:
                # 模擬真實 MQTT 二進位位元組解析
                raw_bytes = case["payload_raw"]
                decoded_str = raw_bytes.decode("utf-8")
                payload_dict = json.loads(decoded_str)
                actual, _ = validator.validate_payload(payload_dict)
            else:
                actual, _ = validator.validate_payload(case["payload"])
        except Exception:
            # 發生解碼或 JSON 解析錯誤，視為契約/解析拒絕 (Valid = False)
            actual = False

        is_pass = (actual == expected)
        if is_pass:
            passed_count += 1
            print(f"[{name}] ➔ 預期: {expected} | 實際: {actual} ➔ 【PASS】")
        else:
            failed_count += 1
            print(f"[{name}] ➔ 預期: {expected} | 實際: {actual} ➔ 【FAIL】")

    print("=" * 60)
    print(f"測試結果統計：總計 {len(test_cases)} 項 | 通過: {passed_count} 項 | 失敗: {failed_count} 項")
    print("=" * 60)

    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_robustness_tests()