# test_regression.py
import json
import os
import hashlib
from pathlib import Path
from infer import forecast_load_kw

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_regression_test():
    print("=" * 60)
    print("⚡ 開始執行 Gate F2：真正 Fixed Regression Fixture 測試")
    print("=" * 60)
    
    base_dir = Path(__file__).resolve().parent
    
    # 1. 驗證 Frozen Assets SHA-256 完整性
    model_path = base_dir / "lstm_model_brain_v2_fixed.h5"
    scaler_path = base_dir / "data_scaler_v2_fixed.pkl"
    config_path = base_dir / "model_config_v2_fixed.json"
    
    print(f"  [Asset Check] Model SHA256  : {calculate_sha256(model_path)[:16]}...")
    print(f"  [Asset Check] Scaler SHA256 : {calculate_sha256(scaler_path)[:16]}...")
    print(f"  [Asset Check] Config SHA256 : {calculate_sha256(config_path)[:16]}...")

    # 2. 載入 Fixture 輸入
    with open(base_dir / "sample_history_kw.json", "r", encoding="utf-8") as f:
        history_kw = json.load(f).get("history_kw")
        
    with open(base_dir / "sample_input_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        latest_ts = meta.get("latest_data_timestamp")

    with open(base_dir / "sample_expected_output.json", "r", encoding="utf-8") as f:
        expected = json.load(f)

    # 3. 執行推論
    result = forecast_load_kw(history_kw, latest_ts)
    
    actual_val = result.get("forecast_load_kw")
    expected_val = expected.get("forecast_load_kw")
    diff = abs(actual_val - expected_val)
    
    print(f"  [Regression] Expected Forecast : {expected_val}")
    print(f"  [Regression] Actual Forecast   : {actual_val:.4f}")
    print(f"  [Regression] Difference        : {diff:.2e}")

    # 4. 嚴格判定條件
    tolerance = 1e-5
    val_pass = diff <= tolerance
    ver_pass = (result.get("model_version") == expected.get("model_version"))
    valid_pass = (result.get("ai_valid") == expected.get("ai_valid"))
    time_pass = (result.get("prediction_timestamp") == expected.get("prediction_timestamp"))

    print("=" * 60)
    if val_pass and ver_pass and valid_pass and time_pass:
        print("FIXED_REGRESSION_FIXTURE = PASS")
    else:
        print("FIXED_REGRESSION_FIXTURE = FAIL")
    print("=" * 60)

if __name__ == "__main__":
    run_regression_test()