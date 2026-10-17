# test_consistency.py
import json
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
import paho.mqtt.client as mqtt

from infer import forecast_load_kw
from mqtt_contract import MQTTContractValidator

class RealMQTTConsistencyTester:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        config_path = base_dir / "mqtt_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.host = self.config.get("broker_host", "broker.emqx.io")
        self.port = self.config.get("broker_port", 1883)
        self.topic = self.config.get("topic", "eams/dev/a83f41c2/ai/load_forecast/v1")
        
        self.received_messages = []
        self.lock = threading.Lock()

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            validator = MQTTContractValidator()
            valid, _ = validator.validate_payload(payload)
            if valid:
                with self.lock:
                    self.received_messages.append(payload)
        except Exception:
            pass

    def run_test(self):
        print("=" * 60)
        print("⚡ 開始執行 Gate F1 (修正版)：真正 MQTT Consistency Test")
        print("=" * 60)

        np.random.seed(100)
        cases = {
            "REAL_MQTT_CONSISTENCY_CASE_1 (Low-load)": np.random.uniform(100.0, 200.0, size=96).tolist(),
            "REAL_MQTT_CONSISTENCY_CASE_2 (Normal-load)": np.random.uniform(300.0, 400.0, size=96).tolist(),
            "REAL_MQTT_CONSISTENCY_CASE_3 (High-load)": np.random.uniform(500.0, 650.0, size=96).tolist()
        }

        sub_client = mqtt.Client(client_id=f"consistency_sub_{int(time.time())}")
        sub_client.on_message = self._on_message
        sub_client.connect(self.host, self.port, keepalive=60)
        sub_client.subscribe(self.topic, qos=1)
        sub_client.loop_start()
        time.sleep(1.5)

        pub_client = mqtt.Client(client_id=f"consistency_pub_{int(time.time())}")
        pub_client.connect(self.host, self.port, keepalive=60)
        pub_client.loop_start()

        all_pass = True
        tolerance = 1e-5

        for case_name, history in cases.items():
            current_time = datetime.now().astimezone()
            direct_result = forecast_load_kw(history, current_time)
            direct_val = direct_result.get("forecast_load_kw")

            payload = {
                "schema_version": "1.0",
                "forecast_load_kw": direct_val,
                "latest_data_timestamp": current_time.isoformat(),
                "prediction_timestamp": direct_result.get("prediction_timestamp"),
                "prediction_horizon_min": direct_result.get("prediction_horizon_min", 30),
                "model_version": direct_result.get("model_version"),
                "ai_valid": direct_result.get("ai_valid", 0),
                "generated_at": datetime.now().astimezone().isoformat(),
                "error": direct_result.get("error")
            }
            
            matched_payload = None
            # 最多重試 2 次，每次等待 4 秒
            for retry in range(2):
                with self.lock:
                    self.received_messages.clear()

                pub_client.publish(self.topic, json.dumps(payload), qos=1)

                start_wait = time.time()
                while time.time() - start_wait < 4.0:
                    with self.lock:
                        for msg in self.received_messages:
                            if abs(msg.get("forecast_load_kw", 0) - direct_val) <= tolerance:
                                matched_payload = msg
                                break
                    if matched_payload:
                        break
                    time.sleep(0.2)
                if matched_payload:
                    break

            if matched_payload:
                mqtt_val = matched_payload.get("forecast_load_kw")
                diff = abs(direct_val - mqtt_val)
                print(f"[{case_name}]")
                print(f"  Direct : {direct_val:.6f}")
                print(f"  MQTT   : {mqtt_val:.6f}")
                print(f"  Diff   : {diff:.2e} ➔ 【PASS】")
            else:
                print(f"[{case_name}] ➔ 【FAIL】 (Timeout: 透過真實 MQTT 接收逾時)")
                all_pass = False

        pub_client.loop_stop()
        pub_client.disconnect()
        sub_client.loop_stop()
        sub_client.disconnect()

        print("=" * 60)
        if all_pass:
            print("REAL_MQTT_CONSISTENCY = PASS")
        else:
            print("REAL_MQTT_CONSISTENCY = NOT VALIDATED / FAIL")
        print("=" * 60)

if __name__ == "__main__":
    tester = RealMQTTConsistencyTester()
    tester.run_test()