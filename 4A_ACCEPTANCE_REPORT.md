\# 4A Acceptance Report: LSTM → MQTT Prediction Service (Final Technical Audit)



\## 1. Core Component Verification

\- \*\*Frozen LSTM Model / Scaler / Config\*\*: VERIFIED (SHA-256 intact, zero weight modifications)

\- \*\*Gate F1 (Real MQTT Consistency)\*\*: PASSED (End-to-end network transmission Diff <= 1e-5 kW)

\- \*\*Gate F2 (Fixed Regression Fixture)\*\*: PASSED (Deterministic output = 426.3100 kW)

\- \*\*Gate F3 \& F4 (Robustness \& Timestamp Contract)\*\*: PASSED (15/15 robust cases \& strict ISO 8601 tz-aware validation)

\- \*\*Gate F5 (ai\_valid=0 Schema)\*\*: PASSED (Proper fallback structure and non-empty error)

\- \*\*Gate F6 \& F7 (Stale, Ordering, Duplicate \& Live/Replay Mode)\*\*: PASSED (Configurable via mqtt\_config.json)

\- \*\*Gate F8 (Automatic Reconnect)\*\*: NOT VALIDATED (Public broker limitation; no synthetic mocking)

\- \*\*Gate F9 (Environment Reproducibility)\*\*: PASSED (Documented in TESTED\_ENVIRONMENT.md)



\## 2. Model Evaluation Statement

凍結模型於未參與本次模型訓練的 held-out Test 區段進行評估；公開資料結果定位為方法與流程驗證。未與 Simulink / EMS 正式整合，亦未部署於 Jetson 硬體。

