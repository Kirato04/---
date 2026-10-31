# =========================================================
# 版本 B 前端：月度用電量／電費預測頁面 (monthly_app.py)
# 在 app.py 中呼叫 render_monthly_tab() 即可使用
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import monthly_forecast as mf      # 檔名為 monthly_forecast.py 時
except ModuleNotFoundError:
    import month_infer as mf           # 檔名為 month_infer.py 時

ASSET_DIR = Path(mf.__file__).resolve().parent  # 模型與設定檔存放位置


# ---------------------------------------------------------
# 小工具
# ---------------------------------------------------------
def _read_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(file)
        sheet = st.selectbox("工作表", xl.sheet_names)
        header = st.number_input("標題列位置（第一列為 0）", 0, 50, 0)
        return xl.parse(sheet, header=int(header))
    try:
        return pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="cp950")


def _guess(cols, keys, default):
    for i, c in enumerate(cols):
        if any(k in str(c) for k in keys):
            return i
    return min(default, len(cols) - 1)


def _clean_digits(df, cols):
    """「107年」「1月」這類文字，只保留數字部分。"""
    d = df.copy()
    for c in cols:
        d[c] = d[c].astype(str).str.extract(r"(\d+)")[0]
    return d


def _metrics_table(metrics):
    rows = []
    for model, by_split in metrics.items():
        for split, m in by_split.items():
            rows.append({"模型": model, "資料集": split, "MAPE (%)": round(m["MAPE_%"], 2),
                         "RMSE": round(m["RMSE"], 2), "MAE": round(m["MAE"], 2)})
    return pd.DataFrame(rows)


def _plot_predictions(frame, target, unit):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame.index, frame["actual"], color="black", marker="o", label="Actual")
    colors = {"seasonal_naive": "tab:orange", "holt_winters": "tab:green", "lstm": "tab:red"}
    for m in [c for c in frame.columns if c in colors]:
        ax.plot(frame.index, frame[m], color=colors[m], marker="x", linestyle="--", label=m)
    test_start = frame.index[frame["split"] == "test"][0]
    ax.axvline(test_start, color="gray", linestyle=":", label="Test start")
    ax.set_title("Monthly Forecast Result (Validation & Test)")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{target} ({unit})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    return fig


def _plot_forecast(series, months, values, label, unit):
    hist = series.iloc[-36:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist.index, hist.values, color="black", marker="o", label="History")
    x = pd.to_datetime([m + "-01" for m in months])
    ax.plot(x, values, color="tab:red", marker="o", linestyle="--", label=label)
    ax.set_title("Monthly Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    return fig


# ---------------------------------------------------------
# 主畫面
# ---------------------------------------------------------
def render_monthly_tab():
    st.header("版本 B：月度用電量／電費預測")
    st.caption("使用真實月度資料，不經過正弦波還原。切分方式為時間順序 5:1:1（訓練／驗證／測試）。")

    file = st.file_uploader("上傳月度資料（CSV 或 Excel）", type=["csv", "xlsx", "xls"],
                            key="monthly_upload")
    if file is None:
        st.info("資料需要有「年」「月」「數值（度數或電費）」三種欄位，每個月一列。")
        return

    try:
        df = _read_upload(file)
    except Exception as e:
        st.error(f"讀取檔案失敗：{e}")
        return
    st.dataframe(df.head(10), use_container_width=True)

    cols = [str(c) for c in df.columns]
    df.columns = cols
    c1, c2, c3 = st.columns(3)
    ycol = c1.selectbox("年份欄位", cols, index=_guess(cols, ["年"], 0))
    mcol = c2.selectbox("月份欄位", cols, index=_guess(cols, ["月"], 1))
    vcol = c3.selectbox("數值欄位（度數或電費）", cols, index=_guess(cols, ["度數", "電費", "kWh"], 2))

    c4, c5 = st.columns(2)
    roc = c4.radio("年份格式", ["民國年（107）", "西元年（2018）"], horizontal=True).startswith("民國")
    unit = c5.text_input("單位", value="kWh")

    try:
        series = mf.build_monthly_series(_clean_digits(df, [ycol, mcol]), ycol, mcol, vcol, roc_year=roc)
        train, val, test = mf.split_5_1_1(series)
    except Exception as e:
        st.error(f"資料整理失敗：{e}")
        return

    st.success(
        f"共 {len(series)} 個月（{series.index[0]:%Y-%m} ～ {series.index[-1]:%Y-%m}）　"
        f"訓練 {len(train)}／驗證 {len(val)}／測試 {len(test)} 個月"
    )
    st.line_chart(series.rename(vcol))

    # ---------------- 訓練與評估 ----------------
    st.subheader("訓練與評估")
    try:
        import statsmodels  # noqa: F401
        has_hw = True
    except ImportError:
        has_hw = False
    use_hw = st.checkbox("加入 Holt-Winters 比較", value=has_hw, disabled=not has_hw)
    if not has_hw:
        st.caption("未安裝 statsmodels，已略過 Holt-Winters（requirements.txt 加入 statsmodels 即可啟用）。")
    epochs = st.slider("LSTM 最大訓練輪數（會自動提早停止）", 50, 1000, 300, step=50)

    if st.button("開始訓練並評估", type="primary"):
        with st.spinner("訓練中，請稍候…"):
            try:
                res = mf.run_experiment(series, out_dir=str(ASSET_DIR), target_name=vcol,
                                        unit=unit, use_holt_winters=use_hw, epochs=epochs)
                st.session_state["monthly_result"] = {"res": res, "key": (file.name, ycol, mcol, vcol)}
            except Exception as e:
                st.error(f"訓練失敗：{e}")

    saved = st.session_state.get("monthly_result")
    if saved and saved["key"] == (file.name, ycol, mcol, vcol):
        res = saved["res"]
        st.write(f"驗證集表現最佳的模型：**{res['best_model']}**")
        st.dataframe(_metrics_table(res["metrics"]), use_container_width=True, hide_index=True)
        st.pyplot(_plot_predictions(res["predictions"], vcol, unit))
        st.caption("LSTM 的驗證集同時用於提早停止，所以驗證集成績略偏樂觀；測試集才是最終成績。"
                   "LSTM 必須勝過「seasonal_naive（去年同月）」才有意義。")

        # ---------------- 預測未來 ----------------
        st.subheader("預測未來月份")
        n_ahead = st.number_input("預測幾個月", 1, 12, 3)
        if st.button("執行預測"):
            predictor = mf.MonthlyEMSPredictor()
            out = predictor.forecast_monthly(series.to_numpy(), series.index[-1], int(n_ahead))
            if out["ai_valid"] == 1:
                vals, label = out["forecast_values"], "LSTM forecast"
                st.success("LSTM 預測成功")
            else:
                vals, label = out["fallback_values"], "Fallback (same month last year)"
                st.warning(f"LSTM 無法使用：{out['error_reason']}")
            if vals is None:
                st.error("也無法產生備援值。")
            else:
                tbl = pd.DataFrame({"月份": out["prediction_months"], f"預測值 ({unit})": vals})
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                st.pyplot(_plot_forecast(series, out["prediction_months"], vals, label, f"{vcol} ({unit})"))
                st.download_button("下載預測結果 CSV", tbl.to_csv(index=False).encode("utf-8-sig"),
                                   "monthly_forecast.csv", "text/csv")
            with st.expander("系統稽核報表"):
                st.json(predictor.get_system_audit_report())


# 若這個檔案被當成主程式執行（例如暫時命名為 app.py），就直接顯示月度頁面；
# 被 app.py 以 import 方式引用時，不會自動執行。
if __name__ == "__main__":
    render_monthly_tab()