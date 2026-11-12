# =========================================================
# 微電網 EMS 系統 - 智慧校園用電分析與 AI 預測平台 (app.py)
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

import json
from datetime import datetime
from pathlib import Path

# =========================================================
# 月度用電量預測：資料讀取與模型推論（模型在 Colab 訓練）
# =========================================================
LOOKBACK_MONTHS = 12


CAMPUS_SHEET = "107-113各月度數費用合併"


CAMPUS_YEARS = {2018: "107年", 2019: "108年", 2020: "109年", 2021: "110年",
                2022: "111年", 2023: "112年", 2024: "113年"}


CAMPUS_LOCATIONS = ["不分區 (全校總用電)", "行政區", "商學", "校友", "男", "女", "中科", "文創"]


_UNIT_EN = {"度": "kWh", "度數": "kWh", "元": "NTD"}


def build_monthly_series(df, year_col, month_col, value_col, roc_year=True):
    d = df[[year_col, month_col, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    year = pd.to_numeric(d[year_col], errors="coerce")
    month = pd.to_numeric(d[month_col], errors="coerce")
    if roc_year:
        year = year + 1911
    d["_y"], d["_m"] = year, month
    d = d.dropna(subset=["_y", "_m", value_col])
    if d.empty:
        raise ValueError("沒有可用的「年／月／數值」資料，請確認欄位選擇是否正確")

    idx = pd.to_datetime(
        {"year": d["_y"].astype(int), "month": d["_m"].astype(int), "day": 1}
    )
    s = pd.Series(d[value_col].to_numpy(dtype=float), index=idx).sort_index()

    if s.index.duplicated().any():
        raise ValueError(f"有重複的月份：{list(s.index[s.index.duplicated()].strftime('%Y-%m'))}")
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    missing = full.difference(s.index)
    if len(missing) > 0:
        raise ValueError(f"缺少月份：{list(missing.strftime('%Y-%m'))}")
    if (s < 0).any():
        raise ValueError("資料中含有負值")
    s.index.freq = "MS"
    return s


def _fit_scaler(train_values):
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if std < 1e-9:
        std = 1.0
    return {"mean": mean, "std": std}


def _scale(x, sc):
    return (np.asarray(x, dtype=float) - sc["mean"]) / sc["std"]


def _unscale(x, sc):
    return np.asarray(x, dtype=float) * sc["std"] + sc["mean"]


def _month_feats(index):
    ang = 2 * np.pi * np.asarray(index.month) / 12.0
    return np.column_stack([np.sin(ang), np.cos(ang)])


def load_temperature_file(file):
    """支援兩種格式：(1) 年份,月份,平均氣溫  (2) CODiS 月報表（列＝年份，欄＝1~12 月，最後一欄年平均）。"""
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="cp950")
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    if "年份" in cols and "月份" in cols:
        tcol = next((c for c in cols if "氣溫" in c), None)
        if tcol is None:
            raise ValueError("找不到「氣溫」欄位")
        y = pd.to_numeric(df["年份"], errors="coerce")
        m = pd.to_numeric(df["月份"], errors="coerce")
        v = pd.to_numeric(df[tcol], errors="coerce")
    else:
        month_cols = [str(i) for i in range(1, 13)]
        if not all(c in cols for c in month_cols):
            raise ValueError("格式無法辨認：需要「年份,月份,平均氣溫」三欄，或 CODiS 月報表（欄位為 1～12 月）")
        rows = []
        for _, r in df.iterrows():
            yy = pd.to_numeric(r[cols[0]], errors="coerce")
            for mm in range(1, 13):
                rows.append((yy, mm, pd.to_numeric(r[str(mm)], errors="coerce")))
        t = pd.DataFrame(rows, columns=["y", "m", "v"])
        y, m, v = t["y"], t["m"], t["v"]
    d = pd.DataFrame({"y": y, "m": m, "v": v}).dropna()
    d = d[(d["m"] >= 1) & (d["m"] <= 12)]
    if d.empty:
        raise ValueError("氣溫檔沒有可用的資料")
    idx = pd.to_datetime({"year": d["y"].astype(int), "month": d["m"].astype(int), "day": 1})
    s = pd.Series(d["v"].to_numpy(dtype=float), index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if ((s < -10) | (s > 45)).any():
        raise ValueError("氣溫數值超出合理範圍 (-10～45 °C)，請確認單位")
    return s


def align_temperature(series, temp):
    """取出與用電資料同月份的氣溫；缺任何一個月就報錯。"""
    missing = [d for d in series.index if d not in temp.index or not np.isfinite(temp[d])]
    if missing:
        names = [d.strftime("%Y-%m") for d in missing[:6]]
        raise ValueError(f"氣溫檔缺少這些月份：{names}{'…' if len(missing) > 6 else ''}（共 {len(missing)} 個）")
    return temp.reindex(series.index).to_numpy(dtype=float)


def forecast_temp_adjusted(series, temp, n_months, beta, clim, known_future=None):
    """未來 n_months 個月的氣溫修正預測。known_future 有值時，優先使用其中的實際氣溫。"""
    last = series.index[-1]
    months = [last + pd.DateOffset(months=i) for i in range(1, n_months + 1)]
    vals = []
    for m in months:
        prev = m - pd.DateOffset(years=1)
        t_exp = clim[m.month]
        if known_future is not None and m in known_future.index and np.isfinite(known_future[m]):
            t_exp = float(known_future[m])
        vals.append(float(series[prev] * (1 + beta * (t_exp - temp[prev]))))
    return months, vals


def _en_unit(u):
    u = (u or "").strip()
    return _UNIT_EN.get(u, u if (u and u.isascii()) else "unit")


def _campus_year_row(df_excel, target_mco, location):
    if location == "不分區 (全校總用電)":
        # 例如 target_mco = "107年" → 找到「107度數」區塊，取其中的「總和」列。
        # （「107費用」區塊也有「總和」列，但那是電費，不能拿來加總）
        label = target_mco.replace("年", "度數")
        n = len(df_excel)
        for idx in range(n):
            v = df_excel.iloc[idx, 0]
            if pd.notna(v) and str(v).strip() == label:
                for j in range(idx + 1, min(idx + 15, n)):
                    w = df_excel.iloc[j, 0]
                    if pd.notna(w) and str(w).strip() == "總和":
                        vals = pd.to_numeric(
                            pd.Series(df_excel.iloc[j, 1:13].values), errors="coerce"
                        ).to_numpy(dtype=float)
                        vals[vals == 0] = np.nan   # 尚未發生的月份（例如 113 年 11、12 月）在 Excel 裡是 0
                        return vals
        return None

    block = None
    for idx, row in df_excel.iterrows():
        val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if target_mco in val and "度數" not in val and "費用" not in val:
            block = target_mco
            continue
        elif any(y in val for y in CAMPUS_YEARS.values()) and val != target_mco:
            block = None
        if block == target_mco and val == location:
            return row.iloc[1:13].values
    return None


def load_campus_series(xl, sheet, location):
    df_excel = xl.parse(sheet, header=None)
    rows, missing = [], []
    for year, mco in CAMPUS_YEARS.items():
        data = _campus_year_row(df_excel, mco, location)
        if data is None:
            missing.append(mco)
            continue
        vals = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy()
        rows.extend((year, m, v) for m, v in enumerate(vals, start=1))
    if missing:
        raise ValueError(f"在檔案中找不到 {'、'.join(missing)} 的【{location}】資料")
    tbl = pd.DataFrame(rows, columns=["year", "month", "kwh"])
    return build_monthly_series(tbl, "year", "month", "kwh", roc_year=False)


def _style_prediction_fig(fig, title, unit, x_title="Month"):
    """讓 AI 預測圖和「用電統計」圖用同一種版面：白底、互動、滑過顯示同月所有數值。"""
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title, tickformat="%Y-%m"),
        yaxis=dict(title=f"Monthly value ({_en_unit(unit)})", tickformat=","),
        template="plotly_white", height=450, hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=60, b=40),
    )


def _plot_monthly_trend(df_target, year, location):
    """月趨勢圖：優先用 Plotly（可互動、中文不會變方塊）；沒安裝 plotly 時退回 Matplotlib。"""
    s = df_target["load_kWh"]
    peak_month, peak = s.idxmax(), float(s.max())
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None

    if go is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines+markers", name=f"{location} Load",
            line=dict(color="forestgreen", width=3), marker=dict(size=9)))
        fig.add_annotation(
            x=peak_month, y=peak, text=f"Peak: {peak:,.0f} kWh", showarrow=True, arrowhead=2,
            ax=0, ay=-40, font=dict(color="crimson", size=13))
        fig.update_layout(
            title=f"{year} {location} Monthly Electricity Load Analysis",
            xaxis=dict(title=f"Month ({year})", tickmode="array", tickvals=list(s.index),
                       ticktext=[m.strftime("%b") for m in s.index]),
            yaxis_title="Power Consumption (kWh / Month)",
            template="plotly_white", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s.index, s.values, marker="o", color="forestgreen", linewidth=2.5, markersize=7,
            label=f"{location} Load")
    ax.annotate(f"Peak: {peak:,.0f} kWh", xy=(peak_month, peak), xytext=(0, 15),
                textcoords="offset points", ha="center", fontweight="bold", color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
    ax.set_title(f"{year} {location} Monthly Electricity Load Analysis", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Month ({year})", fontsize=12)
    ax.set_ylabel("Power Consumption (kWh / Month)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xticks(s.index)
    ax.set_xticklabels([m.strftime("%b") for m in s.index])
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------
# 月度 LSTM：特徵、推論與連續預測（Colab 訓練與網站推論共用同一份程式）
# ---------------------------------------------------------
LOCATION_SLUGS = {
    "不分區 (全校總用電)": "total", "行政區": "admin", "商學": "business", "校友": "alumni",
    "男": "male", "女": "female", "中科": "ctsp", "文創": "culture",
}
MODEL_DIR_NAME = "models"


def window_features(y_win, idx_win, t_win=None, t_target=None):
    """一個視窗（12 個月）的特徵：[用電量, sin(月), cos(月)]；含氣溫時再加 [當月氣溫, 目標月氣溫]。"""
    cols = [np.asarray(y_win, dtype=float)]
    f = _month_feats(idx_win)
    cols += [f[:, 0], f[:, 1]]
    if t_win is not None:
        cols += [np.asarray(t_win, dtype=float), np.full(len(y_win), float(t_target))]
    return np.column_stack(cols)


def make_samples(y_s, index, t_s, start, end, use_temp, target_t_s=None):
    """目標月 t 屬於 [start, end)：輸入是 t 之前的 12 個月。target_t_s 指定『目標月氣溫』（預設用實際氣溫）。"""
    X, y = [], []
    for t in range(max(start, LOOKBACK_MONTHS), end):
        w = slice(t - LOOKBACK_MONTHS, t)
        if use_temp:
            tt = t_s[t] if target_t_s is None else target_t_s[t]
            X.append(window_features(y_s[w], index[w], t_s[w], tt))
        else:
            X.append(window_features(y_s[w], index[w]))
        y.append(y_s[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    """等價於 Keras 的「LSTM 堆疊 → Dropout → Dense」（Dropout 在推論時不作用）。
    layers：[(kernel, recurrent, bias), ...]，每個元素是一層 LSTM；除了最後一層，每層都輸出整段序列給下一層。
    權重排列與 Keras 相同：kernel (輸入數, 4*單元數)、recurrent (單元數, 4*單元數)，四個閘門依序為 i, f, c, o。"""

    def __init__(self, layers, dense_w, dense_b):
        self.layers = [(np.asarray(k, dtype=np.float64), np.asarray(r, dtype=np.float64), np.asarray(b, dtype=np.float64))
                       for k, r, b in layers]
        self.dense_w = np.asarray(dense_w, dtype=np.float64)
        self.dense_b = np.asarray(dense_b, dtype=np.float64)

    @classmethod
    def from_npz(cls, path):
        z = np.load(path)
        if "kernel" in z.files:                                   # 舊格式：單層 LSTM
            layers = [(z["kernel"], z["recurrent"], z["bias"])]
        else:                                                     # 新格式：kernel_0, kernel_1, ...
            n = sum(1 for k in z.files if k.startswith("kernel_"))
            layers = [(z[f"kernel_{i}"], z[f"recurrent_{i}"], z[f"bias_{i}"]) for i in range(n)]
        return cls(layers, z["dense_w"], z["dense_b"])

    @property
    def n_features(self):
        return self.layers[0][0].shape[0]

    @property
    def units(self):
        return [r.shape[0] for _, r, _ in self.layers]

    def predict(self, X, verbose=0):
        seq = np.asarray(X, dtype=np.float64)
        for kernel, recurrent, bias in self.layers:
            B, T, _ = seq.shape
            H = recurrent.shape[0]
            h = np.zeros((B, H))
            c = np.zeros((B, H))
            outs = []
            for t in range(T):
                z = seq[:, t, :] @ kernel + h @ recurrent + bias
                i = _sigmoid(z[:, :H])
                f = _sigmoid(z[:, H:2 * H])
                g = np.tanh(z[:, 2 * H:3 * H])
                o = _sigmoid(z[:, 3 * H:])
                c = f * c + i * g
                h = o * np.tanh(c)
                outs.append(h)
            seq = np.stack(outs, axis=1)                          # (B, T, H)：下一層的輸入
        return seq[:, -1, :] @ self.dense_w + self.dense_b


def forecast_recursive(model, y_hist, idx_hist, n_months, y_sc, t_hist=None, t_sc=None, t_future=None, strict=True):
    """連續預測 n_months 個月：每預測一個月就把結果接回視窗，再預測下一個月。
    t_hist：視窗內各月的實際氣溫（模型含氣溫時才需要）；t_future：未來各月使用的氣溫（預期或已知）。"""
    use_temp = t_hist is not None
    ys = list(_scale(np.asarray(y_hist, dtype=float)[-LOOKBACK_MONTHS:], y_sc))
    idx = list(pd.DatetimeIndex(idx_hist)[-LOOKBACK_MONTHS:])
    ts = list(_scale(np.asarray(t_hist, dtype=float)[-LOOKBACK_MONTHS:], t_sc)) if use_temp else None
    out = []
    for i in range(n_months):
        win_idx = pd.DatetimeIndex(idx[-LOOKBACK_MONTHS:])
        if use_temp:
            tt = float(_scale(t_future[i], t_sc))
            x = window_features(np.array(ys[-LOOKBACK_MONTHS:]), win_idx, np.array(ts[-LOOKBACK_MONTHS:]), tt)
        else:
            x = window_features(np.array(ys[-LOOKBACK_MONTHS:]), win_idx)
        p = float(np.asarray(model.predict(x[None, ...].astype(np.float32), verbose=0)).reshape(-1)[0])
        val = float(_unscale(p, y_sc))
        if not np.isfinite(val) or val < 0:
            if strict:                                  # 網頁使用：異常就直接報錯，不顯示不可信的數字
                raise ValueError(f"第 {i + 1} 個月的預測值異常：{val}")
            if not np.isfinite(val):                    # 評估使用：無法計算的月份記為 NaN
                out.extend([float("nan")] * (n_months - i))
                break
            val = 0.0
            p = float(_scale(val, y_sc))
        out.append(val)
        ys.append(p)
        idx.append(idx[-1] + pd.DateOffset(months=1))
        if use_temp:
            ts.append(tt)
    return out



# ---------------------------------------------------------
# 網站：載入 Colab 訓練好的模型並預測（網頁上不訓練）
# ---------------------------------------------------------
MODEL_LABELS = {"lstm_notemp": "LSTM", "temp_adjusted": "氣溫修正模型"}
MODEL_LABELS_EN = {"lstm_notemp": "LSTM", "temp_adjusted": "Temperature-adjusted"}
MODEL_COLORS = {"lstm_notemp": "#d62728", "temp_adjusted": "#2ca02c"}
MAIN_MODEL = "lstm_notemp"


@st.cache_resource(show_spinner=False)
def load_bundle(slug):
    """讀取 models/<區域>.json 與兩個 .npz 權重檔（純 numpy，不需要 TensorFlow）。"""
    base = Path(__file__).resolve().parent / MODEL_DIR_NAME
    path = base / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到模型設定檔 {MODEL_DIR_NAME}/{slug}.json")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    models = {}
    for variant, info in cfg["variants"].items():
        if variant != "notemp":                       # 含氣溫的 LSTM 已不使用，權重檔可以不上傳
            continue
        wp = base / info["weights"]
        if not wp.exists():
            raise FileNotFoundError(f"找不到模型權重檔 {MODEL_DIR_NAME}/{info['weights']}")
        models[variant] = NumpyLSTM.from_npz(wp)
    return cfg, models


@st.cache_data(show_spinner=False)
def load_site_temperature():
    path = Path(__file__).resolve().parent / MODEL_DIR_NAME / "temperature.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到氣溫表 {MODEL_DIR_NAME}/temperature.csv")
    with open(path, "rb") as f:
        return load_temperature_file(f)


def predict_site(series, temp_s, cfg, models, n_months=12, use_known=False):
    """用 Colab 訓練好的模型，預測 series 之後的 n_months 個月。回傳 (月份清單, {模型代號: 預測值})。"""
    vals = series.to_numpy(dtype=float)
    if len(vals) < LOOKBACK_MONTHS:
        raise ValueError(f"至少需要 {LOOKBACK_MONTHS} 個月的資料，目前只有 {len(vals)} 個月")
    recent = vals[-LOOKBACK_MONTHS:]
    if not np.isfinite(recent).all() or (recent < 0).any():
        raise ValueError("最近 12 個月的資料含有 NaN 或負值")

    last = series.index[-1]
    future = [last + pd.DateOffset(months=i) for i in range(1, n_months + 1)]
    align_temperature(series.iloc[-LOOKBACK_MONTHS:], temp_s)                   # 確認氣溫表涵蓋最近 12 個月

    out = {}
    out["lstm_notemp"] = forecast_recursive(models["notemp"], vals, series.index, n_months, cfg["y_scaler"])
    tm = cfg["temp_model"]
    _, adj = forecast_temp_adjusted(series, temp_s, n_months, tm["beta_all"],
                                    {int(k): v for k, v in tm["climatology_all"].items()},
                                    temp_s if use_known else None)
    out["temp_adjusted"] = adj
    return [m.strftime("%Y-%m") for m in future], out


def _metrics_table_bundle(cfg):
    rows = []
    for kind, kname in [("one_step", "單步預測（每個月都用真實歷史資料）"),
                        ("recursive", "連續預測（從期初一路預測整段期間）")]:
        for key, label in MODEL_LABELS.items():
            for split, sname in [("val", "驗證期"), ("test", "測試期")]:
                m = cfg["metrics"][kind][key][split]
                rows.append({"預測方式": kname, "模型": label, "資料集": sname,
                             "MAPE (%)": round(m["MAPE_%"], 2), "RMSE": round(m["RMSE"]), "MAE": round(m["MAE"])})
    return pd.DataFrame(rows)


def _frame_from_bundle(cfg):
    fr = pd.DataFrame(cfg["predictions"])
    fr.index = pd.to_datetime(fr["month"] + "-01")
    return fr


def _plot_predictions_mpl(frame, unit):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame.index, frame["actual"], color="dodgerblue", linewidth=2.5, marker="o", label="Actual")
    for k in [c for c in MODEL_LABELS if c in frame.columns]:
        ax.plot(frame.index, frame[k], color=MODEL_COLORS[k], marker="x", linestyle="--", label=MODEL_LABELS_EN[k])
    test_start = frame.index[frame["split"] == "test"][0]
    ax.axvline(test_start, color="gray", linestyle=":", label="Test start")
    ax.set_title("Monthly Forecast Result (Validation & Test)")
    ax.set_xlabel("Month"); ax.set_ylabel(f"Monthly value ({_en_unit(unit)})")
    ax.grid(True, alpha=0.3); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    return fig


def _show_predictions(frame, unit):
    """驗證期＋測試期：真實值與各模型的預測對照（資料來自 Colab 的評估結果）。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.pyplot(_plot_predictions_mpl(frame, unit))
        return
    u = _en_unit(unit)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["actual"], mode="lines+markers", name="Actual",
                             line=dict(color="dodgerblue", width=3.5), marker=dict(size=10),
                             hovertemplate=f"%{{y:,.0f}} {u}"))
    for k in [c for c in MODEL_LABELS if c in frame.columns]:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[k], mode="lines+markers", name=MODEL_LABELS_EN[k],
                                 line=dict(color=MODEL_COLORS[k], width=2.5, dash="dash"),
                                 marker=dict(size=8, symbol="x"), hovertemplate=f"%{{y:,.0f}} {u}"))
    test_start = frame.index[frame["split"] == "test"][0].strftime("%Y-%m-%d")
    last = frame.index[-1].strftime("%Y-%m-%d")
    fig.add_shape(type="rect", xref="x", yref="paper", x0=test_start, x1=last, y0=0, y1=1,
                  fillcolor="rgba(120,120,120,0.10)", line_width=0, layer="below")
    fig.add_shape(type="line", xref="x", yref="paper", x0=test_start, x1=test_start, y0=0, y1=1,
                  line=dict(color="gray", width=1.5, dash="dot"))
    fig.add_annotation(x=test_start, y=1, xref="x", yref="paper", text="Test period", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color="gray", size=12))
    _style_prediction_fig(fig, "Monthly Forecast Result (Validation & Test)", unit)
    st.plotly_chart(fig, use_container_width=True)


def _plot_forecast_mpl(series, months, lines, unit):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime([m + "-01" for m in months])
    prev = [(pd.to_datetime(m + "-01"), float(series.loc[pd.to_datetime(m + "-01") - pd.DateOffset(years=1)]))
            for m in months if pd.to_datetime(m + "-01") - pd.DateOffset(years=1) in series.index]
    if prev:
        ax.plot([p[0] for p in prev], [p[1] for p in prev], color="dodgerblue", linewidth=2.5, marker="s", label="Actual (Last Year)")
    for k, v in lines.items():
        ax.plot(x, list(v), color=MODEL_COLORS[k], marker="o", linestyle="--", label=MODEL_LABELS_EN[k])
    ax.set_title("Monthly Forecast Result"); ax.set_xlabel("Month"); ax.set_ylabel(f"Monthly value ({_en_unit(unit)})")
    ax.grid(True, alpha=0.3); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    return fig


def _show_forecast(series, months, lines, unit):
    """未來月份預測：去年同月真實值與各模型的預測值。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.pyplot(_plot_forecast_mpl(series, months, lines, unit))
        return
    u = _en_unit(unit)
    x = pd.to_datetime([m + "-01" for m in months])
    prev_x, prev_vals = [], []
    for m in months:
        dt0 = pd.to_datetime(m + "-01") - pd.DateOffset(years=1)
        if dt0 in series.index:
            prev_x.append(pd.to_datetime(m + "-01")); prev_vals.append(float(series.loc[dt0]))
    fig = go.Figure()
    if prev_vals:
        fig.add_trace(go.Scatter(x=prev_x, y=prev_vals, mode="lines+markers", name="Actual (Last Year)",
                                 line=dict(color="dodgerblue", width=3.5), marker=dict(size=10, symbol="square"),
                                 hovertemplate=f"%{{y:,.0f}} {u}"))
    for k, v in lines.items():
        fig.add_trace(go.Scatter(x=x, y=list(v), mode="lines+markers", name=MODEL_LABELS_EN[k],
                                 line=dict(color=MODEL_COLORS[k], width=3, dash="dash" if k == MAIN_MODEL else "dot"),
                                 marker=dict(size=9), hovertemplate=f"%{{y:,.0f}} {u}"))
    main = lines[MAIN_MODEL]
    k0 = int(pd.Series(main).astype(float).idxmax())
    fig.add_annotation(x=x[k0], y=float(main[k0]), text=f"Peak: {float(main[k0]):,.0f} {u}", showarrow=True,
                       arrowhead=2, ax=0, ay=-40, font=dict(color="crimson", size=13))
    _style_prediction_fig(fig, "Monthly Forecast Result", unit)
    fig.update_xaxes(tickmode="array", tickvals=list(x), ticktext=[d.strftime("%Y-%m") for d in x])
    st.plotly_chart(fig, use_container_width=True)


def _source_campus():
    file = st.file_uploader("請上傳校園電度電費 Excel 統計檔案", type=["xlsx", "xls"], key="monthly_campus_excel")
    if file is None:
        st.info("💡 請先上傳 107～113 年的電度電費 Excel，系統會自動取出各年 12 個月的用電度數。")
        return None
    try:
        xl = pd.ExcelFile(file)
        default = CAMPUS_SHEET if CAMPUS_SHEET in xl.sheet_names else xl.sheet_names[0]
        sheet = st.selectbox("資料工作表 (Sheet)", xl.sheet_names, index=xl.sheet_names.index(default),
                             key="monthly_campus_sheet")
        location = st.selectbox("分析地點 / 區域 / 大樓", CAMPUS_LOCATIONS, key="monthly_campus_loc")
        series = load_campus_series(xl, sheet, location)
    except Exception as e:
        st.error(f"❌ 讀取校園 Excel 失敗：{e}")
        return None
    return series, f"度數（{location}）", "kWh", ("campus", file.name, sheet, location)


def render_monthly_tab():
    st.caption("模型已在 Google Colab 訓練完成並存成檔案，網頁只負責載入模型做預測，不會在網頁上訓練。")
    got = _source_campus()
    if got is None:
        return
    series, label, unit, key = got
    location = key[3]
    slug = LOCATION_SLUGS.get(location)

    try:
        cfg, models = load_bundle(slug)
        temp_s = load_site_temperature()
    except Exception as e:
        st.error(f"❌ 無法載入模型：{e}")
        st.info("請先執行 Colab 筆記本（EMS_Monthly_LSTM_Colab.ipynb），把下載的 models 資料夾放進 GitHub，和 app.py 同一層。")
        return

    st.success(f"共 {len(series)} 個月（{series.index[0]:%Y-%m} ～ {series.index[-1]:%Y-%m}）｜"
               f"模型於 {cfg['trained_at']} 在 Colab 訓練，訓練資料 {cfg['data_range'][0]} ～ {cfg['data_range'][1]}")
    if series.index[-1].strftime("%Y-%m") != cfg["data_range"][1]:
        st.warning("⚠️ 你上傳的資料期間與模型訓練時不同。模型仍會用最近 12 個月預測，但建議重新在 Colab 訓練。")

    st.subheader("模型評估（在 Colab 以驗證期與測試期資料計算）")
    one = cfg["metrics"]["one_step"]
    st.write(f"測試期（單步）MAPE：LSTM **{one['lstm_notemp']['test']['MAPE_%']:.2f}%**、"
             f"氣溫修正模型 **{one['temp_adjusted']['test']['MAPE_%']:.2f}%**")
    with st.expander("📋 點此展開／收合模型評估指標表 (Metrics)", expanded=False):
        st.dataframe(_metrics_table_bundle(cfg), use_container_width=True, hide_index=True)
    _show_predictions(_frame_from_bundle(cfg), unit)
    st.caption("測試期只有約 11 個月，樣本不多。LSTM 只使用過去的用電量；氣溫修正模型是簡單公式（去年同月用電量，依今年與去年的氣溫差修正）。"
               "評估時，未來氣溫用「訓練期的同月平均氣溫」，沒有偷看實際氣溫。")

    st.subheader("預測未來月份（未來 12 個月）")
    future = [series.index[-1] + pd.DateOffset(months=i) for i in range(1, 13)]
    use_known = False
    if all((m in temp_s.index) and np.isfinite(temp_s[m]) for m in future):
        use_known = st.checkbox("使用氣溫表中的實際氣溫（預設用歷史同月平均氣溫）", value=False, key="monthly_use_known")
    try:
        months, lines = predict_site(series, temp_s, cfg, models, 12, use_known)
    except Exception as e:
        st.error(f"❌ 預測失敗：{e}")
        return
    tbl = pd.DataFrame({"月份": months, **{f"{MODEL_LABELS[k]} ({unit})": [round(v, 2) for v in vals]
                                          for k, vals in lines.items()}})
    with st.expander("📋 點此展開／收合預測數據表格", expanded=False):
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    _show_forecast(series, months, lines, unit)
    st.download_button("下載預測結果 CSV", tbl.to_csv(index=False).encode("utf-8-sig"), "monthly_forecast.csv", "text/csv")



st.set_page_config(page_title="微電網 AI 預測與分析系統", layout="wide")

# 💡 引入 Google Fonts (Noto Sans TC 思源黑體) 並套用到全站
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans TC', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ 微電網 EMS 智慧用電分析與 AI 預測平台")
st.markdown("結合**校園長短期用電統計分析**與 **月度用電量 AI 預測**。")

tab1, tab2 = st.tabs([
    "📈 1. 歷史用電統計與動態趨勢分析", 
    "🚀 2. 月度用電量 AI 預測系統"
])

with tab1:
    st.header("📈 歷史用電統計與動態趨勢分析")
    
    with st.expander("📌 點此查看【歷史統計檔案】的規格與欄位說明", expanded=False):
        st.markdown("""
        * **檔案格式**：Excel 試算表（`.xlsx` 或 `.xls`）。
        * **工作表名稱（Sheet）**：需包含各月份用電度數彙整表（例如 `107-113各月度數費用合併`）。
        """)

    uploaded_excel_tab1 = st.file_uploader(
        "請上傳校園電度電費 Excel 統計檔案", 
        type=["xlsx", "xls"], 
        key="excel_uploader_tab1"
    )

    if uploaded_excel_tab1 is not None:
        try:
            xls = pd.ExcelFile(uploaded_excel_tab1)
            default_sheet = '107-113各月度數費用合併' if '107-113各月度數費用合併' in xls.sheet_names else xls.sheet_names[0]
            sheet_to_use = st.selectbox("請選擇資料工作表 (Sheet)", xls.sheet_names, index=xls.sheet_names.index(default_sheet) if default_sheet in xls.sheet_names else 0, key="sheet_tab1")
            
            df_excel = pd.read_excel(uploaded_excel_tab1, sheet_name=sheet_to_use, header=None)
            st.success(f"✅ 成功讀取檔案：{uploaded_excel_tab1.name} (工作表: {sheet_to_use})")

            col_y1, col_y2 = st.columns(2)
            with col_y1:
                selected_year = st.selectbox("選擇分析西元年份", [2018, 2019, 2020, 2021, 2022, 2023, 2024], index=6, key="trend_year")
                mco_year_map = {
                    2018: '107年', 2019: '108年', 2020: '109年', 
                    2021: '110年', 2022: '111年', 2023: '112年', 2024: '113年'
                }
                target_mco = mco_year_map.get(selected_year, '113年')

            with col_y2:
                location_options = ['不分區 (全校總用電)', '行政區', '商學', '校友', '男', '女', '中科', '文創']
                selected_location = st.selectbox("選擇分析地點 / 區域 / 大樓", location_options, key="trend_location")

            target_row_data = _campus_year_row(df_excel, target_mco, selected_location)

            if target_row_data is not None:
                months_range = pd.date_range(start=f'{selected_year}-01-01', end=f'{selected_year}-12-01', freq='MS')
                df_target = pd.DataFrame({'load_kWh': pd.to_numeric(target_row_data, errors='coerce')}, index=months_range)
                df_target = df_target.dropna()

                if len(df_target) > 0:
                    total_load = df_target['load_kWh'].sum()
                    mean_load = df_target['load_kWh'].mean()
                    max_load = df_target['load_kWh'].max()
                    min_load = df_target['load_kWh'].min()
                    max_month = df_target['load_kWh'].idxmax().strftime('%Y-%m')
                    min_month = df_target['load_kWh'].idxmin().strftime('%Y-%m')

                    st.markdown("---")
                    st.subheader(f"📊 {selected_year} 年 ({target_mco}) - 【{selected_location}】 用電統計摘要")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("區間總用電量", f"{total_load:,.2f} kWh")
                    col2.metric("每月平均用電", f"{mean_load:,.2f} kWh")
                    col3.metric("用電最高月份", max_month, f"{max_load:,.2f} kWh")
                    col4.metric("用電最低月份", min_month, f"{min_load:,.2f} kWh")

                    _plot_monthly_trend(df_target, selected_year, selected_location)
                else:
                    st.warning(f"⚠️ {selected_year} 年的【{selected_location}】目前沒有有效的月用電數值。")
            else:
                st.error(f"❌ 在上傳的檔案中找不到 {selected_year} 年 ({target_mco}) 對應的【{selected_location}】資料。")

        except Exception as e:
            st.error(f"❌ 解析上傳的 Excel 檔案時發生錯誤: {str(e)}")
    else:
        st.info("💡 請先上傳您的電度電費 Excel 統計檔案，即可檢視月統計與趨勢圖。")

with tab2:
    st.header("🚀 月度用電量 AI 預測系統")
    render_monthly_tab()