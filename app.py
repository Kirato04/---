# =========================================================
# 微電網 EMS 系統 - 智慧校園用電分析與 AI 預測平台 (app.py)
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from infer import EMSLoadPredictor

# 設定網頁標題與排版
st.set_page_config(page_title="微電網 AI 預測與分析系統", layout="wide")
st.title("⚡ 微電網 EMS 智慧用電分析與 AI 預測平台")
st.markdown("結合**校園長短期用電統計分析**與 **LSTM 五大防禦即時預測**。")

@st.cache_resource
def load_predictor():
    return EMSLoadPredictor()

predictor = load_predictor()

# 側邊欄：系統狀態檢視
st.sidebar.header("🛡️ 系統防禦與狀態")
if predictor.is_ready:
    st.sidebar.success("✅ AI 模型與防禦引擎已就緒")
    st.sidebar.info(f"模型版本: {predictor.model_version}\n預測步長: {predictor.horizon_min} 分鐘")
    if hasattr(predictor, "get_system_audit_report"):
        with st.sidebar.expander("查看系統稽核計數"):
            st.json(predictor.get_system_audit_report())
else:
    st.sidebar.error(f"❌ 錯誤: {predictor.init_error}")

# 建立功能頁籤 (Tabs)
tab1, tab2 = st.tabs([
    "📈 1. 歷史用電統計與動態趨勢分析", 
    "🚀 2. 即時歷史負載上傳與 AI 預測系統 "
])

# -------------------------------------------------------------------------
# 頁籤一：歷史用電統計與動態分析
# -------------------------------------------------------------------------
with tab1:
    st.header("📈 歷史用電統計與動態趨勢分析")
    
    with st.expander("📌 點此查看【歷史統計檔案】的規格與欄位說明（必讀）", expanded=False):
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

            target_row_data = None

            if selected_location == '不分區 (全校總用電)':
                records_target = []
                for idx, row in df_excel.iterrows():
                    val = str(row.iloc[0]).strip()
                    if val == '總和':
                        for prev in range(idx-1, max(-1, idx-10), -1):
                            if target_mco in str(df_excel.iloc[prev, 0]):
                                records_target.append(row.iloc[1:13].values)
                                break
                if len(records_target) >= 2:
                    table1 = np.array(records_target[0], dtype=float)
                    table2 = np.array(records_target[1], dtype=float)
                    target_row_data = table1 + table2
            else:
                current_year_block = None
                for idx, row in df_excel.iterrows():
                    val = str(row.iloc[0]).strip()
                    if target_mco in val and '度數' not in val and '費用' not in val:
                        current_year_block = target_mco
                        continue
                    elif any(y in val for y in ['107年', '108年', '109年', '110年', '111年', '112年', '113年']) and val != target_mco:
                        current_year_block = None

                    if current_year_block == target_mco and val == selected_location:
                        target_row_data = row.iloc[1:13].values
                        break

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

                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df_target.index, df_target['load_kWh'], marker='o', color='forestgreen', linewidth=2.5, markersize=7, label=f'{selected_location} Load')

                    max_idx = df_target['load_kWh'].argmax()
                    ax.annotate(f'Peak: {max_load:,.0f} kWh',
                                 xy=(df_target.index[max_idx], max_load),
                                 xytext=(0, 15), textcoords='offset points', ha='center',
                                 fontweight='bold', color='crimson',
                                 arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))

                    ax.set_title(f'{selected_year} {selected_location} Monthly Electricity Load Analysis', fontsize=14, fontweight='bold')
                    ax.set_xlabel(f'Month ({selected_year})', fontsize=12)
                    ax.set_ylabel('Power Consumption (kWh / Month)', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.set_xticks(df_target.index)
                    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(df_target)])
                    ax.legend(loc='upper right', fontsize=11)

                    fig.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(f"⚠️ {selected_year} 年的【{selected_location}】目前沒有有效的月用電數值。")
            else:
                st.error(f"❌ 在上傳的檔案中找不到 {selected_year} 年 ({target_mco}) 對應的【{selected_location}】資料。")

        except Exception as e:
            st.error(f"❌ 解析上傳的 Excel 檔案時發生錯誤: {str(e)}")
    else:
        st.info("💡 請先上傳您的電度電費 Excel 統計檔案，即可檢視月統計與趨勢圖。")

# -------------------------------------------------------------------------
# 頁籤二：即時歷史負載上傳與 AI 預測系統 (支援 Excel 自動轉換)
# -------------------------------------------------------------------------
with tab2:
    st.header("🚀 即時歷史負載上傳與 AI 預測系統 (支援 Excel 自動轉換)")
    
    with st.expander("📌 點此查看【預測檔案】格式與轉換說明（必讀）", expanded=False):
        st.markdown("""
        * **支援檔案格式**：
          1. **CSV 檔案**：傳統高頻 15 分鐘負載數據（需至少 192 筆以上）。
          2. **Excel 檔案**：上傳後選擇大樓，系統會**自動根據你輸入的時間戳年份**轉換為 15 分鐘高頻序列供 AI 預測！
        """)

    file_type_choice = st.radio("請選擇您要上傳的檔案類型進行預測：", ["高頻 CSV 檔案", "長短期 Excel 統計檔案 (自動轉高頻)"], horizontal=True)

    raw_history = []
    
    # 讓使用者輸入時間戳 (系統會自動抓取其中的年份來對應 Excel 資料)
    input_timestamp = st.text_input("指定資料截止時間戳 (Timestamp)", value="2024-06-07 12:00:00", key="pred_ts")

    forecast_days_choice = st.slider(
        "選擇預測天數範圍 (Forecast Horizon)", 
        min_value=1, 
        max_value=7, 
        value=1, 
        step=1,
        help="可自由選擇在網頁上預測未來 1 天到 7 天的用電趨勢"
    )

    if file_type_choice == "高頻 CSV 檔案":
        uploaded_csv = st.file_uploader("請上傳包含 15 分鐘高頻歷史用電的 CSV 檔案", type=["csv"], key="pred_csv_uploader")
        if uploaded_csv is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_csv)
                st.success(f"✅ 成功讀取 CSV：{uploaded_csv.name} (共 {len(df_uploaded)} 筆資料)")
                numeric_columns = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    selected_col = st.selectbox("請選擇代表負載數值 (kW) 的欄位名稱", numeric_columns, key="pred_col_csv")
                    raw_history = df_uploaded[selected_col].dropna().tolist()
            except Exception as e:
                st.error(f"❌ 讀取 CSV 發生錯誤: {str(e)}")
        else:
            if st.button("📥 載入內建範例測試資料 (CSV 格式)"):
                raw_history = np.linspace(500, 800, 200).tolist()
                st.success(f"已載入內建範例數據（共 {len(raw_history)} 筆）")

    else:
        uploaded_excel_tab2 = st.file_uploader("請上傳您的電度電費 Excel 統計檔案 (.xlsx)", type=["xlsx", "xls"], key="pred_excel_uploader")
        
        if uploaded_excel_tab2 is not None:
            try:
                parsed_input_dt = pd.to_datetime(input_timestamp)
                if pd.isna(parsed_input_dt):
                    st.error("❌ 時間戳格式錯誤，請確保格式如：2024-06-07 12:00:00")
                    sel_year2 = 2024
                else:
                    sel_year2 = parsed_input_dt.year

                xls2 = pd.ExcelFile(uploaded_excel_tab2)
                default_sheet2 = '107-113各月度數費用合併' if '107-113各月度數費用合併' in xls2.sheet_names else xls2.sheet_names[0]
                sheet_to_use2 = st.selectbox("選擇資料工作表 (Sheet)", xls2.sheet_names, index=xls2.sheet_names.index(default_sheet2) if default_sheet2 in xls2.sheet_names else 0, key="sheet_tab2")
                
                df_excel2 = pd.read_excel(uploaded_excel_tab2, sheet_name=sheet_to_use2, header=None)
                
                mco_map2 = {2018: '107年', 2019: '108年', 2020: '109年', 2021: '110年', 2022: '111年', 2023: '112年', 2024: '113年'}
                t_mco2 = mco_map2.get(sel_year2, '113年')

                loc_opts2 = ['不分區 (全校總用電)', '行政區', '商學', '校友', '男', '女', '中科', '文創']
                sel_loc2 = st.selectbox("選擇轉換大樓/區域", loc_opts2, key="conv_loc")

                target_data2 = None
                if sel_loc2 == '不分區 (全校總用電)':
                    recs = []
                    for idx, row in df_excel2.iterrows():
                        val = str(row.iloc[0]).strip()
                        if val == '總和':
                            for prev in range(idx-1, max(-1, idx-10), -1):
                                if t_mco2 in str(df_excel2.iloc[prev, 0]):
                                    recs.append(row.iloc[1:13].values)
                                    break
                    if len(recs) >= 2:
                        target_data2 = np.array(recs[0], dtype=float) + np.array(recs[1], dtype=float)
                else:
                    cur_block = None
                    for idx, row in df_excel2.iterrows():
                        val = str(row.iloc[0]).strip()
                        if t_mco2 in val and '度數' not in val and '費用' not in val:
                            cur_block = t_mco2
                            continue
                        elif any(y in val for y in ['107年', '108年', '109年', '110年', '111年', '112年', '113年']) and val != t_mco2:
                            cur_block = None
                        if cur_block == t_mco2 and val == sel_loc2:
                            target_data2 = row.iloc[1:13].values
                            break

                if target_data2 is not None:
                    high_freq_list = []
                    for m_idx, m_kwh in enumerate(target_data2, start=1):
                        if pd.isna(m_kwh):
                            m_kwh = 0.0
                        days_in_m = calendar.monthrange(sel_year2, m_idx)[1]
                        intervals = days_in_m * 96
                        day_idx = np.arange(intervals)
                        hour_d = (day_idx % 96) * 0.25
                        
                        # 💡 修正：確保日負載特徵曲線的基準與數值比例正確（避免數值被灌水放大）
                        profile = 1.0 + 0.3 * np.sin(2 * np.pi * (hour_d - 6) / 24)
                        profile = np.clip(profile, 0.4, 1.6)
                        
                        avg_kw = m_kwh / (days_in_m * 24) if days_in_m > 0 else 0
                        p_mean = np.mean(profile)
                        if p_mean > 0:
                            profile = profile / p_mean
                        kw_series = avg_kw * profile
                        high_freq_list.extend(kw_series.tolist())

                    raw_history = high_freq_list
                    
                    # 💡 建立對應該年份的完整時間索引，確保切片能正確抓到該時間點的歷史
                    full_time_index = pd.date_range(start=f'{sel_year2}-01-01 00:00:00+08:00', periods=len(raw_history), freq='15min')
                    st.session_state['df_temp_history'] = pd.Series(raw_history, index=full_time_index)
                    
                    st.success(f"✅ 成功將 {sel_year2} 年【{sel_loc2}】轉換為高頻序列 (共 {len(raw_history)} 筆)")
                else:
                    st.warning(f"⚠️ 在檔案中找不到 {sel_year2} 年 ({t_mco2}) 對應的【{sel_loc2}】數值。")

            except Exception as e:
                st.error(f"❌ Excel 轉換處理時發生錯誤: {str(e)}")

    # 執行 AI 預測按鈕
    st.markdown("---")
    if st.button("🚀 開始執行 AI 預測與五大防禦檢核", type="primary"):
        if not predictor.is_ready:
            st.error(f"❌ 模型尚未就緒，無法進行預測。原因: {predictor.init_error}")
        elif len(raw_history) == 0:
            st.error("⚠️ 目前沒有可用的歷史負載資料，請先上傳檔案或完成資料轉換！")
        else:
            try:
                with st.spinner(f"正在進行未來 {forecast_days_choice} 天的五大防禦檢核與模型推論中..."):
                    
                    parsed_input_dt = pd.to_datetime(input_timestamp)
                    if parsed_input_dt.tzinfo is None:
                        parsed_input_dt = parsed_input_dt.tz_localize('Asia/Taipei')
                    
                    # 建立完整時間序列
                    sel_year_for_index = parsed_input_dt.year
                    full_time_index = pd.date_range(start=f'{sel_year_for_index}-01-01 00:00:00+08:00', periods=len(raw_history), freq='15min')
                    df_temp_history = pd.Series(raw_history, index=full_time_index)
                    
                    # 嚴格切出該時間戳「之前」的所有資料
                    sliced_history = df_temp_history[df_temp_history.index <= parsed_input_dt]
                    
                    # 💡 關鍵：如果切出來的資料大於 192 筆，我們精準取其「最後 192 筆」作為當下的 Lookback 視窗！
                    if len(sliced_history) >= predictor.lookback_steps:
                        history_input = sliced_history.iloc[-predictor.lookback_steps:].tolist()
                    else:
                        st.warning("⚠️ 指定的時間戳之前資料量不足 192 筆，已自動取該年度前 192 筆資料。")
                        history_input = raw_history[:predictor.lookback_steps]

                    # 呼叫推論引擎
                    result = predictor.forecast_load_kw(history_input, input_timestamp, forecast_days=forecast_days_choice)

                st.markdown("---")
                st.subheader("📊 預測結果與系統稽核")

                if result and result.get("ai_valid") == 1:
                    st.success("✅ 推論成功：資料完全合規，AI 模型正常預測！")
                    
                    pred_loads = result.get("forecast_load_kw") or result.get("forecast_load_kWh")
                    pred_times = result.get("prediction_timestamps")
                    
                    df_chart = pd.DataFrame({
                        "預測時間": pd.to_datetime(pred_times),
                        "AI 預測負載 (kW)": pred_loads
                    }).set_index("預測時間")

                    # 繪製圖表
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(
                        df_chart.index, 
                        df_chart['AI 預測負載 (kW)'], 
                        color='crimson', 
                        linewidth=1.8, 
                        label=f'AI Forecast Load ({forecast_days_choice}-Day Horizon)'
                    )
                    ax.set_title(f'Multi-Step Load Forecasting Result ({forecast_days_choice} Days)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Test Timeline', fontsize=12)
                    ax.set_ylabel('Power Load (kW)', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.legend(loc='upper right', fontsize=11)

                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    with st.expander(f"查看詳細 {len(pred_loads)} 步預測數據表格"):
                        st.dataframe(df_chart)
                        
                    csv_data = df_chart.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 下載預測結果 CSV 檔案",
                        data=csv_data,
                        file_name=f"forecast_result_{forecast_days_choice}days.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("❌ 系統防禦機制已攔截！(ai_valid = 0)")
                    st.warning(f"**觸發錯誤原因**：{result.get('error_reason')}")
                    st.info(f"🛡️ **啟動備援策略 (Fallback)**：`{result.get('fallback_strategy')}`")

            except Exception as e:
                st.error(f"❌ 執行預測過程發生例外錯誤: {str(e)}")