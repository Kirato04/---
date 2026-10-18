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
# 頁籤二：即時歷史負載上傳與 AI 預測系統
# -------------------------------------------------------------------------
with tab2:
    st.header("🚀 即時歷史負載上傳與 AI 預測系統")
    
    with st.expander("📌 點此查看【預測檔案】格式與轉換說明", expanded=False):
        st.markdown("""
        * **支援檔案格式**：
          1. **CSV 檔案**：傳統高頻 15 分鐘負載數據（需包含 Timestamp 欄位）。
          2. **Excel 檔案**：上傳後系統會自動轉換為完整年度高頻序列，並根據你選擇的時間點自動切片預測！
        """)

    file_type_choice = st.radio("請選擇您要上傳的檔案類型進行預測：", ["高頻 CSV 檔案", "長短期 Excel 統計檔案 "], horizontal=True)

    if 'active_history_series' not in
                st.error(f"❌ 執行預測過程發生例外錯誤: {str(e)}")
