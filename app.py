# -------------------------------------------------------------------------
# 頁籤二：即時歷史負載上傳與 AI 預測系統 (繪圖區塊替換)
# -------------------------------------------------------------------------
            if result.get("ai_valid") == 1:
                st.success("✅ 推論成功：資料完全合規，AI 模型正常預測！")
                
                pred_loads = result.get("forecast_load_kw") or result.get("forecast_load_kWh")
                pred_times = result.get("prediction_timestamps")
                
                df_chart = pd.DataFrame({
                    "預測時間": pd.to_datetime(pred_times),
                    "AI 預測負載 (kW)": pred_loads
                }).set_index("預測時間")

                # 🎨 使用與論文級參考圖一致的 Matplotlib 專業繪圖風格
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # 繪製 AI 預測曲線 (採用參考圖的紅/粉色系與質感線條)
                ax.plot(
                    df_chart.index, 
                    df_chart['AI 預測負載 (kW)'], 
                    color='crimson', 
                    linewidth=1.8, 
                    label='AI Forecast Load (AI 預測負載)'
                )

                # 圖表標題與軸標籤設定
                ax.set_title('Full Test Set Load Forecasting Result', fontsize=14, fontweight='bold')
                ax.set_xlabel('Test Timeline', fontsize=12)
                ax.set_ylabel('Power Load (kW)', fontsize=12)
                
                # 加入細格線與圖例
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper right', fontsize=11)

                fig.tight_layout()
                st.pyplot(fig)
                
                with st.expander("查看詳細 96 步預測數據表格"):
                    st.dataframe(df_chart)
                    
                csv_data = df_chart.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 下載預測結果 CSV 檔案",
                    data=csv_data,
                    file_name="forecast_result.csv",
                    mime="text/csv",
                )