# =========================================================
# Excel 轉 CSV 轉換工具 (convert_excel_to_csv.py)
# =========================================================

import pandas as pd

excel_path = '107-113年電度電費數據20241115更新至10月-1.xlsx'
sheet_name = '107-113各月度數費用合併'

try:
    # 1. 讀取 Excel 檔案
    print("正在讀取 Excel 檔案...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # 2. 檢視前幾行結構，過濾掉不需要的標題與合併儲存格
    print("原始資料大小:", df.shape)
    
    # 範例：假設我們要萃取出商學大樓或其他特定區域歷年各月的用電數據
    # 尋找包含「商學」的列
    target_row = df[df.astype(str).apply(lambda x: x.str.contains('商學').any(), axis=1)]
    
    if not target_row.empty:
        print("✅ 成功找到目標區域資料！")
        print(target_row)
        
        # 轉成乾淨的 CSV 格式供網頁使用
        target_row.to_csv('cleaned_shangxue_load.csv', index=False, encoding='utf-8-sig')
        print("🎉 已成功將資料轉換並存檔為 'cleaned_shangxue_load.csv'！")
    else:
        print("⚠️ 未找到指定關鍵字，將整個工作表轉存為乾淨的標準 CSV 格式...")
        df.dropna(how='all').to_csv('cleaned_full_load.csv', index=False, encoding='utf-8-sig')
        print("🎉 已成功轉存為 'cleaned_full_load.csv'！")

except Exception as e:
    print(f"❌ 轉檔過程發生錯誤: {str(e)}")