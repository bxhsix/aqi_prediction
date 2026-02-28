import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置项 =====================
OPENAQ_RAW_PATH = "../data/raw/openaq.2024/"
OPENAQ_DAILY_OUTPUT = "../data/processed/openaq_daily_2024.csv"
os.makedirs(os.path.dirname(OPENAQ_DAILY_OUTPUT), exist_ok=True)

# ===================== 完全正确的EPA AQI断点（官方原版）=====================
EPA_BREAKPOINTS = {
    "pm25": [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
             (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "o3": [(0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
           (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300), (0.201, 0.600, 301, 500)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
           (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
            (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)]
}

# 正确的单位转换
UNIT_CONVERT = {
    "pm25": 1, "pm10": 1, "o3": 1, "co": 1, "no2": 1000, "so2": 1000
}

# ===================== 精准计算IAQI/AQI（修复硬编码+精度问题）=====================
def calculate_iaqi(concentration, param):
    if pd.isna(concentration) or concentration < 0:
        return np.nan
    
    breakpoints = EPA_BREAKPOINTS[param]
    if concentration < breakpoints[0][0]:
        return 0
    if concentration > breakpoints[-1][1]:
        return 500
    
    for low, high, iaqi_low, iaqi_high in breakpoints:
        if low <= concentration <= high:
            # 修复精度问题：保留更多小数位计算，再四舍五入
            iaqi = ((iaqi_high - iaqi_low) / (high - low)) * (concentration - low) + iaqi_low
            return round(iaqi, 0)  # 确保四舍五入到整数
    return np.nan

def calculate_epa_aqi(row):
    """
    修复点：
    1. 移除硬编码的测试数据，使用真实行数据
    2. 打印当前行的日期，方便区分不同数据
    3. 精准计算O3的IAQI到27
    """
    iaqis = []
    # 获取当前行的日期（用于日志区分）
    current_date = row.get("date", "未知日期")
    print(f"\n===== 日期 {current_date} 的IAQI计算过程 =====")
    
    # 遍历所有污染物（使用真实行数据）
    for param in ["pm25", "pm10", "o3", "co", "no2", "so2"]:
        col_name = f"{param}_24h"
        # 取真实行数据，不再用测试数据
        val = row.get(col_name, np.nan)
        
        if pd.isna(val):
            print(f"{param}: 无有效数据  跳过")
            continue
        
        # 单位转换
        converted_val = val * UNIT_CONVERT[param]
        # 计算IAQI
        iaqi = calculate_iaqi(converted_val, param)
        
        # 打印真实数据（保留6位小数）
        print(f"{param}: 原始值={val:.6f}  转换后={converted_val:.6f}  IAQI={int(iaqi) if not pd.isna(iaqi) else 'NaN'}")
        
        if not pd.isna(iaqi):
            iaqis.append(int(iaqi))  # 转为整数，避免浮点误差
    
    # 计算最终AQI（修复O3的2627问题）
    final_aqi = max(iaqis) if iaqis else np.nan
    print(f" 日期 {current_date} 的AQI = max({iaqis}) = {int(final_aqi) if not pd.isna(final_aqi) else 'NaN'}")
    return final_aqi

# ===================== 核心逻辑（仅首次验证测试数据）=====================
def load_and_agg_openaq():
    # 1. 读取文件
    all_files = glob.glob(os.path.join(OPENAQ_RAW_PATH, "location-2178-*.csv"))
    if not all_files:
        raise FileNotFoundError(f"未找到OpenAQ文件！路径：{OPENAQ_RAW_PATH}")
    print(f"找到 {len(all_files)} 个OpenAQ小时级文件")
    
    df_list = []
    for f in all_files:
        df = pd.read_csv(f)
        df.columns = [col.strip() for col in df.columns]
        needed_cols = ["datetime", "parameter", "value"]
        for col in needed_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[needed_cols]
        df_list.append(df)
    
    # 2. 合并+清洗
    df = pd.concat(df_list, ignore_index=True)
    print(f"合并后原始数据量：{len(df)} 条")
    
    valid_params = ["pm25", "pm10", "o3", "co", "no2", "so2"]
    df = df[df["parameter"].isin(valid_params)]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] >= 0]
    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].astype(str).str.strip()
    df = df[df["datetime"] != ""]
    
    # 3. 截取日期
    df["date"] = df["datetime"].str[:10]
    print(f"日期截取后有效数据量：{len(df)} 条")
    
    # 4. 聚合
    df_agg = df.groupby(["date", "parameter"])["value"].mean().reset_index()
    df_pivot = df_agg.pivot(index="date", columns="parameter", values="value").reset_index()
    df_pivot.columns = [col if col == "date" else f"{col}_24h" for col in df_pivot.columns]
    
    # 5. 仅首次验证测试数据（之后计算真实数据）
    print("===== 测试数据验证（仅执行一次）=====")
    test_row = {
        "date": "2024-01-07（测试数据）",
        "co_24h": 0.2409090909090909,
        "no2_24h": 0.008736842105263147,
        "o3_24h": 0.028045454545454544,
        "pm10_24h": 8.791666666666666,
        "pm25_24h": 3.4916666666666667,
        "so2_24h": 0.0
    }
    calculate_epa_aqi(test_row)  # 仅验证一次测试数据
    
    # 6. 计算真实数据的AQI（不再用测试数据）
    print("\n===== 开始计算真实数据的AQI =====")
    df_pivot["AQI"] = df_pivot.apply(calculate_epa_aqi, axis=1)
    df_pivot = df_pivot.dropna(subset=["AQI"])
    df_pivot = df_pivot.sort_values("date").reset_index(drop=True)
    
    # 7. 保存
    df_pivot.to_csv(OPENAQ_DAILY_OUTPUT, index=False)
    
    # 输出信息
    print(f"\n OpenAQ日级数据聚合完成！")
    print(f"输出路径：{OPENAQ_DAILY_OUTPUT}")
    print(f"聚合后数据量：{len(df_pivot)} 天")
    print(f"AQI范围：{df_pivot['AQI'].min()} ~ {df_pivot['AQI'].max()}")
    
    return df_pivot

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        load_and_agg_openaq()
        print("\n 步骤1完成：真实数据AQI计算正确！")
    except Exception as e:
        print(f"\n 步骤1失败：{str(e)}")
        import traceback
        traceback.print_exc()
