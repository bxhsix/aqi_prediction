import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置项 =====================
NOAA_RAW_PATH = "../data/raw/noaa.2024/72365023050.csv"
OPENAQ_DAILY_PATH = "../data/processed/openaq_daily_2024.csv"
FINAL_OUTPUT_PATH = "../data/processed/aqi_train_data.csv"
os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)

# ===================== 读取并清洗NOAA数据（字符串日期）=====================
def load_and_clean_noaa():
    # 1. 读取NOAA数据
    df = pd.read_csv(NOAA_RAW_PATH)
    print(f"读取NOAA数据：{len(df)} 条")
    
    # 2. 数据清洗
    df.columns = [col.strip() for col in df.columns]  # 列名去空格
    # DATE列本身就是字符串（2024-01-01），无需转换，直接重命名为date
    df.rename(columns={"DATE": "date"}, inplace=True)
    # 提取核心气象特征
    weather_cols = ["date", "TEMP", "DEWP", "SLP", "WDSP", "MAX", "MIN", "PRCP"]
    df = df[weather_cols]
    # 转换数值类型（处理NOAA数据的字符串空格）
    for col in ["TEMP", "DEWP", "SLP", "WDSP", "MAX", "MIN", "PRCP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 处理NOAA异常值（999.9表示缺失）
    df = df.replace(999.9, np.nan)
    # 缺失值填充（用中位数）
    df = df.fillna(df.median(numeric_only=True))
    # 确保date列是字符串类型
    df["date"] = df["date"].astype(str).str.strip()
    
    print(f"NOAA数据清洗完成：有效数据 {len(df)} 条")
    print(f"NOAA日期格式：字符串（例如：{df['date'].iloc[0]}）")
    return df

# ===================== 合并数据+特征工程（字符串日期）=====================
def merge_and_feature_engineering(noaa_df, openaq_df):
    # 1. 按字符串日期内连接（核心：和NOAA格式完全对齐）
    df_merged = pd.merge(noaa_df, openaq_df, on="date", how="inner")
    print(f"NOAA+OpenAQ合并后数据量：{len(df_merged)} 条")
    if len(df_merged) == 0:
        raise ValueError(" 无重合日期数据！请检查两份数据的日期范围是否一致")
    
    # 2. 构建气象衍生特征
    df_merged["TEMP_DIFF"] = df_merged["MAX"] - df_merged["MIN"]  # 气温日较差
    df_merged["PRCP_BINARY"] = (df_merged["PRCP"] > 0).astype(int)  # 是否降雨（0/1）
    
    # 3. 按日期排序（为滞后特征做准备）
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    
    # 4. 构建AQI相关滞后特征
    lag_cols_aqi = ["AQI", "pm25_24h", "pm10_24h", "o3_24h", "co_24h", "no2_24h", "so2_24h"]
    for col in lag_cols_aqi:
        df_merged[f"{col}_lag1"] = df_merged[col].shift(1)  # 滞后1天
    
    # 5. 构建NOAA气象特征滞后特征（核心新增：明确写入NOAA滞后数据）
    lag_cols_noaa = ["TEMP", "DEWP", "SLP", "WDSP", "MAX", "MIN", "PRCP", "TEMP_DIFF", "PRCP_BINARY"]
    for col in lag_cols_noaa:
        df_merged[f"{col}_lag1"] = df_merged[col].shift(1)  # 气象特征滞后1天
        df_merged[f"{col}_lag2"] = df_merged[col].shift(2)  # 可选：滞后2天，按需扩展
        # 可根据需求添加更多滞后天数（如lag3）
    
    # 6. 删除滞后特征导致的缺失值行
    df_merged = df_merged.dropna()
    
    # 7. 保存最终训练数据
    df_merged.to_csv(FINAL_OUTPUT_PATH, index=False)
    print(f"最终训练数据保存完成！输出路径：{FINAL_OUTPUT_PATH}")
    print(f"最终数据量：{len(df_merged)} 条")
    print(f"特征总数：{len(df_merged.columns)}")
    print(f"\n数据字段预览：")
    print(df_merged.columns.tolist())
    print(f"\nNOAA滞后特征列表：")
    noaa_lag_features = [col for col in df_merged.columns if any(lag in col for lag in ["_lag1", "_lag2"]) and col.split("_lag")[0] in lag_cols_noaa]
    print(noaa_lag_features)
    print(f"\n数据前5行预览：")
    print(df_merged.head())
    return df_merged

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        # 1. 读取并清洗NOAA数据
        noaa_df = load_and_clean_noaa()
        # 2. 读取步骤1的OpenAQ日级数据（字符串日期）
        openaq_df = pd.read_csv(OPENAQ_DAILY_PATH)
        openaq_df["date"] = openaq_df["date"].astype(str).str.strip()  # 确保是字符串
        # 3. 合并数据+特征工程
        merge_and_feature_engineering(noaa_df, openaq_df)
        print("\n 步骤2完成：NOAA+OpenAQ数据已合并（含NOAA滞后特征），训练数据集生成！")
    except FileNotFoundError as e:
        print(f"\n 步骤2失败：文件不存在 - {str(e)}")
        print("请先运行 step1_openaq_daily_agg.py 生成OpenAQ日级数据！")
    except Exception as e:
        print(f"\n 步骤2失败：{str(e)}")
        import traceback
        traceback.print_exc()
