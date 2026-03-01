import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置项 =====================
API_BASE_URL = "http://localhost:8000"
TRAIN_DATA_PATH = "../data/processed/aqi_train_data.csv"
# 页面配置
st.set_page_config(
    page_title="AQI Prediction System - Albuquerque",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 绘图配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

# ===================== 加载本地数据（可视化）=====================
@st.cache_data
def load_train_data():
    df = pd.read_csv(TRAIN_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

train_df = load_train_data()
# 新特征列表（与后端一致）
feature_cols = ['pm25_24h_lag1', 'pm10_24h_lag1', 'o3_24h_lag1', 'co_24h_lag1', 
                'no2_24h_lag1', 'so2_24h_lag1', 'TEMP_lag1', 'DEWP_lag1', 
                'SLP_lag1', 'WDSP_lag1', 'MAX_lag1', 'MIN_lag1', 
                'PRCP_lag1', 'TEMP_DIFF_lag1', 'PRCP_BINARY_lag1']

# ===================== 调用后端API =====================
def call_api(endpoint, method="GET", data=None):
    """通用API调用函数"""
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API调用失败：{str(e)}")
        return None

# ===================== AQI等级颜色映射 =====================
def get_aqi_color(aqi):
    if aqi <= 50:
        return "#00e400"  # 绿
    elif aqi <= 100:
        return "#ffff00"  # 黄
    elif aqi <= 150:
        return "#ff7e00"  # 橙
    elif aqi <= 200:
        return "#ff0000"  # 红
    elif aqi <= 300:
        return "#8f3f97"  # 紫
    else:
        return "#7e0023"  # 褐

# ===================== 页面导航 =====================
st.sidebar.title(" AQI预测系统")
st.sidebar.markdown("### 美国阿尔伯克基市 - 基于NOAA+OpenAQ数据")
page = st.sidebar.radio("导航菜单", ["数据概览", "AQI预测", "历史统计", "健康建议"])
st.sidebar.divider()
st.sidebar.markdown("#### 数据来源")
st.sidebar.markdown("- NOAA Global Surface Summary of the Day")
st.sidebar.markdown("- OpenAQ Air Quality Data")
st.sidebar.markdown("#### 模型训练")
st.sidebar.markdown("- 多基础模型+AutoGluon AutoML")
st.sidebar.markdown("- 预测目标：EPA标准AQI（未来1天）")

# ===================== 页面1：数据概览 =====================
if page == "数据概览":
    st.title(" 数据概览")
    st.markdown("### 训练数据集基本信息（阿尔伯克基2024年）")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("数据总量", f"{len(train_df)} 条")
    with col2:
        st.metric("特征数量", f"{len(feature_cols)} 个")
    with col3:
        st.metric("时间范围", f"{train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')}")
    
    st.divider()
    # 历史AQI趋势图
    st.markdown("### 历史AQI趋势")
    fig, ax = plt.subplots()
    ax.plot(train_df["date"], train_df["AQI"], color="#1f77b4", linewidth=1, alpha=0.8)
    ax.axhline(y=50, color="#00e400", linestyle="--", label="Good(50)")
    ax.axhline(y=100, color="#ffff00", linestyle="--", label="Moderate(100)")
    ax.axhline(y=150, color="#ff7e00", linestyle="--", label="Unhealthy for Sensitive(150)")
    ax.set_xlabel("Date")
    ax.set_ylabel("EPA AQI")
    ax.set_title("2024 ALBUQUERQUE AQI Trend")
    ax.legend()
    st.pyplot(fig)
    
    # 特征与AQI相关性热力图（适配新特征）
    st.markdown("### 滞后特征与AQI相关性热力图")
    corr_cols = ["AQI"] + feature_cols[:6] + feature_cols[6:10]  # 选取核心特征展示
    corr_df = train_df[corr_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    
    # 数据预览
    st.markdown("### 训练数据预览（前10行）")
    st.dataframe(train_df.head(10), use_container_width=True)

# ===================== 页面2：AQI预测 =====================
elif page == "AQI预测":
    st.title(" 未来1天AQI预测")
    st.markdown("### 输入前1日滞后特征，点击预测获取结果")
    st.warning("提示：特征参数请填写与训练数据同量级的数值（可参考数据概览中的统计范围）")
    
    # 构建输入表单（分3栏布局，更紧凑）
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 污染物滞后特征
        pm25_24h_lag1 = st.number_input("PM2.5滞后值(µg/m)", value=10.0, step=0.1)
        pm10_24h_lag1 = st.number_input("PM10滞后值(µg/m)", value=20.0, step=0.1)
        o3_24h_lag1 = st.number_input("O3滞后值(ppm)", value=0.05, step=0.001)
        co_24h_lag1 = st.number_input("CO滞后值(ppm)", value=0.5, step=0.01)
        no2_24h_lag1 = st.number_input("NO2滞后值(ppm)", value=0.003, step=0.0001)
        so2_24h_lag1 = st.number_input("SO2滞后值(ppm)", value=0.0, step=0.0001)
    
    with col2:
        # 气象滞后特征（基础）
        TEMP_lag1 = st.number_input("日均气温滞后值(F)", value=50.0, step=0.1)
        DEWP_lag1 = st.number_input("露点温度滞后值(F)", value=30.0, step=0.1)
        SLP_lag1 = st.number_input("海平面气压滞后值(hPa)", value=1020.0, step=0.1)
        WDSP_lag1 = st.number_input("平均风速滞后值(knots)", value=5.0, step=0.1)
        MAX_lag1 = st.number_input("日最高温滞后值(F)", value=60.0, step=0.1)
        MIN_lag1 = st.number_input("日最低温滞后值(F)", value=40.0, step=0.1)
    
    with col3:
        # 气象滞后特征（衍生）
        PRCP_lag1 = st.number_input("降雨量滞后值(in)", value=0.0, step=0.01)
        TEMP_DIFF_lag1 = st.number_input("气温日较差滞后值(F)", value=20.0, step=0.1)
        PRCP_BINARY_lag1 = st.selectbox("是否降雨滞后值", [0, 1], index=0)
    
    # 构造请求数据（严格匹配后端新特征列表）
    request_data = {
        "pm25_24h_lag1": pm25_24h_lag1,
        "pm10_24h_lag1": pm10_24h_lag1,
        "o3_24h_lag1": o3_24h_lag1,
        "co_24h_lag1": co_24h_lag1,
        "no2_24h_lag1": no2_24h_lag1,
        "so2_24h_lag1": so2_24h_lag1,
        "TEMP_lag1": TEMP_lag1,
        "DEWP_lag1": DEWP_lag1,
        "SLP_lag1": SLP_lag1,
        "WDSP_lag1": WDSP_lag1,
        "MAX_lag1": MAX_lag1,
        "MIN_lag1": MIN_lag1,
        "PRCP_lag1": PRCP_lag1,
        "TEMP_DIFF_lag1": TEMP_DIFF_lag1,
        "PRCP_BINARY_lag1": PRCP_BINARY_lag1
    }
    
    # 预测按钮
    if st.button(" 开始预测", use_container_width=True):
        with st.spinner("正在调用模型预测..."):
            result = call_api("predict_aqi", method="POST", data=request_data)
        if result and result["code"] == 200:
            st.success("预测完成！")
            # 展示预测结果（带颜色和样式）
            predict_aqi = result["data"]["predict_aqi"]
            aqi_grade = result["data"]["aqi_grade"]
            health_info = result["data"]["health_info"]
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                    <h2>未来1天AQI预测值：<span style="color: {get_aqi_color(predict_aqi)}; font-size: 2em;">{predict_aqi}</span></h2>
                    <h3>空气质量等级：{aqi_grade}</h3>
                    <p style="font-size: 1.2em;">{health_info}</p>
                </div>
            """, unsafe_allow_html=True)

# ===================== 页面3：历史统计 =====================
elif page == "历史统计":
    st.title(" 历史AQI统计")
    with st.spinner("正在获取统计数据..."):
        stats_result = call_api("aqi_stats")
    if stats_result and stats_result["code"] == 200:
        data = stats_result["data"]
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("历史最低AQI", data["aqi_min"], delta_color="normal")
        with col2:
            st.metric("历史最高AQI", data["aqi_max"], delta_color="inverse")
        with col3:
            st.metric("历史平均AQI", data["aqi_mean"], delta_color="normal")
        with col4:
            st.metric("历史中位数AQI", data["aqi_median"], delta_color="normal")
        
        st.divider()
        # AQI等级分布饼图
        st.markdown("### AQI等级分布")
        fig, ax = plt.subplots()
        grades = list(data["aqi_grade_dist"].keys())
        counts = list(data["aqi_grade_dist"].values())
        # 匹配等级颜色
        colors = [
            get_aqi_color(50 if "Good" in g else 100 if "Moderate" in g else 150 if "Sensitive" in g 
                          else 200 if "Unhealthy" in g and "Very" not in g else 300 if "Very" in g else 500) 
            for g in grades
        ]
        ax.pie(counts, labels=grades, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
        
        # 等级数量表格
        st.markdown("### AQI等级数量明细")
        grade_df = pd.DataFrame(list(data["aqi_grade_dist"].items()), columns=["空气质量等级", "天数"])
        grade_df = grade_df.sort_values("天数", ascending=False)
        st.dataframe(grade_df, use_container_width=True)

# ===================== 页面4：健康建议 =====================
elif page == "健康建议":
    st.title(" AQI健康建议（EPA标准）")
    st.markdown("### 不同AQI等级的健康影响与行动建议")
    # 健康建议表格
    health_advice = pd.DataFrame({
        "AQI范围": ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500"],
        "空气质量等级": ["Good（优）", "Moderate（良）", "Unhealthy for Sensitive Groups（对敏感人群不健康）",
                       "Unhealthy（不健康）", "Very Unhealthy（非常不健康）", "Hazardous（危险）"],
        "颜色标识": ["<span style='color: #00e400; font-weight: bold;'>绿色</span>",
                  "<span style='color: #ffff00; font-weight: bold;'>黄色</span>",
                  "<span style='color: #ff7e00; font-weight: bold;'>橙色</span>",
                  "<span style='color: #ff0000; font-weight: bold;'>红色</span>",
                  "<span style='color: #8f3f97; font-weight: bold;'>紫色</span>",
                  "<span style='color: #7e0023; font-weight: bold;'>褐色</span>"],
        "健康影响": ["基本无空气污染", "极少数敏感人群有较弱影响", "敏感人群出现不适", "普通人群出现不适",
                  "所有人群可能出现健康问题", "所有人群面临严重健康风险"],
        "行动建议": ["各类人群可正常活动", "敏感人群减少户外活动", "敏感人群避免户外活动", "所有人群减少户外活动",
                  "所有人群避免户外活动，留在室内", "所有人群避免所有户外活动，紧急防护"]
    })
    st.markdown(health_advice.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 敏感人群定义")
    st.markdown("""
    1. 患有哮喘、慢性阻塞性肺疾病等呼吸系统疾病的人群；
    2. 心脏病患者；
    3. 老年人（65岁及以上）；
    4. 儿童（18岁及以下）；
    5. 孕妇。
    """)
    st.markdown("### 通用防护建议")
    st.markdown("""
    1. AQI超标时，关闭门窗，减少室内外空气交换；
    2. 使用空气净化器（优先选择带HEPA滤网的产品）；
    3. 外出时佩戴符合标准的口罩（如N95/KN95）；
    4. 减少剧烈户外运动，降低呼吸频率和深度。
    """)

# ===================== 页脚 =====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666666;">
    <p> 2024 AQI Prediction System | 基于NOAA+OpenAQ数据 | 机器学习模型预测</p>
    <p>适配美国EPA AQI标准 | 支持未来1天24小时尺度AQI预测</p>
</div>
""", unsafe_allow_html=True)
