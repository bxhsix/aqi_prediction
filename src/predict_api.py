import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== 配置项 =====================
# AutoGluon模型路径（你的实际路径）
MODEL_PATH = "../model/autogluon_model/"
API_HOST = "0.0.0.0"
API_PORT = 8000

# ===================== AQI等级配置 =====================
AQI_GRADES = [
    (0, 50, "Good（优）", "基本无空气污染，各类人群可正常活动"),
    (51, 100, "Moderate（良）", "极少数敏感人群有较弱影响，敏感人群可减少户外活动"),
    (101, 150, "Unhealthy for Sensitive Groups（对敏感人群不健康）", "敏感人群出现不适，应避免户外活动"),
    (151, 200, "Unhealthy（不健康）", "普通人群出现不适，所有人群应减少户外活动"),
    (201, 300, "Very Unhealthy（非常不健康）", "所有人群可能出现健康问题，应避免户外活动，留在室内"),
    (301, 500, "Hazardous（危险）", "所有人群面临严重健康风险，应避免所有户外活动，紧急防护")
]

# ===================== 初始化FastAPI应用 =====================
app = FastAPI(title="AQI Prediction API", version="1.0")

# ===================== 数据模型定义（严格匹配前端15个lag1特征）=====================
class AQIPredictionRequest(BaseModel):
    """请求模型：严格匹配前端传递的15个lag1特征"""
    pm25_24h_lag1: float   # 前1日PM2.5 24h均值滞后值
    pm10_24h_lag1: float   # 前1日PM10 24h均值滞后值
    o3_24h_lag1: float     # 前1日O3 24h均值滞后值
    co_24h_lag1: float     # 前1日CO 24h均值滞后值
    no2_24h_lag1: float    # 前1日NO2 24h均值滞后值
    so2_24h_lag1: float    # 前1日SO2 24h均值滞后值
    TEMP_lag1: float       # 前1日平均气温滞后值
    DEWP_lag1: float       # 前1日露点温度滞后值
    SLP_lag1: float        # 前1日海平面气压滞后值
    WDSP_lag1: float       # 前1日平均风速滞后值
    MAX_lag1: float        # 前1日最高温滞后值
    MIN_lag1: float        # 前1日最低温滞后值
    PRCP_lag1: float       # 前1日降雨量滞后值
    TEMP_DIFF_lag1: float  # 前1日气温日较差滞后值
    PRCP_BINARY_lag1: int  # 前1日是否降雨滞后值（0/1）

# ===================== 加载AutoGluon模型 =====================
try:
    from autogluon.tabular import TabularPredictor
    # 加载AutoGluon模型
    model = TabularPredictor.load(MODEL_PATH)
    print(f" 模型加载成功：{MODEL_PATH}")
    
    # 验证模型特征（可选，调试用）
    print(f" 模型训练时的特征列表：{model.feature_metadata_in.get_features()}")
except ImportError as e:
    raise HTTPException(status_code=500, detail=f"缺少AutoGluon依赖：{str(e)}\n请执行：pip install autogluon")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"模型加载失败：{str(e)}")

# ===================== 辅助函数 =====================
def get_aqi_grade(aqi_value: float) -> tuple:
    """根据AQI值获取对应的等级和健康信息"""
    aqi_value = round(aqi_value)
    for min_val, max_val, grade, health_info in AQI_GRADES:
        if min_val <= aqi_value <= max_val:
            return grade, health_info
    return "Hazardous（危险）", "所有人群面临严重健康风险，应避免所有户外活动，紧急防护"

# ===================== API接口 =====================
@app.get("/health", summary="健康检查接口")
async def health_check():
    return {"code": 200, "status": "healthy", "message": "API服务正常运行"}

@app.post("/predict_aqi", summary="AQI预测接口")
async def predict_aqi(request: AQIPredictionRequest):
    try:
        # 1. 将请求数据转换为DataFrame（严格匹配模型特征顺序）
        input_data = pd.DataFrame([request.dict()])
        
        # 2. 调试：打印输入数据（可选）
        print(f" 接收预测请求：\n{input_data}")
        
        # 3. 使用模型进行预测
        prediction = model.predict(input_data)
        predict_aqi = float(prediction.iloc[0])
        
        # 4. 获取AQI等级和健康建议
        aqi_grade, health_info = get_aqi_grade(predict_aqi)
        
        # 5. 构造响应（匹配前端预期格式）
        return {
            "code": 200,
            "msg": "success",  # 前端预期的是msg，不是message
            "data": {
                "predict_aqi": round(predict_aqi),  # 整数返回，符合AQI标准
                "aqi_grade": aqi_grade,
                "health_info": health_info
            }
        }
    except Exception as e:
        # 详细错误日志，方便调试
        print(f" 预测失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")

@app.get("/aqi_stats", summary="获取AQI统计信息")
async def get_aqi_stats():
    """返回历史AQI统计数据（从训练数据读取）"""
    try:
        # 读取训练数据获取真实统计
        train_df = pd.read_csv("../data/processed/aqi_train_data.csv")
        aqi_series = train_df["AQI"]
        
        # 计算等级分布
        aqi_grade_dist = {}
        for aqi in aqi_series:
            grade, _ = get_aqi_grade(aqi)
            aqi_grade_dist[grade] = aqi_grade_dist.get(grade, 0) + 1
        
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "aqi_min": round(aqi_series.min()),
                "aqi_max": round(aqi_series.max()),
                "aqi_mean": round(aqi_series.mean(), 2),
                "aqi_median": round(aqi_series.median()),
                "aqi_grade_dist": aqi_grade_dist
            }
        }
    except Exception as e:
        print(f" 获取统计信息失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败：{str(e)}")

# ===================== 启动服务 =====================
if __name__ == "__main__":
    # 自动安装依赖（可选）
    try:
        import autogluon
    except ImportError:
        os.system("pip install autogluon")
    
    print(f" 启动AQI预测API服务：http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
