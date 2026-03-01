import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== 配置项 =====================
MODEL_CONFIG_PATH = "../model/model_config.json"  # 替换原硬编码的MODEL_PATH
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

# ===================== 加载模型+特征校验配置 =====================
# 全局变量：保存模型、训练特征列表、模型类型
model = None
train_feature_cols = None
best_model_type = None

def load_model_and_validate_features():
    """加载模型并校验特征配置"""
    global model, train_feature_cols, best_model_type
    
    # 1. 读取模型配置
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError(f"模型配置文件不存在：{MODEL_CONFIG_PATH}\n请先执行模型训练脚本")
    
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = json.load(f)
    
    # 2. 提取配置信息
    best_model_type = model_config["model_type"]
    best_model_path = model_config["best_model_path"]
    train_feature_cols = model_config["feature_cols"]
    print(f" 加载最优模型配置：\n  - 模型类型：{best_model_type}\n  - 模型路径：{best_model_path}\n  - 训练特征列表：{train_feature_cols}")
    
    # 3. 加载模型
    if best_model_type == "autogluon":
        try:
            from autogluon.tabular import TabularPredictor
            model = TabularPredictor.load(best_model_path)
            # 额外校验AutoGluon模型的特征（双重保障）
            autogluon_features = model.feature_metadata_in.get_features()
            if set(autogluon_features) != set(train_feature_cols):
                raise ValueError(
                    f"AutoGluon模型内置特征与配置文件特征不匹配！\n"
                    f"  配置文件特征：{train_feature_cols}\n  AutoGluon模型特征：{autogluon_features}"
                )
        except ImportError as e:
            raise ImportError(f"缺少AutoGluon依赖：{str(e)}\n请执行：pip install autogluon")
    else:  # sklearn/xgboost模型
        try:
            model = joblib.load(best_model_path)
        except ImportError as e:
            raise ImportError(f"缺少Sklearn/XGBoost依赖：{str(e)}\n请执行：pip install scikit-learn xgboost")
    
    # 4. 校验训练特征数量（必须是15个）
    if len(train_feature_cols) != 15:
        raise ValueError(f"训练特征数量错误！预期15个，实际{len(train_feature_cols)}个：{train_feature_cols}")
    
    print(f" 模型加载成功！训练特征数量：{len(train_feature_cols)}")

# 启动时加载模型并校验
try:
    load_model_and_validate_features()
except Exception as e:
    raise HTTPException(status_code=500, detail=f"模型初始化失败：{str(e)}")

# ===================== 辅助函数 =====================
def get_aqi_grade(aqi_value: float) -> tuple:
    """根据AQI值获取对应的等级和健康信息"""
    aqi_value = round(aqi_value)
    for min_val, max_val, grade, health_info in AQI_GRADES:
        if min_val <= aqi_value <= max_val:
            return grade, health_info
    return "Hazardous（危险）", "所有人群面临严重健康风险，应避免所有户外活动，紧急防护"

def validate_request_features(request_data: pd.DataFrame):
    """校验请求数据的特征顺序/名称与训练时一致"""
    # 1. 获取请求数据的特征列
    request_cols = list(request_data.columns)
    
    # 2. 校验列名数量
    if len(request_cols) != len(train_feature_cols):
        raise ValueError(
            f"请求特征数量错误！预期{len(train_feature_cols)}个，实际{len(request_cols)}个\n"
            f"  预期特征：{train_feature_cols}\n  实际特征：{request_cols}"
        )
    
    # 3. 校验列名顺序（核心！顺序不一致直接报错）
    if request_cols != train_feature_cols:
        raise ValueError(
            f"请求特征顺序/名称与训练时不匹配！\n"
            f"  训练特征顺序：{train_feature_cols}\n  请求特征顺序：{request_cols}"
        )
    print(" 特征校验通过：请求特征与训练特征完全一致")

# ===================== API接口 =====================
@app.get("/health", summary="健康检查接口")
async def health_check():
    return {"code": 200, "status": "healthy", "message": "API服务正常运行"}

@app.post("/predict_aqi", summary="AQI预测接口")
async def predict_aqi(request: AQIPredictionRequest):
    try:
        # 1. 将请求数据转换为DataFrame（先按请求模型的顺序）
        input_data = pd.DataFrame([request.dict()])
        print(f" 接收预测请求：\n{input_data}")
        
        # 2. 核心：校验特征顺序/名称（失败则直接抛异常）
        validate_request_features(input_data)
        
        # 3. 按训练特征顺序重新排序（双重保障，即使请求顺序错了也能修正，可选）
        input_data = input_data[train_feature_cols]
        
        # 4. 使用模型进行预测
        if best_model_type == "autogluon":
            prediction = model.predict(input_data)
            predict_aqi = float(prediction.iloc[0])
        else:  # sklearn/xgboost模型
            # 可选：转换为数组（sklearn兼容DF，但数组更稳妥）
            input_array = input_data.values
            prediction = model.predict(input_array)
            predict_aqi = float(prediction[0])
        
        # 5. 获取AQI等级和健康建议
        aqi_grade, health_info = get_aqi_grade(predict_aqi)
        
        # 6. 构造响应（匹配前端预期格式）
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "predict_aqi": round(predict_aqi),
                "aqi_grade": aqi_grade,
                "health_info": health_info
            }
        }
    except ValueError as e:
        # 特征校验失败的友好提示
        print(f" 特征校验失败：{str(e)}")
        raise HTTPException(status_code=400, detail=f"请求参数错误：{str(e)}")
    except Exception as e:
        # 其他预测错误
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
