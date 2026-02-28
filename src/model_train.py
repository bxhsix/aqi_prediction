import pandas as pd  
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置项 =====================
TRAIN_DATA_PATH = "../data/processed/aqi_train_data.csv"  # step2生成的含lag数据
MODEL_SAVE_PATH = "../model/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
TARGET = "AQI"  # 预测目标：当天AQI
# 绘图配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 6)

# ===================== 模型评估函数 =====================
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} 评估指标：")
    print(f"MAE（平均绝对误差）：{mae:.2f}")
    print(f"MSE（均方误差）：{mse:.2f}")
    print(f"RMSE（均方根误差）：{rmse:.2f}")
    print(f"R（决定系数）：{r2:.2f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def plot_prediction(y_true, y_pred, model_name, save_path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6, color="blue", label="预测值")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", label="完美预测线")
    plt.xlabel("真实AQI")
    plt.ylabel("预测AQI")
    plt.title(f"{model_name} - 真实AQI vs 预测AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{model_name}_prediction.png"))
    plt.close()

# ===================== 读取数据（直接使用lag特征）=====================
def load_and_split_data(path):
    # 1. 读取step2生成的含lag特征的数据
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"=== 数据校验 ===")
    print(f"总数据行数：{len(df)}")
    print(f"AQI统计信息：\n{df[TARGET].describe()}")

    # 2. 定义输入特征：仅使用所有lag1列（前一天NOAA+OpenAQ）
    feature_cols = [col for col in df.columns if col.endswith("_lag1")  and col != f"{TARGET}_lag1"]
    print(f"\n=== 预测特征（仅前一天lag1数据）===")
    print(f"特征数量：{len(feature_cols)}")
    print(f"特征列表：{feature_cols}")

    # 3. 定义X（前一天lag1特征）和y（当天AQI）
    X = df[feature_cols]
    y = df[TARGET]

    # 4. 时序划分（8:2，按时间顺序）
    train_size = int(0.8 * len(df))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    print(f"\n=== 数据集划分 ===")
    print(f"训练集：{len(X_train)} 条，测试集：{len(X_test)} 条")
    print(f"特征维度：{X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, X, y

# ===================== 模型训练 =====================
def train_basic_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    model_metrics = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n开始训练 {name} 模型...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, name)
        model_metrics[name] = metrics
        trained_models[name] = model
        plot_prediction(y_test, y_pred, name, MODEL_SAVE_PATH)

    print("\n=== 基础模型性能排名（R从高到低）===")
    for name, metrics in sorted(model_metrics.items(), key=lambda x: x[1]["r2"], reverse=True):
        print(f"{name}: R={metrics['r2']:.2f}, RMSE={metrics['rmse']:.2f}")
    return trained_models, model_metrics

def train_automl_model(X, y):
    print("\n开始AutoML模型训练（AutoGluon）...")
    train_data = X.copy()
    train_data[TARGET] = y

    predictor = TabularPredictor(
        label=TARGET,
        path=os.path.join(MODEL_SAVE_PATH, "autogluon_model"),
        eval_metric="r2",
        problem_type='regression'
    )

    predictor.fit(
        train_data=train_data,
        time_limit=600,
        presets="medium_quality_faster_train",
        excluded_model_types=['NN_TORCH', 'FASTAI', 'GBM', 'CAT'],
        verbosity=1
    )

    leaderboard = predictor.leaderboard(silent=True)
    rename_cols = {'val_r2': 'r2', 'val_rmse': 'rmse', 'model': 'model'}
    available_cols = [col for col in rename_cols.keys() if col in leaderboard.columns]
    leaderboard_simple = leaderboard[available_cols].rename(columns=rename_cols)

    print("\n=== AutoML模型排名 ===")
    print(leaderboard_simple.head())
    return predictor

# ===================== 模型选择与保存 =====================
def select_best_model(basic_models, automl_predictor, X_test, y_test):
    # 评估AutoML模型
    automl_pred = automl_predictor.predict(X_test)
    automl_metrics = evaluate_model(y_test, automl_pred, "AutoGluon_Best")
    plot_prediction(y_test, automl_pred, "AutoGluon_Best", MODEL_SAVE_PATH)

    # 收集所有模型指标
    all_metrics = {"AutoGluon_Best": automl_metrics}
    for name, model in basic_models.items():
        y_pred = model.predict(X_test)
        all_metrics[name] = evaluate_model(y_test, y_pred, name)

    # 选择最优模型
    best_model_name = max(all_metrics.items(), key=lambda x: x[1]["r2"])[0]
    print(f"\n=== 最优模型选择 ===")
    print(f"最优模型：{best_model_name}，R={all_metrics[best_model_name]['r2']:.2f}")

    # 保存最优模型
    if best_model_name == "AutoGluon_Best":
        automl_predictor.save()
        print(f"AutoGluon模型已保存至：{os.path.join(MODEL_SAVE_PATH, 'autogluon_model')}")
    else:
        joblib.dump(basic_models[best_model_name], os.path.join(MODEL_SAVE_PATH, "aqi_best_model.pkl"))
        print(f"最优模型已保存至：{os.path.join(MODEL_SAVE_PATH, 'aqi_best_model.pkl')}")
    return best_model_name

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, X, y = load_and_split_data(TRAIN_DATA_PATH)
        basic_models, model_metrics = train_basic_models(X_train, X_test, y_train, y_test)
        automl_predictor = train_automl_model(X, y)
        best_model = select_best_model(basic_models, automl_predictor, X_test, y_test)
        print("\n 模型训练全流程完成！")
    except Exception as e:
        print(f"\n 训练异常：{str(e)}")
        import traceback
        traceback.print_exc()
