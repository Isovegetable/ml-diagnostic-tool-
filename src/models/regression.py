from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from .hyperparameter_optimization import (
    get_base_models,
    get_search_space,
    run_hpo,
)


@dataclass
class ModelResult:
    """单个模型的评估结果"""

    model_name: str
    model: object  # 训练后的模型对象
    train_r2: float
    test_r2: float
    mae: float
    rmse: float
    train_rmse: float  # 训练集 RMSE
    cv_r2_mean: float  # 交叉验证 R² 均值
    cv_r2_std: float  # 交叉验证 R² 标准差
    cv_mae_mean: float  # 交叉验证 MAE 均值
    cv_rmse_mean: float  # 交叉验证 RMSE 均值
    y_train: pd.Series  # 训练集真实值
    y_test: pd.Series  # 测试集真实值
    train_pred: object  # 训练集预测值
    test_pred: object  # 测试集预测值
    best_params: dict  # HPO 选出的最佳超参(无 HPO 时为默认超参)
    x_train: pd.DataFrame = None  # 训练集特征(用于 SHAP/PDP)
    x_test: pd.DataFrame = None  # 测试集特征(用于 SHAP/PDP)


def get_regression_models() -> dict:
    """
    返回所有可用回归模型字典(与 hyperparameter_optimization 共享懒加载)。

    Returns:
        dict: 模型名称到模型对象的映射(含默认初始超参)
    """
    return get_base_models()


def train_and_evaluate_models(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    cv_method: str = "kfold",
    cv_folds: int = 5,
    random_state: int = 42,
    hpo_method: str | None = None,
    n_trials: int = 30,
    selected_models: list[str] | None = None,
    progress_callback: Callable | None = None,
) -> tuple[dict[str, "ModelResult"], pd.DataFrame, dict[str, dict]]:
    """
    训练并评估所有回归模型（支持 HPO）。

    Args:
        x: 特征数据
        y: 目标数据
        test_size: 测试集比例
        cv_method: 交叉验证方法 ("kfold" 或 "loocv")
        cv_folds: K折交叉验证的折数
        random_state: 随机种子
        hpo_method: 超参数优化方法, None 表示不优化(用默认超参);
                    可选 "bayesian" / "random" / "hyperband" / "grid"
        n_trials: HPO 试验次数(grid 会忽略)
        selected_models: 要训练的模型名列表, None 表示全部可用模型
        progress_callback: 进度回调(接受 msg 和 state 参数),用于前端实时显示进度

    Returns:
        tuple: (
            模型结果字典 {model_name: ModelResult},
            指标对比 DataFrame(按 CV R² 均值降序),
            HPO 结果字典 {model_name: hpo_result_dict}(无 HPO 时为空 dict)
        )
    """
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # 确定交叉验证的折数
    if cv_method == "loocv":
        cv = len(x_train)  # 留一法：折数=训练集样本数
    else:
        cv = cv_folds  # K折交叉验证

    # ========== 阶段 1: HPO(在训练集上做内部 CV 搜索) ==========
    hpo_results: dict[str, dict] = {}
    if hpo_method is not None:
        if progress_callback:
            progress_callback(f"🔍 正在 HPO 优化: {hpo_method.upper()}", state="running")
        base_models = get_base_models()
        for model_name in base_models.keys():
            if selected_models is not None and model_name not in selected_models:
                continue
            if progress_callback:
                progress_callback(f"🔍 HPO {hpo_method.upper()} 进行中: `{model_name}` ({n_trials} 轮)...", state="running")
            hpo_result = run_hpo(
                method=hpo_method,
                model_name=model_name,
                x=x_train,
                y=y_train,
                n_trials=n_trials,
                cv=cv,
                random_state=random_state,
            )
            hpo_results[model_name] = hpo_result
            if progress_callback:
                progress_callback(f"✅ HPO {hpo_method.upper()} 完成: `{model_name}` 最佳 R² = {hpo_result['best_score']:.4f}", state="running")

    # ========== 阶段 2: 训练和评估每个模型 ==========
    results: dict[str, ModelResult] = {}
    rows: list[dict] = []

    base_models = get_base_models()
    for model_name, default_model in base_models.items():
        if selected_models is not None and model_name not in selected_models:
            continue
        if progress_callback:
            phase = "HPO + 训练" if hpo_method else "训练"
            progress_callback(f"🚀 {phase}最终模型: `{model_name}`", state="running")
        # 选模型实例:HPO 结果优先,否则用默认
        if model_name in hpo_results:
            best_params = hpo_results[model_name]["best_params"]
            from .hyperparameter_optimization import _instantiate_model

            model = _instantiate_model(model_name, best_params, random_state)
        else:
            best_params = {}
            model = default_model

        # ========== 交叉验证评估(在训练集上,统一计算方式) ==========
        cv_r2_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="r2")
        cv_r2_mean = float(cv_r2_scores.mean())
        cv_r2_std = float(cv_r2_scores.std())

        # ========== 最终模型训练（用全部训练集） ==========
        model.fit(x_train, y_train)

        # 预测
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        # 计算指标
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        mae = mean_absolute_error(y_test, test_pred)
        rmse = root_mean_squared_error(y_test, test_pred)
        train_rmse = root_mean_squared_error(y_train, train_pred)

        # 交叉验证的 MAE / RMSE(无 HPO 时算,有 HPO 时复用最优 trial 的 CV R² 即可, MAE/RMSE 简化为 0 占位)
        if model_name in hpo_results:
            # HPO 内部只用了 R² 作为评分,这里为了不重复算,留 NaN
            cv_mae_mean = float("nan")
            cv_rmse_mean = float("nan")
        else:
            cv_mae_scores = -cross_val_score(
                model, x_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
            )
            cv_rmse_scores = -cross_val_score(
                model, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
            )
            cv_mae_mean = float(cv_mae_scores.mean())
            cv_rmse_mean = float(cv_rmse_scores.mean())

        if progress_callback:
            progress_callback(
                f"✅ 模型完成: `{model_name}` | "
                f"Test R² = {test_r2:.3f}, CV R² = {cv_r2_mean:.3f}",
                state="running",
            )

        # 保存结果
        result = ModelResult(
            model_name=model_name,
            model=model,
            train_r2=float(train_r2),
            test_r2=float(test_r2),
            mae=float(mae),
            rmse=float(rmse),
            train_rmse=float(train_rmse),
            cv_r2_mean=cv_r2_mean,
            cv_r2_std=cv_r2_std,
            cv_mae_mean=cv_mae_mean,
            cv_rmse_mean=cv_rmse_mean,
            y_train=y_train,
            y_test=y_test,
            train_pred=train_pred,
            test_pred=test_pred,
            best_params=best_params,
            x_train=x_train,
            x_test=x_test,
        )
        results[model_name] = result

        rows.append(
            {
                "model": model_name,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "cv_r2_mean": cv_r2_mean,
                "cv_r2_std": cv_r2_std,
                "mae": mae,
                "rmse": rmse,
            }
        )

    # 创建指标对比表（优先按 CV R² 均值排序）
    metrics_df = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False)

    return results, metrics_df, hpo_results
