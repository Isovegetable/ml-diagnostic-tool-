"""
SHAP 可解释性分析模块。

提供 SHAP 值计算、摘要图(beeswarm)和柱状图(特征重要性)。
需要安装 shap 库: pip install shap
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# shap 是可选依赖，模块级导入由各函数内部延迟加载


def _resolve_explainer(model: Any, x_train: pd.DataFrame, x_test_sample: pd.DataFrame):
    """
    根据模型类型自动选择最合适的 SHAP Explainer。

    Args:
        model: 训练好的模型
        x_train: 训练集(作为 LinearExplainer 的背景)
        x_test_sample: 测试集样本(用于计算 SHAP 值)

    Returns:
        (explainer, shap_values, expected_value) 或抛出异常
    """
    import shap

    model_type = type(model).__name__

    # --- TreeExplainer: 树模型/集成模型 ---
    tree_like_types = (
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor", "XGBModel",
        "LGBMRegressor", "LGBMModel",
        "CatBoostRegressor",
        "ExtraTreesRegressor",
    )
    if model_type in tree_like_types or hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test_sample)
        expected_value = explainer.expected_value
        # TreeExplainer 可能返回列表(多输出)，取第一个
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(expected_value, (list, np.ndarray)) and expected_value.ndim == 0:
            expected_value = float(expected_value)
        elif isinstance(expected_value, (list, np.ndarray)) and expected_value.size > 1:
            expected_value = float(expected_value[0])
        return explainer, shap_values, expected_value

    # --- LinearExplainer: 线性模型 ---
    linear_types = ("LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDRegressor")
    if model_type in linear_types:
        explainer = shap.LinearExplainer(model, x_train)
        shap_values = explainer.shap_values(x_test_sample)
        expected_value = explainer.expected_value
        return explainer, shap_values, expected_value

    # --- KernelExplainer(回退): 通用模型 ---
    x_train_bg = x_train.sample(min(100, len(x_train)), random_state=42)
    explainer = shap.KernelExplainer(model.predict, x_train_bg)
    shap_values = explainer.shap_values(x_test_sample)
    expected_value = explainer.expected_value
    return explainer, shap_values, expected_value


def compute_shap_values(
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    n_samples: int = 500,
) -> tuple:
    """
    计算 SHAP 值，自动选择适合的 Explainer。

    Args:
        model: 训练好的模型
        x_train: 训练集特征(作为背景数据)
        x_test: 测试集特征(用于计算 SHAP)
        n_samples: SHAP 计算的样本上限，默认 500(测试集不足时用全部)

    Returns:
        (shap_values, expected_value, x_test_sample) 或 (None, None, None)
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        warnings.warn("shap 未安装，请执行 pip install shap")
        return None, None, None

    # 样本量过大时抽样
    if len(x_test) > n_samples:
        x_test_sample = x_test.sample(n_samples, random_state=42)
    else:
        x_test_sample = x_test

    try:
        _, shap_values, expected_value = _resolve_explainer(model, x_train, x_test_sample)
    except Exception as exc:
        warnings.warn(f"SHAP 值计算失败: {exc}")
        return None, None, None

    return shap_values, expected_value, x_test_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    x_test_sample: pd.DataFrame,
    feature_names: list[str],
    max_display: int = 15,
) -> plt.Figure | None:
    """
    绘制 SHAP 摘要图(beeswarm)。

    展示每个特征对模型输出的影响方向和大小。
    红色=特征值高，蓝色=特征值低，横轴=SHAP 值(对预测的影响)。

    Args:
        shap_values: SHAP 值矩阵
        x_test_sample: 对应的测试集样本
        feature_names: 特征名列表
        max_display: 显示的特征数量上限

    Returns:
        matplotlib Figure 或 None
    """
    if shap_values is None:
        return None

    import shap

    # shap.summary_plot 使用当前 figure, 先创建好
    fig = plt.figure(figsize=(10, max(3, min(max_display * 0.35, 7))))

    shap.summary_plot(
        shap_values,
        x_test_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()
    return fig


def plot_shap_bar(
    shap_values: np.ndarray,
    x_test_sample: pd.DataFrame,
    feature_names: list[str],
    max_display: int = 15,
) -> plt.Figure | None:
    """
    绘制 SHAP 柱状图(特征重要性)。

    按 mean(|SHAP|) 从大到小排序，展示各特征的全局重要性。

    Args:
        shap_values: SHAP 值矩阵
        x_test_sample: 对应的测试集样本(仅用于 shap.summary_plot 接口)
        feature_names: 特征名列表
        max_display: 显示的特征数量上限

    Returns:
        matplotlib Figure 或 None
    """
    if shap_values is None:
        return None

    import shap

    fig = plt.figure(figsize=(10, max(3, min(max_display * 0.35, 7))))

    shap.summary_plot(
        shap_values,
        x_test_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()
    return fig
