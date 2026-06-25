"""
偏依赖图(PDP)分析模块。

展示单个特征(或特征对)对模型预测结果的边际效应。
使用 sklearn.inspection.PartialDependenceDisplay，零额外依赖。
"""

import warnings
from typing import Any

import pandas as pd
from matplotlib.figure import Figure
from sklearn.inspection import PartialDependenceDisplay


def plot_partial_dependence(
    model: Any,
    x_train: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 6,
    figsize: tuple = (12, 8),
) -> Figure | None:
    """
    绘制偏依赖图(PDP)，展示各特征如何影响模型预测。

    选择前 top_n 个特征绘制 PDP，帮助理解"当某个特征变化时，
    模型预测值如何变化"。

    Args:
        model: 训练好的模型
        x_train: 训练集特征(作为背景数据)
        feature_names: 全部特征名列表
        top_n: 要绘制的特征数(会取前 top_n 个，如果特征不够则全绘)
        figsize: 图片尺寸

    Returns:
        matplotlib Figure 或 None
    """
    n_features = len(feature_names)
    n_plot = min(top_n, n_features)

    if n_plot == 0:
        return None

    features = list(range(n_plot))

    try:
        display = PartialDependenceDisplay.from_estimator(
            model,
            x_train,
            features,
            feature_names=feature_names,
            grid_resolution=30,
            kind="average",
        )
        fig = display.figure_
        fig.set_size_inches(figsize)
    except Exception as exc:
        warnings.warn(f"PDP 绘制失败: {exc}")
        return None

    fig.suptitle("Partial Dependence Plots (PDP)", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_pdp_with_individual(
    model: Any,
    x_train: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 4,
    figsize: tuple = (12, 8),
) -> Figure | None:
    """
    绘制 PDP + 个体条件期望(ICE)曲线。

    ICE 曲线展示每个样本的预测如何随特征变化，比单独 PDP 更细致。

    Args:
        model: 训练好的模型
        x_train: 训练集特征
        feature_names: 全部特征名列表
        top_n: 要绘制的特征数
        figsize: 图片尺寸

    Returns:
        matplotlib Figure 或 None
    """
    n_features = len(feature_names)
    n_plot = min(top_n, n_features)

    if n_plot == 0:
        return None

    features = list(range(n_plot))

    try:
        display = PartialDependenceDisplay.from_estimator(
            model,
            x_train,
            features,
            feature_names=feature_names,
            grid_resolution=30,
            kind="both",  # PDP + ICE 曲线
            subsample=50,  # ICE 抽 50 条样本，避免太密
        )
        fig = display.figure_
        fig.set_size_inches(figsize)
    except Exception as exc:
        warnings.warn(f"PDP+ICE 绘制失败: {exc}")
        return None

    fig.suptitle("PDP + ICE Curves", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
