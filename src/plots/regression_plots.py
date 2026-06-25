import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predicted_vs_actual(
    y_true_test,
    y_pred_test,
    y_true_train,
    y_pred_train,
    title: str = "Actual vs Predicted",
    r2_test: float | None = None,
    rmse_test: float | None = None,
    r2_train: float | None = None,
    rmse_train: float | None = None,
    is_best: bool = False,
):
    """
    绘制真实值 vs 预测值散点图(训练集+测试集双色,标注双指标)。

    Args:
        y_true_test, y_pred_test: 测试集
        y_true_train, y_pred_train: 训练集
        title: 图表标题
        r2_test, rmse_test: 测试集指标
        r2_train, rmse_train: 训练集指标
        is_best: 是否为最佳模型(标题加 ★)

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # 确定全局范围(让对角线贯穿整个图)
    all_true = np.concatenate([y_true_train, y_true_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    min_v = min(all_true.min(), all_pred.min())
    max_v = max(all_true.max(), all_pred.max())

    # 绘制理想预测线
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="red", linewidth=1.5, zorder=0)

    # 训练集(橙,小点)
    ax.scatter(
        y_true_train, y_pred_train,
        alpha=0.5, s=20, c="#ff7f0e", edgecolors="none", label="Train",
        zorder=1,
    )
    # 测试集(深蓝,大点)
    ax.scatter(
        y_true_test, y_pred_test,
        alpha=0.8, s=40, c="#1f77b4", edgecolors="white", linewidth=0.5, label="Test",
        zorder=2,
    )

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)

    # 标题(最佳模型加星)
    star = " ★" if is_best else ""
    ax.set_title(f"{title}{star}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(loc="upper left")

    # 标注指标(图右下角,分为训练/测试两栏)
    lines = []
    if r2_train is not None or rmse_train is not None:
        parts = []
        if r2_train is not None:
            parts.append(f"R²={r2_train:.4f}")
        if rmse_train is not None:
            parts.append(f"RMSE={rmse_train:.4f}")
        lines.append("Train  " + " | ".join(parts))
    if r2_test is not None or rmse_test is not None:
        parts = []
        if r2_test is not None:
            parts.append(f"R²={r2_test:.4f}")
        if rmse_test is not None:
            parts.append(f"RMSE={rmse_test:.4f}")
        lines.append("Test   " + " | ".join(parts))

    if lines:
        ax.text(
            0.95, 0.05, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    fig.tight_layout()
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame):
    """
    绘制模型指标对比柱状图。

    Args:
        metrics_df: 包含模型名称和 test_r2 的 DataFrame

    Returns:
        matplotlib.figure.Figure: 图表对象
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics_df["model"], metrics_df["test_r2"])
    ax.set_ylabel("Test R²")
    ax.set_title("Model Comparison")
    ax.tick_params(axis="x", rotation=20)

    return fig


def plot_feature_importance(model, feature_names: list[str]):
    """
    绘制特征重要性图。

    仅支持具有 feature_importances_ 属性的模型（如树模型）。

    Args:
        model: 训练后的模型对象
        feature_names: 特征名称列表

    Returns:
        matplotlib.figure.Figure or None: 图表对象，如果模型不支持则返回 None
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, len(feature_names) * 0.25)))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")

    return fig
