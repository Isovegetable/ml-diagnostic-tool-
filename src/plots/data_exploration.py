import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def test_normality(data: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    对每列做 Shapiro-Wilk 正态性检验。

    对每列返回 statistic、p_value、is_normal (p > alpha 视为正态)、
    sample_size。Shapiro-Wilk 在 n > 5000 时不准确,此时随机抽样 5000 个。

    Args:
        data: 数值型数据(每列独立检验)
        alpha: 显著性水平,默认 0.05

    Returns:
        dict: {列名: {"statistic", "p_value", "is_normal", "sample_size"}}
              is_normal 为 None 表示样本量不足无法判断
    """
    results = {}
    for col in data.columns:
        col_data = data[col].dropna()
        n = len(col_data)

        if n < 3:
            # 样本量太少,Shapiro-Wilk 无法计算
            results[col] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "is_normal": None,
                "sample_size": n,
            }
            continue

        # Shapiro-Wilk 在 n > 5000 时不准确,做随机抽样
        if n > 5000:
            col_data = col_data.sample(n=5000, random_state=42)

        statistic, p_value = stats.shapiro(col_data)
        results[col] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": bool(p_value > alpha),
            "sample_size": n,
        }
    return results


def decide_correlation_method(data: pd.DataFrame, alpha: float = 0.05) -> tuple[str, dict]:
    """
    决定使用哪种相关性方法(严格标准)。

    严格标准: 全部数值列通过正态性检验(p > alpha)才用 Pearson,
    否则用 Spearman(对非正态数据更稳健)。

    Args:
        data: 数值型数据
        alpha: 显著性水平,默认 0.05

    Returns:
        (method, test_results): method 为 "pearson" 或 "spearman"
    """
    test_results = test_normality(data, alpha=alpha)

    # 排除无法判断的列(is_normal is None),只看有结论的列
    decidable = [r for r in test_results.values() if r["is_normal"] is not None]
    if not decidable:
        # 全部列都无法判断,保守起见用 Spearman
        return "spearman", test_results

    all_normal = all(r["is_normal"] for r in decidable)
    method = "pearson" if all_normal else "spearman"
    return method, test_results


def plot_target_distribution(y: pd.Series, target_column: str):
    """
    绘制目标列分布图。

    Args:
        y: 目标列数据
        target_column: 目标列名称

    Returns:
        matplotlib.figure.Figure: 图表对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 直方图
    ax1.hist(y, bins=30, edgecolor="black", alpha=0.7)
    ax1.set_xlabel(target_column)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"{target_column} Distribution (Histogram)")
    ax1.grid(True, alpha=0.3)

    # 箱线图
    ax2.boxplot(y, vert=True)
    ax2.set_ylabel(target_column)
    ax2.set_title(f"{target_column} Distribution (Box Plot)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame, target_column: str, method: str = "pearson"
):
    """
    绘制特征相关性热力图。

    Args:
        data: 包含所有特征和目标列的数据
        target_column: 目标列名称
        method: 相关性方法, "pearson" 或 "spearman"

    Returns:
        matplotlib.figure.Figure: 图表对象
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method 必须是 'pearson' 或 'spearman',当前为: {method}")

    # 计算相关性矩阵
    corr_matrix = data.corr(method=method)

    # 获取与目标列相关性最高的前10个特征
    target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
    top_features = target_corr.head(11).index.tolist()  # 包含目标列自己

    # 只显示top特征的相关性
    corr_subset = corr_matrix.loc[top_features, top_features]

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_subset,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(
        f"Feature Correlation Heatmap ({method.capitalize()}, Top 10 vs {target_column})"
    )
    plt.tight_layout()
    return fig


def plot_missing_values(data: pd.DataFrame):
    """
    绘制缺失值可视化图。

    Args:
        data: 原始数据

    Returns:
        matplotlib.figure.Figure: 图表对象，如果没有缺失值则返回None
    """
    # 计算缺失值比例
    missing_rate = data.isna().mean().sort_values(ascending=False)
    missing_rate = missing_rate[missing_rate > 0]

    if len(missing_rate) == 0:
        return None  # 没有缺失值

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_rate) * 0.3)))
    ax.barh(missing_rate.index, missing_rate.values * 100)
    ax.set_xlabel("Missing Rate (%)")
    ax.set_title("Missing Values by Column")
    ax.grid(True, alpha=0.3, axis="x")

    # 添加数值标签
    for i, v in enumerate(missing_rate.values):
        ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va="center")

    plt.tight_layout()
    return fig


def plot_feature_distributions(data: pd.DataFrame, feature_columns: list, max_plots: int = 6):
    """
    绘制特征分布图（网格布局）。

    Args:
        data: 数据
        feature_columns: 特征列名列表
        max_plots: 最多显示几个特征

    Returns:
        matplotlib.figure.Figure: 图表对象
    """
    # 最多显示max_plots个特征
    features_to_plot = feature_columns[:max_plots]
    n_features = len(features_to_plot)

    # 计算网格布局
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        ax.hist(data[feature].dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{feature} Distribution")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig
