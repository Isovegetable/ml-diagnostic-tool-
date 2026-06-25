"""
HPO 可视化模块。

提供 3 类图表:
1. plot_hpo_convergence  - 单次 HPO 的收敛曲线(best R² + best RMSE 双轴)
2. plot_multi_hpo_convergence  - 多 HPO 方法收敛曲线叠加对比
3. plot_hpo_comparison  - 不同 HPO 方法在不同模型上的最终最佳 CV R² 对比
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _format_history_to_series(
    hpo_result: dict,
) -> tuple[list[int], list[float], list[float]]:
    """从 HPO 结果中提取 (trial_number(从1开始), best_r2_so_far, best_rmse_so_far) 序列。"""
    history = hpo_result["history"]
    if not history:
        return [], [], []
    sorted_history = sorted(history, key=lambda h: h["trial"])
    trials = [h["trial"] + 1 for h in sorted_history]  # +1: 从1开始
    r2s = [h["score"] for h in sorted_history]
    rmses = [h.get("rmse", 0) for h in sorted_history]
    best_r2_so_far = list(np.maximum.accumulate(r2s))
    # RMSE 越小越好 → 取最小值累积
    best_rmse_so_far = list(np.minimum.accumulate(rmses))
    return trials, best_r2_so_far, best_rmse_so_far


def plot_hpo_convergence(hpo_result: dict, title: str | None = None) -> plt.Figure:
    """
    绘制单次 HPO 的收敛曲线(双轴: 左 R² + 右 RMSE)。

    Args:
        hpo_result: HPO 模块返回的 dict
        title: 自定义标题(默认用 method + model_name)

    Returns:
        matplotlib.figure.Figure
    """
    trials, best_r2, best_rmse = _format_history_to_series(hpo_result)
    if not trials:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No trials recorded", ha="center", va="center")
        return fig

    method = hpo_result.get("method", "unknown")
    model_name = hpo_result.get("model_name", "")
    default_title = f"HPO Convergence: {method.upper()} on {model_name}"

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    # 左轴: R²(蓝)
    line1 = ax1.plot(trials, best_r2, marker="o", markersize=3, linewidth=1.5, color="#1f77b4", label="R²")
    ax1.fill_between(trials, best_r2, alpha=0.15, color="#1f77b4")
    ax1.set_ylabel("Best CV R² (so far)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    # 右轴: RMSE(红)
    line2 = ax2.plot(trials, best_rmse, marker="o", markersize=3, linewidth=1.5, color="#d62728", label="RMSE")
    ax2.set_ylabel("Best CV RMSE (so far)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_xlabel("Trial Number")
    ax1.set_title(title or default_title)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")

    fig.tight_layout()
    return fig


def plot_multi_hpo_convergence(
    hpo_results_dict: dict[str, dict],
    model_name: str,
) -> plt.Figure:
    """
    把多个 HPO 方法的收敛曲线画在同一张图上(每个方法一条线)。

    Args:
        hpo_results_dict: {method_name: hpo_result_dict}
        model_name: 用于标题的模型名

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = {
        "bayesian": "#d62728",
        "random": "#1f77b4",
        "hyperband": "#2ca02c",
        "grid": "#ff7f0e",
    }

    for method, hpo_result in hpo_results_dict.items():
        trials, best_so_far, _ = _format_history_to_series(hpo_result)
        if not trials:
            continue
        c = colors.get(method, None)
        ax.plot(
            trials,
            best_so_far,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=method.upper(),
            color=c,
        )

    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Best CV R² (so far)")
    ax.set_title(f"HPO Convergence Comparison: {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_hpo_comparison(
    comparison_data: dict[str, dict[str, float]],
) -> plt.Figure:
    """
    不同 HPO 方法在不同模型上的最终最佳 CV R² 对比柱状图。

    Args:
        comparison_data: {model_name: {method_name: best_score}}
                         例如: {"Random Forest": {"bayesian": 0.85, "random": 0.83, ...}, ...}

    Returns:
        matplotlib.figure.Figure
    """
    if not comparison_data:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No HPO results", ha="center", va="center")
        return fig

    models = list(comparison_data.keys())
    # 收集所有方法名(保持出现顺序)
    methods_seen: list[str] = []
    for method_dict in comparison_data.values():
        for m in method_dict.keys():
            if m not in methods_seen:
                methods_seen.append(m)

    colors = {
        "bayesian": "#d62728",
        "random": "#1f77b4",
        "hyperband": "#2ca02c",
        "grid": "#ff7f0e",
    }

    n_models = len(models)
    n_methods = len(methods_seen)
    bar_width = 0.8 / max(n_methods, 1)
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(7, n_models * 2.5), 4.5))

    for i, method in enumerate(methods_seen):
        scores = [comparison_data[m].get(method, 0.0) for m in models]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            scores,
            bar_width,
            label=method.upper(),
            color=colors.get(method, None),
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Best CV R²")
    ax.set_title("HPO Method Comparison (Best CV R²)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    # 留出顶部空间显示数值
    ymax = max(
        (s for d in comparison_data.values() for s in d.values()),
        default=0.0,
    )
    ax.set_ylim(top=ymax * 1.1 + 0.02)

    return fig


def format_best_params(best_params: dict) -> str:
    """
    把 best_params 字典格式化成可读字符串,便于在界面上显示。
    """
    if not best_params:
        return "（无）"
    lines = []
    for k, v in best_params.items():
        lines.append(f"`{k}` = `{v}`")
    return ", ".join(lines)
