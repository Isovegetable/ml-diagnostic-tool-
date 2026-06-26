"""
使用示例数据生成真实的样例报告（HTML + PDF）。
用法：python scripts/generate_sample_report.py
"""
import base64, os, sys, io, hashlib
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 确保能 import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_tabular_data, prepare_regression_data
from src.data.preprocess import detect_outliers_iqr, detect_near_duplicates, normalize_features
from src.models import train_and_evaluate_models
from src.diagnostics import generate_diagnostics, diagnose_level
from src.plots import plot_predicted_vs_actual, plot_hpo_convergence
from src.reports import generate_html_report, generate_pdf_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "发布包")


class _NamedBytesIO(io.BytesIO):
    """给 BytesIO 加上 .name 属性，骗过 load_tabular_data。"""
    def __init__(self, buf, name):
        super().__init__(buf)
        self.name = name


def main():
    data_path = os.path.join(OUT, "04_示例数据.xlsx")
    print(f"[1/8] 加载示例数据: {data_path}")
    with open(data_path, "rb") as f:
        buf = _NamedBytesIO(f.read(), "04_示例数据.xlsx")
    data = load_tabular_data(buf)

    target_column = data.columns[-1]
    print(f"     目标列: {target_column}")
    print(f"     样本数: {len(data)}, 特征数: {len(data.columns) - 1}")

    print("[2/8] 预处理数据...")
    regression_data = prepare_regression_data(data, target_column)
    x_norm, _ = normalize_features(regression_data.x, "zscore")

    print("[3/8] 检测数据质量...")
    numeric_cols = regression_data.x.select_dtypes(include=["number"]).columns.tolist()
    outlier_info, outlier_mask = detect_outliers_iqr(data, factor=1.5)
    outlier_rows = int(outlier_mask.sum())
    near_dup_result = detect_near_duplicates(data, numeric_cols, decimals=2)
    near_dup_rows = near_dup_result.get("total_near_dup", 0)

    print("[4/8] 训练模型 (Random Forest, Gradient Boosting, Linear Regression)...")
    results, metrics_df, hpo_results = train_and_evaluate_models(
        x_norm,
        regression_data.y,
        test_size=0.3,
        cv_method="kfold",
        cv_folds=4,
        hpo_method="bayesian",
        n_trials=15,
        selected_models=["Random Forest", "Gradient Boosting", "Linear Regression"],
    )

    best_model_name = metrics_df.iloc[0]["model"]
    best_result = results[best_model_name]
    print(f"     最佳模型: {best_model_name}")
    print(f"     Test R2: {best_result.test_r2:.4f}, CV R2: {best_result.cv_r2_mean:.4f}")

    print("[5/8] 生成诊断...")
    suggestions = generate_diagnostics(
        sample_size=len(regression_data.model_data),
        feature_count=len(regression_data.feature_columns),
        test_r2=best_result.test_r2,
        train_r2=best_result.train_r2,
        cv_r2_mean=best_result.cv_r2_mean,
        cv_r2_std=best_result.cv_r2_std,
    )
    diagnosis = diagnose_level(
        sample_size=len(regression_data.model_data),
        feature_count=len(regression_data.feature_columns),
        test_r2=best_result.test_r2,
        train_r2=best_result.train_r2,
        cv_r2_mean=best_result.cv_r2_mean,
        cv_r2_std=best_result.cv_r2_std,
        duplicate_rows=int(regression_data.duplicate_rows),
        outlier_rows=outlier_rows,
        near_dup_rows=near_dup_rows,
    )

    print("[6/8] 生成图表...")
    scatter_b64 = None
    convergence_b64 = None
    try:
        fig = plot_predicted_vs_actual(
            best_result.y_test, best_result.test_pred,
            best_result.y_train, best_result.train_pred,
            title=best_model_name,
            r2_test=best_result.test_r2, rmse_test=best_result.rmse,
            r2_train=best_result.train_r2, rmse_train=best_result.train_rmse,
            is_best=True,
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        scatter_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        if hpo_results and best_model_name in hpo_results:
            fig = plot_hpo_convergence(hpo_results[best_model_name])
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            convergence_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close(fig)
    except Exception as e:
        print(f"     图表生成警告: {e}")

    data_fingerprint = hashlib.md5(
        str(regression_data.original_shape).encode() +
        regression_data.target_column.encode() +
        str(regression_data.feature_columns[:3]).encode()
    ).hexdigest()[:20]

    data_quality = {
        "duplicate_rows": int(regression_data.duplicate_rows),
        "outlier_rows": outlier_rows,
        "near_dup_rows": near_dup_rows,
        "dataset": target_column,
        "fingerprint": data_fingerprint,
    }
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    all_params_by_model = {mn: results[mn].model.get_params() for mn in results}

    print("[7/8] 生成 HTML 样例报告...")
    html = generate_html_report(
        original_shape=regression_data.original_shape,
        sample_size=len(regression_data.model_data),
        feature_count=len(regression_data.feature_columns),
        target_column=target_column,
        best_model_name=best_model_name,
        test_r2=best_result.test_r2,
        train_r2=best_result.train_r2,
        cv_r2_mean=best_result.cv_r2_mean,
        cv_r2_std=best_result.cv_r2_std,
        mae=best_result.mae,
        rmse=best_result.rmse,
        suggestions=suggestions,
        hpo_results=hpo_results if hpo_results else None,
        all_params_by_model=all_params_by_model,
        metrics_df=metrics_df,
        scatter_plot_base64=scatter_b64,
        convergence_plot_base64=convergence_b64,
        diagnosis=diagnosis,
        watermark="懂点AI的C学长",
        data_quality=data_quality,
        generated_date=generated_date,
    )
    html_path = os.path.join(OUT, "02_样例报告.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"     OK: {html_path}")

    print("[8/8] 生成 PDF 样例报告...")
    pdf_bytes = generate_pdf_report(
        original_shape=regression_data.original_shape,
        sample_size=len(regression_data.model_data),
        feature_count=len(regression_data.feature_columns),
        target_column=target_column,
        best_model_name=best_model_name,
        test_r2=best_result.test_r2,
        train_r2=best_result.train_r2,
        cv_r2_mean=best_result.cv_r2_mean,
        cv_r2_std=best_result.cv_r2_std,
        mae=best_result.mae,
        rmse=best_result.rmse,
        suggestions=suggestions,
        diagnosis=diagnosis,
        data_quality=data_quality,
        watermark="懂点AI的C学长",
        metrics_df=metrics_df,
        hpo_results=hpo_results if hpo_results else None,
        all_params_by_model=all_params_by_model,
        scatter_plot_base64=scatter_b64,
        convergence_plot_base64=convergence_b64,
        generated_date=generated_date,
    )
    pdf_path = os.path.join(OUT, "02_样例报告.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"     OK: {pdf_path}")

    print("\n✅ 样例报告生成完毕！")


if __name__ == "__main__":
    main()
