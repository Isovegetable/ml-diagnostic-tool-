"""
报告生成模块: 支持 TXT 和 HTML 两种格式。
"""

import base64
from io import BytesIO

import pandas as pd


def generate_report_text(
    original_shape: tuple,
    sample_size: int,
    feature_count: int,
    target_column: str,
    best_model_name: str,
    test_r2: float,
    train_r2: float,
    cv_r2_mean: float,
    cv_r2_std: float,
    mae: float,
    rmse: float,
    suggestions: list[str],
    hpo_results: dict | None = None,
    all_params_by_model: dict | None = None,
    diagnosis: dict | None = None,
) -> str:
    """文本格式报告。"""
    # 诊断等级摘要
    diagnosis_line = ""
    if diagnosis:
        diagnosis_line = f"""
诊断等级：{diagnosis['level']}（{diagnosis['label']}）— 评分 {diagnosis['score']}/100
{diagnosis['reason']}
"""
    hpo_section = ""
    if hpo_results:
        hpo_lines = ["", "超参数优化 (HPO) 结果："]
        methods = {r["method"] for r in hpo_results.values()}
        hpo_lines.append(f"- 优化方法: {', '.join(m.upper() for m in methods)}")
        for model_name, hpo_res in hpo_results.items():
            params_str = ", ".join(f"{k}={v}" for k, v in hpo_res["best_params"].items())
            hpo_lines.append(
                f"- {model_name}: 最佳 CV R² = {hpo_res['best_score']:.4f}, "
                f"RMSE = {hpo_res.get('best_rmse', 0.0):.4f} ({hpo_res['n_trials']} 轮), "
                f"调优超参 = {{{params_str}}}"
            )
            if all_params_by_model and model_name in all_params_by_model:
                all_p = all_params_by_model[model_name]
                tuned_keys = set(hpo_res["best_params"].keys())
                other_params = {k: v for k, v in all_p.items() if k not in tuned_keys}
                if other_params:
                    other_str = ", ".join(f"{k}={v}" for k, v in other_params.items())
                    hpo_lines.append(f"    其他超参(默认): {{{other_str}}}")
        hpo_section = "\n".join(hpo_lines)

    return f"""材料机器学习自动诊断报告
{diagnosis_line}
数据概况：
- 原始样本数：{original_shape[0]}
- 原始列数：{original_shape[1]}
- 有效建模样本数：{sample_size}
- 数值特征数：{feature_count}
- 目标列：{target_column}

最佳模型：{best_model_name}

模型评估结果：
- 交叉验证 R² (CV): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}
- 训练集 R² (Train): {train_r2:.4f}
- 测试集 R² (Test): {test_r2:.4f}
- 测试集 MAE: {mae:.4f}
- 测试集 RMSE: {rmse:.4f}
{hpo_section}

评估说明：
- CV R²：在训练集上交叉验证的平均表现，标准差越小模型越稳定
- Train R²：模型在训练集上的拟合效果
- Test R²：模型在测试集上的预测效果（最重要的指标）

诊断建议：
{chr(10).join(["- " + item for item in suggestions])}

使用说明与免责声明：
本工具仅用于科研数据初步分析和机器学习建模可行性判断。
所有结果均基于用户上传数据和自动化模型流程生成，不代表因果结论，不保证论文发表。
对于高 R² 或异常结果，仍需结合交叉验证、外部验证集、材料机理和实验重复性进一步确认。
请勿上传未授权、涉密或包含商业机密的数据。
"""


def _model_comparison_rows(metrics_df: pd.DataFrame) -> str:
    """从 metrics_df 生成 HTML 表格行。"""
    rows = []
    for _, row in metrics_df.iterrows():
        rows.append(f"""<tr>
            <td>{row['model']}</td>
            <td>{row['train_r2']:.4f}</td>
            <td>{row['test_r2']:.4f}</td>
            <td>{row['cv_r2_mean']:.4f} ± {row['cv_r2_std']:.4f}</td>
            <td>{row['mae']:.4f}</td>
            <td>{row['rmse']:.4f}</td>
        </tr>""")
    return "\n".join(rows)


def _hpo_section_html(
    hpo_results: dict,
    all_params_by_model: dict | None = None,
) -> str:
    """HPO 结果的 HTML 片段。"""
    if not hpo_results:
        return ""
    methods = {r["method"] for r in hpo_results.values()}
    method_str = ", ".join(m.upper() for m in methods)

    items = []
    for model_name, hpo_res in hpo_results.items():
        params_str = ", ".join(f"{k}={v}" for k, v in hpo_res["best_params"].items())
        item = (
            f'<div class="hpo-item">'
            f'<strong>{model_name}</strong> — '
            f'CV R² = {hpo_res["best_score"]:.4f}, '
            f'RMSE = {hpo_res.get("best_rmse", 0.0):.4f} '
            f'({hpo_res["n_trials"]} 轮)'
            f'<br><code>{{{params_str}}}</code>'
        )
        if all_params_by_model and model_name in all_params_by_model:
            all_p = all_params_by_model[model_name]
            tuned_keys = set(hpo_res["best_params"].keys())
            other = {k: v for k, v in all_p.items() if k not in tuned_keys}
            if other:
                other_str = ", ".join(f"{k}={v}" for k, v in other.items())
                item += f'<br><span class="dim">其他(默认): {{{other_str}}}</span>'
        item += "</div>"
        items.append(item)

    return f"""
    <h3>超参数优化 (HPO) 结果</h3>
    <p>优化方法: <strong>{method_str}</strong></p>
    {chr(10).join(items)}
    """


def _diagnosis_html(diagnosis: dict) -> str:
    """生成诊断等级的 HTML 卡片。"""
    level = diagnosis["level"]
    label = diagnosis["label"]
    score = diagnosis["score"]
    colors = {"A": "#1a7d36", "B": "#b8860b", "C": "#cc5500", "D": "#b30000"}
    emojis = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
    color = colors.get(level, "#666")
    emoji = emojis.get(level, "⚪")

    details = "".join(
        f"<li>{d}</li>" for d in diagnosis.get("details", [])
    )
    return f"""
    <div style="display:flex;align-items:stretch;gap:16px;margin:16px 0;
                background:{color}08;border:1.5px solid {color};border-radius:10px;padding:16px;">
      <div style="text-align:center;min-width:80px;padding:8px;">
        <div style="font-size:36px">{emoji}</div>
        <div style="font-size:28px;font-weight:bold;color:{color}">{level}</div>
        <div style="font-size:13px;color:{color}">{label}</div>
        <div style="font-size:12px;color:#888;">{score}/100</div>
      </div>
      <div style="flex:1;">
        <div style="font-weight:bold;margin-bottom:6px;">{diagnosis.get('reason', '')}</div>
        <ul style="margin:4px 0 0 0;padding-left:18px;font-size:13px;color:#555;">{details}</ul>
      </div>
    </div>
    """


def _watermark_css(text: str | None) -> str:
    """如果设置了水印，返回水印 CSS。"""
    if not text:
        return ""
    return """
  .watermark {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    transform: rotate(-30deg);
    font-size: 60px; font-weight: bold; color: rgba(200,0,0,0.12);
    user-select: none;
  }
"""


def _watermark_overlay(text: str | None) -> str:
    """如果设置了水印，返回水印 HTML 叠加层。"""
    if not text:
        return ""
    return f'<div class="watermark">{text}</div>'


def _data_quality_html(dq: dict | None) -> str:
    """数据质量摘要行。"""
    if not dq:
        return ""
    parts = []
    dup = dq.get("duplicate_rows", 0)
    out = dq.get("outlier_rows", 0)
    near = dq.get("near_dup_rows", 0)
    if dup > 0:
        parts.append(f"重复行：<strong>{dup}</strong>")
    if near > 0:
        parts.append(f"近重复：<strong>{near}</strong>")
    if out > 0:
        parts.append(f"异常值：<strong>{out}</strong>")
    if not parts:
        return ""
    return f'<p style="font-size:13px;color:#666;margin:8px 0 0;">{"&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts)}</p>'


def _next_steps_html(diagnosis: dict | None, test_r2: float, sample_size: int) -> str:
    """根据诊断结果生成"下一步实验建议"。"""
    if not diagnosis:
        return ""
    level = diagnosis["level"]
    steps = []

    if level == "A":
        steps = [
            "优先关注 SHAP 重要性排名前 3 的特征，围绕这些变量设计验证实验",
            "检查特征之间的物理/化学关联性，确认模型识别的关系是否合理",
            "如目标 R² 满足要求，可进入论文建模阶段，建议补充外部验证集",
        ]
    elif level == "B":
        steps = [
            "特征数/样本比偏低，建议补充 30%–50% 的样本量",
            "检查是否有高相关特征群（相关性热力图），可合并或保留其中一个",
            "考虑增加与目标变量物理相关的特征，减少无关或噪声特征",
            "检查异常值来源，确认是测量误差还是真实极端条件",
        ]
    elif level == "C":
        steps = [
            "当前模型不稳定，不建议用于正式分析",
            "优先排查数据质量问题：异常值、近重复、缺失模式",
            "检查目标变量分布是否存在极端偏态，考虑对数变换",
            "建议大幅扩充样本量（当前数据量下模型不可靠）",
            "如有可能，寻找文献中同类材料的已知特征-性能关系作为先验",
        ]
    elif level == "D":
        steps = [
            "当前数据不适合机器学习建模",
            "建议重新评估数据采集方案：增加样本量、控制变量、降低测量误差",
            "检查目标变量的变异范围是否足够大（过小的变异无法建模）",
            "考虑是否用了正确的目标变量（数值型 vs 类别型）",
        ]

    items = "".join(f"<li>{s}</li>" for s in steps)
    return f"""
    <h3>下一步建议</h3>
    <div class="suggestions"><ol>{items}</ol></div>
    """


def generate_html_report(
    original_shape: tuple,
    sample_size: int,
    feature_count: int,
    target_column: str,
    best_model_name: str,
    test_r2: float,
    train_r2: float,
    cv_r2_mean: float,
    cv_r2_std: float,
    mae: float,
    rmse: float,
    suggestions: list[str],
    hpo_results: dict | None = None,
    all_params_by_model: dict | None = None,
    metrics_df: pd.DataFrame | None = None,
    scatter_plot_base64: str | None = None,
    convergence_plot_base64: str | None = None,
    diagnosis: dict | None = None,
    watermark: str | None = None,
    data_quality: dict | None = None,
    generated_date: str | None = None,
) -> str:
    """
    生成完整的 HTML 诊断报告（含样式、表格、嵌入图表）。

    Args:
        参数含义与 generate_report_text 相同,额外参数:
        metrics_df: 模型对比 DataFrame(含 model, train_r2, test_r2, cv_r2_mean, cv_r2_std, mae, rmse)
        scatter_plot_base64: 预测散点图的 base64 PNG 编码
        convergence_plot_base64: HPO 收敛曲线的 base64 PNG 编码

    Returns:
        str: 完整的 HTML 文档
    """
    # 模型对比表格
    comparison_table = ""
    if metrics_df is not None:
        comparison_table = f"""
        <h3>模型对比</h3>
        <table>
            <tr><th>模型</th><th>Train R²</th><th>Test R²</th><th>CV R²</th><th>MAE</th><th>RMSE</th></tr>
            {_model_comparison_rows(metrics_df)}
        </table>
        """

    # HPO 章节
    hpo_html = _hpo_section_html(hpo_results, all_params_by_model)

    # 嵌入图表
    scatter_img = ""
    if scatter_plot_base64:
        scatter_img = f'<div class="chart"><img src="data:image/png;base64,{scatter_plot_base64}" alt="Predicted vs Actual" style="max-width:600px;width:100%"></div>'
    convergence_img = ""
    if convergence_plot_base64:
        convergence_img = f'<div class="chart"><img src="data:image/png;base64,{convergence_plot_base64}" alt="HPO Convergence" style="max-width:600px;width:100%"></div>'

    # 诊断建议
    sug_html = "".join(f"<li>{s}</li>" for s in suggestions)

    report = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>材料机器学习诊断报告</title>
<style>
  body {{
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    max-width: 900px; margin: 0 auto; padding: 30px 20px;
    background: #fafafa; color: #333; line-height: 1.7; font-size: 14px;
  }}
  h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 8px; }}
  h2 {{ color: #333; margin-top: 28px; }}
  h3 {{ color: #555; margin-top: 20px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
  th {{ background: #1f77b4; color: white; }}
  tr:nth-child(even) {{ background: #f7f9fc; }}
  .card {{
    background: white; border-radius: 8px; padding: 16px 20px; margin: 12px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  .metric-grid {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .metric-box {{
    flex: 1; min-width: 130px; background: #f0f4f8; padding: 12px;
    border-radius: 6px; text-align: center;
  }}
  .metric-box .value {{ font-size: 22px; font-weight: bold; color: #1f77b4; }}
  .metric-box .label {{ font-size: 12px; color: #666; }}
  .hpo-item {{ background: #f7f9fc; padding: 10px 14px; margin: 8px 0; border-radius: 6px; font-size: 13px; }}
  .hpo-item code {{ font-size: 12px; background: #e8edf4; padding: 2px 6px; border-radius: 3px; }}
  .dim {{ color: #888; font-size: 12px; }}
  .chart {{ text-align: center; margin: 16px 0; }}
  .suggestions {{ background: #fff8e1; border-left: 4px solid #ffa000; padding: 14px 18px; border-radius: 4px; }}
  .suggestions li {{ margin: 6px 0; }}
  .disclaimer {{
    margin-top: 30px; font-size: 12px; color: #888; padding: 14px;
    border-top: 1px solid #ddd; background: #f5f5f5; border-radius: 4px;
  }}
  footer {{ margin-top: 20px; font-size: 11px; color: #aaa; text-align: center; }}
  {_watermark_css(watermark)}
</style>
</head>
<body>
{_watermark_overlay(watermark)}

<!-- 封面 -->
<div style="text-align:center;padding:30px 20px 20px;margin-bottom:24px;
            border-bottom:3px solid #1f77b4;">
  <h1 style="font-size:28px;margin:0;color:#1f77b4;border:none;padding:0;">
    材料机器学习诊断报告
  </h1>
  <p style="color:#888;font-size:13px;margin:8px 0 0;">
    生成日期：{generated_date or '—'} &nbsp;|&nbsp;
    数据集：{data_quality.get("dataset", "未命名") if data_quality else "未命名"}
  </p>
</div>

{_diagnosis_html(diagnosis) if diagnosis else ""}

<!-- 数据概况 -->
<div class="card">
  <h3 style="margin-top:0;">数据概况</h3>
  <table>
    <tr><td>原始样本数</td><td>{original_shape[0]}</td><td>有效建模样本数</td><td>{sample_size}</td></tr>
    <tr><td>原始列数</td><td>{original_shape[1]}</td><td>数值特征数</td><td>{feature_count}</td></tr>
    <tr><td colspan="2">目标列</td><td colspan="2">{target_column}</td></tr>
  </table>
  {_data_quality_html(data_quality)}
</div>

<h2>最佳模型: {best_model_name}</h2>
<div class="card">
  <div class="metric-grid">
    <div class="metric-box"><div class="value">{cv_r2_mean:.4f}</div><div class="label">CV R²</div><div class="dim">± {cv_r2_std:.4f}</div></div>
    <div class="metric-box"><div class="value">{train_r2:.4f}</div><div class="label">Train R²</div></div>
    <div class="metric-box"><div class="value">{test_r2:.4f}</div><div class="label">Test R²</div></div>
    <div class="metric-box"><div class="value">{mae:.4f}</div><div class="label">MAE</div></div>
    <div class="metric-box"><div class="value">{rmse:.4f}</div><div class="label">RMSE</div></div>
  </div>
</div>

{comparison_table}

{hpo_html}

{convergence_img}

{scatter_img}

<h3>诊断建议</h3>
<div class="suggestions"><ul>{sug_html}</ul></div>

{_next_steps_html(diagnosis, test_r2, sample_size) if diagnosis else ""}

<div class="disclaimer">
  <strong>免责声明</strong><br>
  本工具仅用于科研数据初步分析和机器学习建模可行性判断，不构成学术结论或发表依据。<br>
  所有结果均基于用户上传数据和自动化模型流程生成，不代表因果结论，不保证论文发表。<br>
  对于高 R² 或异常结果，仍需结合交叉验证、外部验证集、材料机理和实验重复性进一步确认。<br>
  请勿上传未授权、涉密或包含商业机密的数据。使用本工具产生的任何决策或结论由用户自行负责。
</div>

<footer>
  材料机器学习自动诊断工具 v1.0 &nbsp;|&nbsp;
  生成时间：{generated_date or '—'} &nbsp;|&nbsp;
  数据指纹：{data_quality.get("fingerprint", "—") if data_quality else "—"}
</footer>
</body>
</html>
"""
    return report
