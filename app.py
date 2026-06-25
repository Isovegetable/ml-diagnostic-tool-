import streamlit as st
import pandas as pd
import warnings

# 静默 LightGBM 的"无分裂增益"警告(小数据集常见,不影响结果)
warnings.filterwarnings("ignore", message="No further splits with positive gain")

from src.data import load_tabular_data, prepare_regression_data
from src.diagnostics import generate_diagnostics, diagnose_level
from src.models import HPO_METHODS, get_search_space, train_and_evaluate_models
from src.plots import (
    plot_correlation_heatmap,
    plot_missing_values,
    plot_predicted_vs_actual,
    plot_target_distribution,
    decide_correlation_method,
    plot_hpo_convergence,
    format_best_params,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_partial_dependence,
    plot_pdp_with_individual,
)
from src.reports import generate_html_report, generate_report_text, generate_pdf_report

# 页面配置
st.set_page_config(
    page_title="材料机器学习自动诊断 Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 锚点滚动偏移（防止标题被 Streamlit 顶栏遮挡）
st.markdown(
    """
<style>
/* 所有小节标题跳转时留出顶栏空间 */
[id^="sec-"] {
    scroll-margin-top: 80px;
}
</style>
""",
    unsafe_allow_html=True,
)

# 初始化 session state
if "current_step" not in st.session_state:
    st.session_state.current_step = "Step 1"

# ========== 侧边栏：流程导航 + 使用指南 ==========
with st.sidebar:
    st.header("🗂️ 流程导航")

    # 流程架构树
    st.markdown(
        """
    <style>
    .nav-tree {
        font-size: 14px;
        line-height: 1.8;
    }
    .nav-item {
        cursor: pointer;
        padding: 2px 0;
    }
    .nav-step {
        font-weight: bold;
        color: #1f77b4;
    }
    .nav-substep {
        margin-left: 20px;
        color: #666;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 导航按钮
    step_choice = st.radio(
        "选择步骤：",
        ["Step 1: 数据加载与探索", "Step 2: 模型训练与评估", "Step 3: 可解释性分析"],
        index=0 if st.session_state.current_step == "Step 1" else (1 if st.session_state.current_step == "Step 2" else 2),
        label_visibility="collapsed"
    )

    # 更新当前步骤
    if "Step 1" in step_choice:
        st.session_state.current_step = "Step 1"
    elif "Step 2" in step_choice:
        st.session_state.current_step = "Step 2"
    else:
        st.session_state.current_step = "Step 3"

    # 显示当前步骤的子项（可点击跳转）
    if st.session_state.current_step == "Step 1":
        st.markdown(
            """
        **当前步骤内容：**
        - [1.1 上传数据](#sec-1-1)
        - [1.2 数据概览](#sec-1-2)
        - [1.3 数据质量检查](#sec-1-3)
        - [1.4 选择预测目标](#sec-1-4)
        - [1.5 数据预处理](#sec-1-5)
        - [1.6 数据探索可视化](#sec-1-6)
        - [1.7 导出处理后数据](#sec-1-7)
        - [1.8 特征工程 ← 可选](#sec-1-8)
        """
        )
    elif st.session_state.current_step == "Step 2":
        st.markdown(
            """
        **当前步骤内容：**
        - [2.1 评估配置](#sec-2-1)
        - [2.2 训练模型](#sec-2-2)
        - [2.3 模型对比结果](#sec-2-3)
        - [2.4 可视化分析](#sec-2-4)
        - [2.5 自动诊断建议](#sec-2-5)
        - [2.6 报告下载](#sec-2-6)
        """
        )
    else:
        st.markdown(
            """
        **当前步骤内容：**
        - [3.1 SHAP 可解释性分析](#sec-3-1)
        - [3.2 PDP 偏依赖图](#sec-3-3)
        - [3.3 ICE 曲线](#sec-3-4)
        """
        )

    st.divider()
    st.header("📖 使用指南")

    with st.expander("🎯 这个工具是什么？", expanded=True):
        st.markdown(
            """
        这是一个**材料实验数据快速诊断工具**，帮助你：

        - 快速判断数据是否适合做机器学习
        - 自动训练多个模型并对比效果
        - 生成诊断报告和改进建议

        **适合场景：**
        - 材料性能预测（强度、硬度、导电率等）
        - 实验数据初步分析
        - 论文/课题前期可行性验证
        """
        )

    with st.expander("📝 使用步骤"):
        st.markdown(
            """
        **Step 1：数据加载与探索**
        - 上传 Excel/CSV 数据
        - 查看数据质量和分布
        - 处理缺失值和类别特征
        - 可导出处理后的数据

        **Step 2：模型训练与评估**
        - 配置评估方式（交叉验证）
        - 自动训练多个模型
        - 查看性能对比和诊断

        **Step 3：可解释性分析**
        - SHAP 特征重要性分析
        - 偏依赖图 (PDP)
        - ICE 曲线
        """
        )

    with st.expander("📊 数据格式要求"):
        st.markdown(
            """
        **必须满足：**
        - ✅ 表格格式（行=样本，列=特征）
        - ✅ 至少 10 行数据
        - ✅ 预测目标必须是数字（不能是文字）

        **示例数据格式：**

        | cement | water | age | compress_strength |
        |--------|-------|-----|-------------------|
        | 540    | 162   | 28  | 79.99            |
        | 540    | 162   | 28  | 61.89            |

        **预测目标**列（如 compress_strength）必须是数字。
        """
        )

    with st.expander("❓ 常见问题"):
        st.markdown(
            """
        **Q: 报错"只支持数值型目标列"？**
        A: 预测目标必须是纯数字，不能是文字（如"高/中/低"）。

        **Q: 什么是"交叉验证"？**
        A: 在训练集内部多次划分训练/验证，评估模型稳定性。

        **Q: R² 是什么？**
        A: 模型预测准确度指标，越接近 1 越好。
        - 0.7-1.0：效果好
        - 0.4-0.7：中等
        - <0.4：效果差

        **Q: 结果可以直接用于论文吗？**
        A: 不可以！这只是初步筛选工具，正式发表需要更严格的验证。
        """
        )

    st.divider()
    st.caption("⚠️ 请勿上传涉密数据")
    st.caption("📄 结果仅供参考，不保证论文发表")

# ========== 主页面 ==========
st.title("材料机器学习自动诊断 Demo")
st.caption("分步骤完成：数据预处理 → 模型训练 → 可解释性分析")

# ========================================================================
# Step 1: 数据加载与探索
# ========================================================================
with st.expander("📊 Step 1: 数据加载与探索", expanded=(st.session_state.current_step == "Step 1")):
    st.markdown('<a id="sec-1-1"></a>', unsafe_allow_html=True)
    st.subheader("1.1 上传数据")

    if "example_loaded" not in st.session_state:
        st.session_state.example_loaded = False
    if "example_data" not in st.session_state:
        st.session_state.example_data = None

    uploaded_file = st.file_uploader(
        "📁 上传你的材料实验数据", type=["csv", "xlsx", "xls"]
    )

    if not uploaded_file and not st.session_state.example_loaded:
        st.info("👆 请先上传 Excel 或 CSV 文件开始分析")
        st.markdown("---")
        st.markdown("### 💡 示例：预测水泥强度")
        st.markdown(
            """
        假设你有一份水泥实验数据：

        | cement | water | slag | flyash | age | compress_strength |
        |--------|-------|------|--------|-----|-------------------|
        | 450    | 180   | 50   | 80     | 28  | 52.3              |
        | 380    | 165   | 120  | 40     | 90  | 61.2              |
        | 520    | 175   | 0    | 120    | 365 | 73.8              |

        **你会得到：**
        - ✅ 数据质量报告
        - ✅ 多个模型对比结果
        - ✅ 预测准确度评估
        - ✅ 诊断建议和改进方向
        """
        )

        # 示例数据加载函数
        from io import BytesIO

        class _NamedBytesIO(BytesIO):
            def __init__(self, buf, name):
                super().__init__(buf)
                self.name = name

        def _load_example(filepath: str, label: str):
            with open(filepath, "rb") as f:
                data = load_tabular_data(_NamedBytesIO(f.read(), label))
            st.session_state.example_data = data
            st.session_state.example_loaded = True
            st.rerun()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🏗️ 水泥强度 (50条)", width="stretch", type="primary"):
                _load_example("src/data/example_concrete.csv", "concrete.csv")
        with col_b:
            if st.button("🔋 电池容量 (100条)", width="stretch"):
                _load_example("src/data/example_battery.csv", "battery.csv")
        with col_c:
            if st.button("⚗️ 催化效率 (45条)", width="stretch"):
                _load_example("src/data/example_catalyst.csv", "catalyst.csv")

        st.stop()

    # 加载数据(示例 or 上传)
    try:
        if st.session_state.example_loaded and not uploaded_file:
            data = st.session_state.example_data
        else:
            data = load_tabular_data(uploaded_file)
            st.session_state.example_loaded = False
    except Exception as error:
        st.error(f"文件读取失败：{error}")
        st.stop()

    st.success(f"✅ 数据加载成功！样本数：{data.shape[0]}，列数：{data.shape[1]}")

    # ========== 1.2 数据概览 ==========
    st.markdown('<a id="sec-1-2"></a>', unsafe_allow_html=True)
    st.subheader("1.2 数据概览")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("📄 原始数据预览（前 10 行）")
        st.dataframe(data.head(10), width="stretch", height=300)
    with col2:
        st.caption("📊 统计量：count(计数) / mean(均值) / std(标准差) / min(最小值) / 25%(下四分位) / 50%(中位数) / 75%(上四分位) / max(最大值)")
        st.dataframe(data.describe(), width="stretch", height=300)

    # ========== 1.3 数据质量检查 ==========
    st.markdown("---")
    st.markdown('<a id="sec-1-3"></a>', unsafe_allow_html=True)
    st.subheader("1.3 数据质量检查")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总样本数", data.shape[0])
    col2.metric("总列数", data.shape[1])
    col3.metric("缺失值总数", int(data.isna().sum().sum()))
    col4.metric("重复行数", int(data.duplicated().sum()))

    # 缺失值可视化
    missing_fig = plot_missing_values(data)
    if missing_fig:
        st.pyplot(missing_fig)
    else:
        st.success("✅ 没有缺失值")

    # ========== 异常值检测 ==========
    from src.data import detect_outliers_iqr, detect_near_duplicates

    outlier_info, outlier_mask = detect_outliers_iqr(data, factor=1.5)
    n_outlier_rows = int(outlier_mask.sum())
    st.session_state["_outlier_rows"] = n_outlier_rows
    if n_outlier_rows > 0:
        with st.expander(f"📊 异常值检测（IQR 法）— 发现 {n_outlier_rows} 行含异常值", expanded=False):
            for col, info in outlier_info.items():
                pct = info["count"] / len(data) * 100
                st.write(
                    f"**{col}**: {info['count']} 行 ({pct:.1f}%) "
                    f"超出 [{info['lower']:.3f}, {info['upper']:.3f}]"
                )
            st.caption(
                "异常值不一定需要删除，建议先检查原始数据是否记录错误。"
                "IQR 法对偏态分布较为稳健。"
            )
    else:
        st.success("✅ 未发现明显异常值")

    # ========== 近重复样本检测 ==========
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    near_dup_result = detect_near_duplicates(data, numeric_cols, decimals=2)
    st.session_state["_near_dup_rows"] = near_dup_result["total_near_dup"]
    if near_dup_result["total_near_dup"] > 0:
        n_groups = len(near_dup_result["groups"])
        with st.expander(
            f"🔍 近重复样本检测 — {near_dup_result['total_near_dup']} 行参与 "
            f"（{n_groups} 组）",
            expanded=False,
        ):
            st.caption(
                "近重复指数值四舍五入到 2 位小数后完全相同的行，"
                "常见于同配方多次测试。不影响建模但会高估模型可信度。"
            )
            for i, group in enumerate(near_dup_result["groups"][:10]):
                st.write(f"**组 {i + 1}**（{group['count']} 行）: 行号 {group['rows']}")
            if len(near_dup_result["groups"]) > 10:
                st.write(f"... 还有 {len(near_dup_result['groups']) - 10} 组")
    else:
        st.success("✅ 未发现近重复样本")

    # ========== 1.4 选择目标列 ==========
    st.markdown("---")
    st.markdown('<a id="sec-1-4"></a>', unsafe_allow_html=True)
    st.subheader("1.4 选择预测目标")

    target_column = st.selectbox(
        "请选择预测目标列（要预测的那一列，如材料强度、导电率等）",
        data.columns,
        index=len(data.columns) - 1,
        help="预测目标必须是数值型列，不能是文字或类别",
    )

    # ========== 1.5 数据预处理 ==========
    st.markdown("---")
    st.markdown('<a id="sec-1-5"></a>', unsafe_allow_html=True)
    st.subheader("1.5 数据预处理")

    try:
        regression_data = prepare_regression_data(data, target_column)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("有效建模样本", len(regression_data.model_data))
    col2.metric("数值特征数", len(regression_data.feature_columns))
    col3.metric("删除的缺失行", regression_data.removed_missing_rows)

    # 显示类别特征编码信息
    if regression_data.encoded_categorical_info:
        st.info(
            f"✅ 已自动编码 {len(regression_data.encoded_categorical_info)} 个类别特征: "
            + ", ".join(
                [
                    f"{col} ({n}类)"
                    for col, n in regression_data.encoded_categorical_info.items()
                ]
            )
        )

    # ========== 1.6 数据探索可视化 ==========
    st.markdown("---")
    st.markdown('<a id="sec-1-6"></a>', unsafe_allow_html=True)
    st.subheader("1.6 数据探索可视化")

    # 目标列分布
    st.markdown("**目标列分布**")
    target_dist_fig = plot_target_distribution(regression_data.y, target_column)
    st.pyplot(target_dist_fig)

    # 特征相关性
    st.markdown("**特征相关性热力图**")

    # 正态性检验 + 自动选择相关性方法(严格标准)
    corr_method, normality_results = decide_correlation_method(
        regression_data.model_data, alpha=0.05
    )

    n_total = len(normality_results)
    n_normal = sum(1 for r in normality_results.values() if r["is_normal"] is True)
    n_abnormal = sum(1 for r in normality_results.values() if r["is_normal"] is False)
    n_undetermined = n_total - n_normal - n_abnormal

    abnormal_cols = [
        (col, r["p_value"])
        for col, r in normality_results.items()
        if r["is_normal"] is False
    ]

    st.info(
        f"📊 **正态性检验 (Shapiro-Wilk, α=0.05)**\n\n"
        f"- 共检测 {n_total} 个数值列\n"
        f"- 通过正态性检验: {n_normal} 列 ✅\n"
        f"- 未通过: {n_abnormal} 列 ❌"
        + (f"\n- 样本量不足无法判断: {n_undetermined} 列 ⚠️" if n_undetermined else "")
    )

    if abnormal_cols:
        detail = ", ".join([f"`{col}` (p={p:.4f})" for col, p in abnormal_cols])
        st.warning(f"**未通过正态性检验的列:** {detail}")

    # p 值解读说明
    with st.expander("📖 怎么通过 p 值判断正态分布?", expanded=False):
        st.markdown(
            """
**Shapiro-Wilk 检验的判断规则(α=0.05):**

| p 值范围 | 结论 | 含义 |
|---|---|---|
| `p > 0.05` | ✅ 正态 | 没有足够证据拒绝"数据服从正态分布"这一假设 |
| `p ≤ 0.05` | ❌ 非正态 | 有显著证据表明数据偏离正态分布 |
| `p ≈ 0.0000` | ❌ 强烈非正态 | p 实际非常小(通常 < 0.0001),强烈拒绝正态假设 |

**核心逻辑:**
- **原假设 H₀**:该列数据服从正态分布
- p 值是"在 H₀ 成立的前提下,看到当前数据(或更极端数据)的概率"
- p 越小 → 当前数据在正态假设下越"反常" → 越不应该相信它服从正态

**为什么 p = 0.0000 也常见?**
- p 值显示到小数点后 4 位,真实 p 可能只是 < 0.0001,不是数学意义上的 0
- 大样本下,即使偏离正态很小,Shapiro-Wilk 也能给出极小的 p 值
- 这时建议结合直方图肉眼判断,不要完全依赖 p 值

**为什么 p < 0.05 就要换 Spearman?**
- Pearson 相关系数的计算公式假设数据近似服从正态分布
- 如果数据严重偏态(右偏/左偏)或存在离群点,Pearson 系数会被"拉偏"
- Spearman 使用数据的**秩(排名)**,对分布形态没有要求,更稳健
- 所以"全部通过才用 Pearson"是更保守、更安全的选择

**额外提示:**
- 通过正态检验 ≠ 一定服从正态,只是"没有证据反驳"
- 未通过正态检验也未必完全不能用 Pearson,只是风险更大
- 本工具选择严格标准,优先保证结论的稳健性
            """
        )

    if corr_method == "pearson":
        st.success(
            f"**决策: 使用 Pearson 相关性** (全部数值列均通过正态性检验)"
        )
    else:
        reason = (
            "部分列未通过正态性检验"
            if n_abnormal > 0
            else "无法判断所有列的正态性,默认使用 Spearman 更稳健"
        )
        st.success(f"**决策: 使用 Spearman 相关性** ({reason})")

    corr_fig = plot_correlation_heatmap(
        regression_data.model_data, target_column, method=corr_method
    )
    st.pyplot(corr_fig)

    # ========== 1.7 导出处理后数据 ==========
    st.markdown("---")
    st.markdown('<a id="sec-1-7"></a>', unsafe_allow_html=True)
    st.subheader("1.7 导出处理后数据（可选）")

    csv_data = regression_data.model_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 下载处理后的数据 (CSV)",
        data=csv_data,
        file_name="preprocessed_data.csv",
        mime="text/csv",
    )

    # ========== 1.8 特征工程（可选） ==========
    st.markdown("---")
    st.markdown('<a id="sec-1-8"></a>', unsafe_allow_html=True)
    st.subheader("1.8 特征工程（可选）")
    st.caption(
        "依次进行：特征筛选（可选）→ 归一化（默认 Z-Score）。"
        "特征筛选在归一化之前，否则低方差过滤会失效。"
    )

    from src.data import normalize_features, filter_features

    norm_method = st.selectbox(
        "特征归一化方式",
        options=[
            "Z-Score (标准化，均值0方差1) — 推荐",
            "MinMax (缩放到0~1)",
            "Robust (中位数±IQR，抗异常值)",
        ],
        index=0,
        help=(
            "所有模型训练前都会进行归一化，保证特征尺度一致。\n"
            "Z-Score: 适合大部分场景，处理后每列均值为0，方差为1\n"
            "MinMax: 缩放到[0,1]区间，对分布范围敏感\n"
            "Robust: 用中位数和 IQR，数据有异常值时比 Z-Score 稳定"
        ),
    )

    # 映射选择到内部方法名
    norm_map = {
        "Z-Score (标准化，均值0方差1) — 推荐": "zscore",
        "MinMax (缩放到0~1)": "minmax",
        "Robust (中位数±IQR，抗异常值)": "robust",
    }
    norm_method_internal = norm_map[norm_method]

    with st.expander("特征筛选选项", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            use_var_filter = st.checkbox(
                "剔除低方差特征", value=False,
                help="剔除方差接近0的常数列，这些特征对模型没有区分能力",
            )
            var_threshold = 0.0
            if use_var_filter:
                var_threshold = st.slider(
                    "方差阈值", 0.0, 0.05, 0.01, step=0.005,
                    help="方差低于此值的特征将被剔除",
                )
        with col2:
            use_corr_filter = st.checkbox(
                "剔除高相关特征", value=False,
                help="剔除相关系数过高的冗余特征（保留第一个）",
            )
            corr_threshold = 1.0
            if use_corr_filter:
                corr_threshold = st.slider(
                    "相关性阈值", 0.85, 0.99, 0.95, step=0.01,
                    help="相关系数高于此值的特征对将被剔除其中一个",
                )

    # 应用特征工程
    x_for_model = regression_data.x.copy()
    feat_engineer_info = {"norm_method": norm_method, "dropped": []}

    # 1️⃣ 先特征筛选（在归一化之前，否则低方差过滤失效）
    if use_var_filter or use_corr_filter:
        x_for_model, filter_info = filter_features(
            x_for_model,
            var_threshold=var_threshold,
            corr_threshold=corr_threshold,
        )
        feat_engineer_info["dropped"] = (
            filter_info["dropped_low_variance"] + filter_info["dropped_high_corr"]
        )
        feat_engineer_info["kept_columns"] = filter_info["kept_columns"]

        if filter_info["dropped_low_variance"]:
            st.info(
                f"📉 已剔除 {len(filter_info['dropped_low_variance'])} 个低方差特征: "
                + ", ".join(filter_info["dropped_low_variance"])
            )
        if filter_info["dropped_high_corr"]:
            st.info(
                f"🔗 已剔除 {len(filter_info['dropped_high_corr'])} 个高相关特征: "
                + ", ".join(filter_info["dropped_high_corr"])
            )
    else:
        feat_engineer_info["kept_columns"] = x_for_model.columns.tolist()

    # 2️⃣ 后归一化（必须做）
    x_for_model, _ = normalize_features(x_for_model, norm_method_internal)
    feat_engineer_info["norm_applied"] = True

    # 存入 session_state 供 Step 2 使用
    st.session_state.x_for_model = x_for_model
    st.session_state.feat_engineer_info = feat_engineer_info

    # 显示特征变化摘要
    n_orig = len(regression_data.feature_columns)
    n_now = x_for_model.shape[1]
    if n_now < n_orig:
        st.warning(f"特征数从 {n_orig} 个减少到 {n_now} 个（筛除了 {n_orig - n_now} 个）")
    else:
        st.success(f"✅ 特征工程完成，当前 {n_now} 个特征将用于模型训练")

    st.success("✅ Step 1 完成！数据已准备就绪，可以开始训练模型。")

# ========================================================================
# Step 2: 模型训练与评估
# ========================================================================
with st.expander("🤖 Step 2: 模型训练与评估", expanded=(st.session_state.current_step == "Step 2")):
    st.markdown('<a id="sec-2-1"></a>', unsafe_allow_html=True)
    # 告诉用户修改配置时会刷新页面，避免看到白屏惊慌
    st.info(
        "💡 **页面刷新提示：** 修改下方的任何配置后，页面会短暂刷新（出现空白），"
        "这是该应用的**正常行为**，不是报错，配置也不会丢失。\n\n"
        "每次修改后稍等 1-2 秒，页面就会恢复。配置完成后，点击「开始训练」即可运行模型。",
    )

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "测试集比例",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="留出一部分数据来测试模型准确度。0.2 表示留 20% 的数据做测试。",
        )

    with col2:
        # 根据样本量智能推荐交叉验证方式
        sample_count = len(regression_data.model_data)
        if sample_count < 50:
            default_cv = "留一法 (样本<50，推荐)"
            cv_help = "样本量较少，推荐使用留一法获得最稳定的评估。"
        else:
            default_cv = "5折交叉验证 (推荐)"
            cv_help = "在训练集内部做交叉验证，评估模型稳定性。"

        cv_option = st.selectbox(
            "交叉验证方式（在训练集上）",
            options=["5折交叉验证 (推荐)", "10折交叉验证 (更稳定)", "留一法 (样本<50，推荐)"],
            index=0 if sample_count >= 50 else 2,
            help=cv_help,
        )

    # 解析交叉验证选项
    if "留一法" in cv_option:
        cv_method = "loocv"
        cv_folds = None
    elif "10折" in cv_option:
        cv_method = "kfold"
        cv_folds = 10
    else:  # 5折
        cv_method = "kfold"
        cv_folds = 5

    # ========== HPO 配置 ==========
    st.markdown("**🔧 超参数优化 (HPO) 配置**")

    col1, col2 = st.columns(2)
    with col1:
        hpo_method_label = st.selectbox(
            "HPO 方法",
            options=[
                "不使用 HPO（默认超参）",
                "Bayesian (Optuna TPE) — 推荐",
                "Random Search",
                "Hyperband",
                "Grid Search（穷举，小空间适用）",
            ],
            index=1,
            help=(
                "HPO 用内部交叉验证在训练集上搜索最优超参。\n"
                "- Bayesian: 智能搜索,推荐\n"
                "- Random Search: 简单基线\n"
                "- Hyperband: 资源高效,先粗后细\n"
                "- Grid Search: 穷举,搜索空间小时适用"
            ),
        )

    with col2:
        if hpo_method_label == "不使用 HPO（默认超参）":
            st.info("未启用 HPO,将使用各模型默认超参")
            n_trials = 30
        else:
            n_trials = st.slider(
                "优化轮数 (n_trials)",
                min_value=5,
                max_value=200,
                value=30,
                step=5,
                help=(
                    "HPO 试验次数,默认 30。\n"
                    "轮数越多越可能找到更优超参,但耗时更长。\n"
                    "Grid Search 会忽略此值,自动穷举整个搜索空间。"
                ),
            )

    # 解析 HPO 方法
    if "Bayesian" in hpo_method_label:
        hpo_method = "bayesian"
    elif "Random" in hpo_method_label:
        hpo_method = "random"
    elif "Hyperband" in hpo_method_label:
        hpo_method = "hyperband"
    elif "Grid" in hpo_method_label:
        hpo_method = "grid"
    else:
        hpo_method = None

    # 显示搜索空间(可折叠)
    with st.expander("📋 查看各模型搜索空间", expanded=False):
        from src.models import get_base_models

        for model_name in get_base_models().keys():
            st.markdown(f"**{model_name}**")
            space = get_search_space(model_name)
            for k, v in space.items():
                st.markdown(f"  - `{k}`: {v}")
            st.markdown("")

    # ========== 模型选择 ==========
    st.markdown("**☑ 选择要训练的模型**")
    st.caption("已检测到以下模型,取消勾选可跳过该模型(不训练、不显示)")
    base_models = get_base_models()
    model_checkboxes: dict[str, bool] = {}
    cols_per_row = 3
    model_names = list(base_models.keys())
    for i in range(0, len(model_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(model_names):
                mn = model_names[idx]
                with col:
                    model_checkboxes[mn] = st.checkbox(mn, value=True)
    selected_models = [mn for mn, checked in model_checkboxes.items() if checked]

    if not selected_models:
        st.warning("请至少选择一个模型")
        st.stop()

    # ========== 2.2 训练模型(按钮触发) ==========
    st.markdown("---")
    st.markdown('<a id="sec-2-2"></a>', unsafe_allow_html=True)
    st.subheader("2.2 训练模型")

    # 数据指纹: 检测是否换了数据或特征工程配置(换数据时清空旧训练结果)
    feat_info = st.session_state.get("feat_engineer_info", {})
    data_fingerprint = (
        f"{regression_data.original_shape}|"
        f"{regression_data.target_column}|"
        f"{regression_data.feature_columns[:3]}|"
        f"norm={feat_info.get('norm_method', '')}|"
        f"drop={feat_info.get('dropped', [])}"
    )
    if "training_data_fp" not in st.session_state:
        st.session_state.training_data_fp = ""

    if st.session_state.get("training_results") is not None and st.session_state.training_data_fp != data_fingerprint:
        st.session_state.training_results = None

    # 初始化训练状态
    if "training_results" not in st.session_state:
        st.session_state.training_results = None

    # 配置摘要
    col1, col2, col3 = st.columns(3)
    col1.metric("测试集比例", f"{test_size}")
    col2.metric("交叉验证", cv_option)
    col3.metric("HPO", hpo_method_label)

    # 按钮区:未训练→开始训练 / 已训练→重新训练(互斥)
    if st.session_state.training_results is None:
        run_btn = st.button("🚀 开始训练", type="primary", width="stretch")
    else:
        if st.button("🔄 重新训练", width="stretch"):
            st.session_state.training_results = None
            st.rerun()

    # 训练流程:只在没结果时执行
    if st.session_state.training_results is None:
        if run_btn:
            st.toast("🚀 训练已启动！训练完成后页面会自动更新，无需重复点击", icon="🚀")
            st.caption("⏳ 训练正在进行... 训练期间页面可能暂时无响应（白屏），这是正常现象，请等待训练完成后自动显示结果，不要刷新页面。")
            progress_placeholder = st.empty()

            def _progress_callback(msg: str, state: str = "running"):
                icon = {"running": "⏳", "complete": "✅", "error": "❌"}.get(state, "⏳")
                progress_placeholder.info(f"{icon} {msg}")

            _progress_callback(
                f"正在进行 {hpo_method_label}..."
                if hpo_method
                else "正在训练模型(默认超参)..."
            )
            results, metrics_df, hpo_results = train_and_evaluate_models(
                st.session_state.get("x_for_model", regression_data.x),
                regression_data.y,
                test_size=test_size,
                cv_method=cv_method,
                cv_folds=cv_folds if cv_folds else 5,
                hpo_method=hpo_method,
                n_trials=n_trials,
                selected_models=selected_models,
                progress_callback=_progress_callback,
            )
            st.session_state.training_results = (results, metrics_df, hpo_results)
            st.session_state.training_data_fp = data_fingerprint
            progress_placeholder.success("✅ 模型训练完成！")
        else:
            st.info(
                "👆 配置完成后点击「开始训练」按钮运行模型\n\n"
                "💡 如果上方配置修改时出现短暂白屏，是正常的页面刷新，请放心等待恢复。"
            )
            st.stop()

    # 这里开始一定有结果(首次训练完成或从 session_state 恢复)
    results, metrics_df, hpo_results = st.session_state.training_results

    # ========== 2.3 模型对比结果 ==========
    st.markdown("---")
    st.markdown('<a id="sec-2-3"></a>', unsafe_allow_html=True)
    st.subheader("2.3 模型对比结果")

    st.dataframe(metrics_df, width="stretch")

    # 获取最佳模型
    best_model_name = metrics_df.iloc[0]["model"]
    best_result = results[best_model_name]

    # 显示详细指标
    st.markdown("### 📊 最佳模型详细指标")
    st.markdown(f"**最佳模型：** `{best_model_name}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CV R² (交叉验证)", f"{best_result.cv_r2_mean:.3f}")
    col2.metric("CV 标准差", f"{best_result.cv_r2_std:.3f}")
    col3.metric("Test R² (测试集)", f"{best_result.test_r2:.3f}")
    col4.metric("Train R² (训练集)", f"{best_result.train_r2:.3f}")

    # ========== 2.3.1 HPO 最佳超参数展示 ==========
    if hpo_results:
        st.markdown("### 🎯 HPO 选出的最佳超参数")
        for model_name, hpo_res in hpo_results.items():
            with st.expander(
                f"`{model_name}` — 最佳 CV R² = {hpo_res['best_score']:.4f} "
                f"(共 {hpo_res['n_trials']} 轮)",
                expanded=(model_name == best_model_name),
            ):
                st.markdown(f"**方法:** `{hpo_res['method'].upper()}`")

                # 区分 HPO 调优 vs 其他默认
                tuned_params = hpo_res["best_params"]
                all_params = results[model_name].model.get_params()
                other_params = {
                    k: v for k, v in all_params.items() if k not in tuned_params
                }

                st.markdown(f"**🎯 HPO 调优的超参数:** {format_best_params(tuned_params)}")
                st.markdown(f"**📊 最佳 CV: R² = {hpo_res['best_score']:.4f}, RMSE = {hpo_res.get('best_rmse', 0.0):.4f}**")
                st.markdown("**📋 其他超参数（使用 sklearn 默认值）:**")
                import json as _json

                st.code(
                    _json.dumps(other_params, indent=2, ensure_ascii=False, default=str),
                    language="json",
                )
                st.markdown(f"**试验次数:** {hpo_res['n_trials']}")
    else:
        st.markdown("### 🎯 当前使用的超参数（默认,未启用 HPO）")
        for model_name in results.keys():
            model = results[model_name].model
            params = model.get_params()
            with st.expander(f"`{model_name}`", expanded=(model_name == best_model_name)):
                import json as _json

                st.code(
                    _json.dumps(params, indent=2, ensure_ascii=False, default=str),
                    language="json",
                )

    # ========== 2.4 可视化 ==========
    st.markdown("---")
    st.markdown('<a id="sec-2-4"></a>', unsafe_allow_html=True)
    st.subheader("2.4 可视化分析")

    # ---- 2.4.1 HPO 优化过程(放在最前面) ----
    if hpo_results:
        st.markdown("**🔧 HPO 优化过程**")
        st.caption(
            "展示每个模型的最佳 CV R² 随 trial 数的变化曲线,越往后曲线越平说明模型越收敛"
        )

        n_models = len(hpo_results)
        if n_models == 1:
            model_name = list(hpo_results.keys())[0]
            st.pyplot(plot_hpo_convergence(hpo_results[model_name]))
        else:
            model_names = list(hpo_results.keys())
            for i in range(0, n_models, 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < n_models:
                        mn = model_names[idx]
                        with col:
                            st.pyplot(plot_hpo_convergence(hpo_results[mn]))

    # ---- 预测结果(每个模型一张散点图,标注 R²/RMSE) ----
    st.markdown("**预测结果**")
    st.caption(
        "每个模型在训练集和测试集上的预测效果,橙色=训练集,深蓝=测试集,右下角标注双指标,★ 为最佳模型"
    )

    model_keys = list(results.keys())
    for i in range(0, len(model_keys), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(model_keys):
                mn = model_keys[idx]
                res = results[mn]
                with col:
                    st.pyplot(
                        plot_predicted_vs_actual(
                            res.y_test, res.test_pred,
                            res.y_train, res.train_pred,
                            title=mn,
                            r2_test=res.test_r2,
                            rmse_test=res.rmse,
                            r2_train=res.train_r2,
                            rmse_train=res.train_rmse,
                            is_best=(mn == best_model_name),
                        )
                    )

    # ========== 2.5 诊断建议 ==========
    st.markdown("---")
    st.markdown('<a id="sec-2-5"></a>', unsafe_allow_html=True)
    st.subheader("2.5 自动诊断建议")

    # 获取数据质量信息
    outlier_rows = int(st.session_state.get("_outlier_rows", 0))
    near_dup_rows = int(st.session_state.get("_near_dup_rows", 0))
    dup_rows = regression_data.duplicate_rows

    diagnosis = diagnose_level(
        sample_size=len(regression_data.model_data),
        feature_count=len(regression_data.feature_columns),
        test_r2=best_result.test_r2,
        train_r2=best_result.train_r2,
        cv_r2_mean=best_result.cv_r2_mean,
        cv_r2_std=best_result.cv_r2_std,
        duplicate_rows=dup_rows,
        outlier_rows=outlier_rows,
        near_dup_rows=near_dup_rows,
    )
    st.session_state.diagnosis = diagnosis

    # 等级展示（带颜色的大卡片）
    level = diagnosis["level"]
    label = diagnosis["label"]
    score = diagnosis["score"]
    level_emojis = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
    level_colors = {"A": "#1a7d36", "B": "#b8860b", "C": "#cc5500", "D": "#b30000"}

    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.markdown(
            f"<div style='text-align:center;padding:20px;border-radius:12px;"
            f"background:{level_colors[level]}15;border:2px solid {level_colors[level]}'>"
            f"<span style='font-size:48px'>{level_emojis[level]}</span><br>"
            f"<span style='font-size:36px;font-weight:bold;color:{level_colors[level]}'>{level}</span><br>"
            f"<span style='font-size:16px;color:{level_colors[level]}'>{label}</span><br>"
            f"<span style='font-size:14px;color:#666'>{score}/100</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(f"**{diagnosis['reason']}**")
        st.markdown("**影响因素：**")
        for detail in diagnosis["details"]:
            st.write(f"- {detail}")

    # 原有的详细指标
    st.markdown("")
    st.markdown(
        f"**交叉验证 R²：** `{best_result.cv_r2_mean:.3f} ± {best_result.cv_r2_std:.3f}` "
        f"({'✅ 稳定' if best_result.cv_r2_std < 0.08 else '⚠️ 波动较大'})"
    )
    st.markdown(
        f"**测试集 R²：** `{best_result.test_r2:.3f}` "
        f"{'✅ 效果好' if best_result.test_r2 >= 0.7 else '⚠️ 效果一般' if best_result.test_r2 >= 0.4 else '❌ 效果较差'}"
    )

    st.markdown("**详细诊断建议：**")
    suggestions = generate_diagnostics(
        sample_size=len(regression_data.model_data),
        feature_count=len(regression_data.feature_columns),
        test_r2=best_result.test_r2,
        train_r2=best_result.train_r2,
        cv_r2_mean=best_result.cv_r2_mean,
        cv_r2_std=best_result.cv_r2_std,
    )
    for suggestion in suggestions:
        st.write(f"- {suggestion}")

    # ========== 2.6 报告下载 ==========
    st.markdown("---")
    st.markdown('<a id="sec-2-6"></a>', unsafe_allow_html=True)
    st.divider()

    # 为 HTML 报告生成嵌入图表(base64)
    from io import BytesIO
    import base64

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
        import matplotlib.pyplot as _plt
        _plt.close(fig)

        if hpo_results and best_model_name in hpo_results:
            fig = plot_hpo_convergence(hpo_results[best_model_name])
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            convergence_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            _plt.close(fig)
    except Exception:
        pass  # 图表嵌入失败不影响报告下载

    # 生成 TXT 报告
    report_text = generate_report_text(
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
        all_params_by_model={
            mn: results[mn].model.get_params() for mn in results
        },
        diagnosis=st.session_state.get("diagnosis"),
    )

    # 生成 HTML 报告(含图表)
    report_html = generate_html_report(
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
        all_params_by_model={
            mn: results[mn].model.get_params() for mn in results
        },
        metrics_df=metrics_df,
        scatter_plot_base64=scatter_b64,
        convergence_plot_base64=convergence_b64,
        diagnosis=st.session_state.get("diagnosis"),
        data_quality={
            "duplicate_rows": regression_data.duplicate_rows,
            "outlier_rows": st.session_state.get("_outlier_rows", 0),
            "near_dup_rows": st.session_state.get("_near_dup_rows", 0),
            "dataset": target_column,
            "fingerprint": data_fingerprint[:40],
        },
        generated_date=__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # 生成 PDF 报告
    try:
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
            diagnosis=st.session_state.get("diagnosis"),
            data_quality={
                "duplicate_rows": regression_data.duplicate_rows,
                "outlier_rows": st.session_state.get("_outlier_rows", 0),
                "near_dup_rows": st.session_state.get("_near_dup_rows", 0),
                "dataset": target_column,
            },
        )
    except Exception as _pdf_err:
        pdf_bytes = None
        import traceback
        st.caption(f"📕 PDF 导出失败：{_pdf_err}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 下载诊断报告 (TXT)",
            data=report_text.encode("utf-8"),
            file_name="materials_ml_report.txt",
            mime="text/plain",
            width="stretch",
        )
    with col2:
        st.download_button(
            "📊 下载诊断报告 (HTML)",
            data=report_html.encode("utf-8"),
            file_name="materials_ml_report.html",
            mime="text/html",
            width="stretch",
        )
    with col3:
        if pdf_bytes:
            st.download_button(
                "📕 下载诊断报告 (PDF)",
                data=pdf_bytes,
                file_name="materials_ml_report.pdf",
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.caption("📕 PDF 导出不可用")
    st.success("✅ Step 2 完成！模型训练和评估已完成。")

    st.success("✅ Step 2 完成！模型训练和评估已完成。")

# ========================================================================
# Step 3: 可解释性分析 (SHAP + PDP)
# ========================================================================
with st.expander("🔍 Step 3: 可解释性分析", expanded=(st.session_state.current_step == "Step 3")):
    st.subheader("模型可解释性分析")

    # 检查是否已完成训练
    if st.session_state.get("training_results") is None:
        st.info("⚠️ 请先在 Step 2 中完成模型训练后再进行可解释性分析。")
        st.stop()

    results, metrics_df, hpo_results = st.session_state.training_results
    best_model_name = metrics_df.iloc[0]["model"]

    # ========== 模型选择（更醒目） ==========
    model_names = list(results.keys())
    selected_model_name = st.selectbox(
        "选择要分析的模型",
        model_names,
        index=model_names.index(best_model_name),
        help="默认为最佳模型(CV R² 最高)，你也可以选择其他模型进行分析",
    )
    selected_model = results[selected_model_name]
    st.info(
        f"当前分析模型: **{selected_model_name}**  "
        f"(Test R² = {selected_model.test_r2:.3f}, "
        f"CV R² = {selected_model.cv_r2_mean:.3f} ± {selected_model.cv_r2_std:.3f})"
    )

    # 使用模型实际训练用的特征列(可能经过特征工程筛选)
    feature_columns = selected_model.x_train.columns.tolist() if selected_model.x_train is not None else regression_data.feature_columns
    x_train = selected_model.x_train
    x_test = selected_model.x_test
    model_obj = selected_model.model

    # ========== SHAP 分析 ==========
    st.markdown("---")
    st.markdown('<a id="sec-3-1"></a>', unsafe_allow_html=True)
    st.markdown("### 3.1 🔬 SHAP 可解释性分析")
    st.caption(
        "SHAP (SHapley Additive exPlanations) 基于博弈论，"
        "量化每个特征对预测结果的贡献。红色=特征值高，蓝色=特征值低。"
    )

    shap_check_container = st.empty()

    with st.spinner("⏳ 正在计算 SHAP 值（可能需要几秒钟）..."):
        # 合并训练集和测试集，保证样本量足够密实（SHAP 不依赖标签）
        x_pool = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
        shap_values, expected_value, x_pool_sample = compute_shap_values(
            model=model_obj,
            x_train=x_train,
            x_test=x_pool,
            n_samples=500,
        )

    if shap_values is None:
        shap_check_container.warning(
            "⚠️ SHAP 分析当前版本暂不适配此模型类型，跳过。"
        )
    else:
        shap_check_container.success(f"✅ SHAP 值计算完成（基于 {len(x_pool_sample)} 条样本）")

        # SHAP 摘要图
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 SHAP 摘要图 (Beeswarm)**")
            st.caption("反映特征重要性和影响方向")
            fig_shap = plot_shap_summary(
                shap_values, x_pool_sample, feature_columns, max_display=15
            )
            if fig_shap:
                st.pyplot(fig_shap)
            else:
                st.info("SHAP 摘要图生成失败")

        with col2:
            st.markdown('<a id="sec-3-2"></a>', unsafe_allow_html=True)
            st.markdown("**📊 SHAP 特征重要性**")
            st.caption("按 mean(|SHAP|) 排序的全局特征重要性")
            fig_bar = plot_shap_bar(
                shap_values, x_pool_sample, feature_columns, max_display=15
            )
            if fig_bar:
                st.pyplot(fig_bar)
            else:
                st.info("SHAP 柱状图生成失败")

        # 补充说明
        with st.expander("📖 如何理解 SHAP 图？", expanded=False):
            st.markdown(
                """
            **SHAP 摘要图 (Beeswarm):**
            - 每个点代表一个样本的某个特征
            - **横轴位置** = SHAP 值，即该特征对当前预测的"贡献"（向右=增大预测值，向左=减小）
            - **颜色** = 特征值高低（红色=高，蓝色=低）
            - 如果红色集中在右侧、蓝色在左侧 → 该特征值越大，预测值越大（正相关）
            - 如果红色集中在左侧、蓝色在右侧 → 该特征值越大，预测值越小（负相关）

            **SHAP 柱状图:**
            - 柱长 = mean(|SHAP|)，即该特征的平均影响强度
            - 从大到小排序，越靠上的特征对模型越重要

            **关键点:**
            - SHAP 解释的是模型行为，不是因果关系
            - 特征重要性高 ≠ 该特征在真实世界中一定重要（模型可能学到了伪相关）
            """
            )

    # ========== PDP 偏依赖图 ==========
    st.markdown("---")
    st.markdown('<a id="sec-3-3"></a>', unsafe_allow_html=True)
    st.markdown("### 3.2 📈 偏依赖图 (PDP)")
    st.caption(
        "展示单个特征变化时，模型平均预测值如何变化。"
        "帮助理解特征与目标之间的函数关系（线性、非线性、单调性等）。"
    )

    n_features = len(feature_columns)
    top_n_pdp = min(6, n_features)
    st.caption(f"展示前 {top_n_pdp} 个特征，如需分析特定特征可切换模型")

    with st.spinner("⏳ 正在生成偏依赖图..."):
        fig_pdp = plot_partial_dependence(
            model=model_obj,
            x_train=x_train,
            feature_names=feature_columns,
            top_n=top_n_pdp,
            figsize=(12, 8),
        )

    if fig_pdp:
        st.pyplot(fig_pdp)
    else:
        st.info("PDP 图生成失败，可能是模型或数据不兼容")

    # PDP + ICE 曲线（可选，折叠）
    st.markdown('<a id="sec-3-4"></a>', unsafe_allow_html=True)
    with st.expander("📊 3.3 ICE 曲线（个体条件期望）", expanded=True):
        st.caption(
            "ICE 曲线展示每个样本的预测随特征变化的个体趋势（灰色细线），"
            "PDP（红色粗线）是所有 ICE 的平均。ICE 越多越分散，说明特征交互越强。"
        )
        with st.spinner("⏳ 正在计算 ICE 曲线..."):
            fig_ice = plot_pdp_with_individual(
                model=model_obj,
                x_train=x_train,
                feature_names=feature_columns,
                top_n=top_n_pdp,
                figsize=(12, 8),
            )
        if fig_ice:
            st.pyplot(fig_ice)
        else:
            st.info("ICE 曲线生成失败")

    # ========== 总结 ==========
    st.markdown("---")
    st.success(
        f"✅ Step 3 完成！已对 `{selected_model_name}` 进行 SHAP 和 PDP 可解释性分析。"
    )
