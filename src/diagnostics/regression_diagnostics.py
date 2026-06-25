def generate_diagnostics(
    sample_size: int,
    feature_count: int,
    test_r2: float,
    train_r2: float,
    cv_r2_mean: float,
    cv_r2_std: float,
) -> list[str]:
    """
    基于规则生成诊断建议（含交叉验证分析）。

    不调用大模型，只基于真实计算指标输出风险提示。

    Args:
        sample_size: 有效建模样本数
        feature_count: 特征数量
        test_r2: 测试集 R²
        train_r2: 训练集 R²
        cv_r2_mean: 交叉验证 R² 均值
        cv_r2_std: 交叉验证 R² 标准差

    Returns:
        list[str]: 诊断建议列表
    """
    suggestions = []

    # 样本量检查
    if sample_size < 50:
        suggestions.append("样本量偏少，建议谨慎解释模型结果，优先作为初步探索。")

    # 特征数量检查
    if feature_count > sample_size / 3:
        suggestions.append("特征数量相对样本量偏多，存在过拟合风险，建议做特征筛选。")

    # 交叉验证稳定性检查
    if cv_r2_std > 0.15:
        suggestions.append(
            f"交叉验证标准差较大 (std={cv_r2_std:.3f})，模型不稳定，可能对数据划分敏感。"
        )
    elif cv_r2_std > 0.08:
        suggestions.append(
            f"交叉验证标准差适中 (std={cv_r2_std:.3f})，模型稳定性一般。"
        )
    else:
        suggestions.append(
            f"交叉验证标准差很小 (std={cv_r2_std:.3f})，模型稳定性好。"
        )

    # 交叉验证与测试集一致性检查
    cv_test_gap = abs(cv_r2_mean - test_r2)
    if cv_test_gap > 0.15:
        suggestions.append(
            f"交叉验证 R²({cv_r2_mean:.3f}) 与测试集 R²({test_r2:.3f}) 差距较大，"
            "可能存在数据划分问题或样本分布不均。"
        )

    # 模型表现检查
    if test_r2 < 0.3:
        suggestions.append("当前测试集 R² 偏低，可能需要补充关键材料特征或增加样本。")
    elif test_r2 < 0.6:
        suggestions.append("当前模型有一定预测能力，但建议结合材料机理分析进一步验证。")
    else:
        suggestions.append("当前模型表现较好，但仍需检查数据泄漏、样本划分和实验重复性。")

    # 过拟合检查
    train_test_gap = train_r2 - test_r2
    if train_test_gap > 0.3:
        suggestions.append(
            f"训练集与测试集 R² 差距较大 (gap={train_test_gap:.3f})，可能存在过拟合。"
        )
    elif train_test_gap > 0.15:
        suggestions.append(
            f"训练集与测试集 R² 有一定差距 (gap={train_test_gap:.3f})，需注意过拟合风险。"
        )

    # 通用提示
    suggestions.append("本工具仅用于初步分析，不保证论文发表，不能证明因果关系。")

    return suggestions


def diagnose_level(
    sample_size: int,
    feature_count: int,
    test_r2: float,
    train_r2: float,
    cv_r2_mean: float,
    cv_r2_std: float,
    duplicate_rows: int = 0,
    outlier_rows: int = 0,
    near_dup_rows: int = 0,
) -> dict:
    """
    综合判断数据建模可行性的诊断等级。

    等级说明:
        A (适合建模) — 数据质量好，模型表现稳定，可进一步优化和解释
        B (可探索)   — 有一定信号，但需增加样本或优化特征
        C (高风险)   — 模型不稳定或数据质量存在问题
        D (不建议)   — 样本太少、噪声过大或目标变量规律不明显

    Args:
        sample_size: 有效建模样本数
        feature_count: 特征数量
        test_r2: 测试集 R²
        train_r2: 训练集 R²
        cv_r2_mean: 交叉验证 R² 均值
        cv_r2_std: 交叉验证 R² 标准差
        duplicate_rows: 重复行数
        outlier_rows: 含异常值的行数
        near_dup_rows: 近重复行数

    Returns:
        dict: {level, label, score, reason, details}
    """
    positives: list[str] = []
    warnings_list: list[str] = []
    score = 100

    # ========== 样本量 × 特征数 复合判断 ==========
    ratio = sample_size / max(feature_count, 1)  # 每特征样本数

    if ratio >= 20:
        positives.append(f"样本/特征比充足 ({sample_size}/{feature_count} ≈ {ratio:.0f})")
    elif ratio >= 10:
        warnings_list.append(f"样本/特征比一般 ({sample_size}/{feature_count} ≈ {ratio:.0f})")
        score -= 10
    elif ratio >= 5:
        warnings_list.append(f"样本/特征比偏低 ({sample_size}/{feature_count} ≈ {ratio:.0f})")
        score -= 20
    elif ratio >= 2:
        warnings_list.append(f"样本/特征比过低 ({sample_size}/{feature_count} ≈ {ratio:.0f})")
        score -= 30
    else:
        warnings_list.append(f"样本/特征比极低 ({sample_size}/{feature_count} ≈ {ratio:.0f})")
        score -= 40

    # 绝对样本量（即使比例不错，总量太少也不行）
    if sample_size >= 200:
        pass  # 已在比例中体现
    elif sample_size >= 50:
        pass  # 已在比例中体现
    if sample_size < 30:
        warnings_list.append(f"绝对样本量偏少 ({sample_size})")
        score -= 5

    # ========== 模型表现（复合判断） ==========
    # 有效估值 = min(CV下界, Test R²) 综合反映预测能力+稳定性+泛化
    cv_lower = cv_r2_mean - cv_r2_std
    composite_r2 = min(cv_lower, test_r2)

    if composite_r2 >= 0.6:
        positives.append(f"模型表现较好 (有效估值={composite_r2:.3f})")
    elif composite_r2 >= 0.3:
        warnings_list.append(f"模型有一定预测能力 (有效估值={composite_r2:.3f})")
        score -= 15
    elif composite_r2 >= 0.08:
        warnings_list.append(f"模型预测能力较弱 (有效估值={composite_r2:.3f})")
        score -= 25
    else:
        warnings_list.append(f"模型基本无预测能力 (有效估值={composite_r2:.3f})")
        score -= 40

    # ========== 过拟合检查 ==========
    train_test_gap = train_r2 - test_r2
    if train_test_gap > 0.5:
        warnings_list.append(f"严重过拟合 (Train/Test 差距={train_test_gap:.3f})")
        score -= 30
    elif train_test_gap > 0.3:
        warnings_list.append(f"过拟合风险高 (Train/Test 差距={train_test_gap:.3f})")
        score -= 25
    elif train_test_gap > 0.15:
        warnings_list.append(f"存在过拟合迹象 (差距={train_test_gap:.3f})")
        score -= 10

    # ========== 数据质量 ==========
    if duplicate_rows > 0:
        warnings_list.append(f"存在 {duplicate_rows} 行完全重复样本")
        score -= 5
    if near_dup_rows > 0:
        warnings_list.append(f"存在 {near_dup_rows} 行近重复样本")
        score -= 5
    if outlier_rows > 0:
        pct = outlier_rows / max(sample_size, 1) * 100
        if pct > 10:
            warnings_list.append(f"异常值比例偏高 ({pct:.0f}%)")
            score -= 10
        else:
            score -= 3

    score = max(0, min(100, score))

    # ========== 等级判定（硬条件 + 分数综合） ==========
    hard_d = sample_size < 10 or composite_r2 < 0.05 or ratio < 1.5
    hard_c = sample_size < 30 or composite_r2 < 0.15 or ratio < 3

    if hard_d:
        level, label = "D", "不建议"
    elif hard_c:
        level, label = "C", "高风险"
    elif score >= 75 and composite_r2 >= 0.6 and train_test_gap < 0.2:
        level, label = "A", "适合建模"
    elif score >= 50:
        level, label = "B", "可探索"
    elif score >= 20:
        level, label = "C", "高风险"
    else:
        level, label = "D", "不建议"

    # 组装原因
    all_details = [f"✅ {p}" for p in positives] + [f"⚠️ {w}" for w in warnings_list]
    summary = _build_summary(level, label, score, positives, warnings_list)

    return {
        "level": level,
        "label": label,
        "score": score,
        "reason": summary,
        "details": all_details,
        "positives": positives,
        "warnings": warnings_list,
    }


def _build_summary(
    level: str, label: str, score: int,
    positives: list[str], warnings_list: list[str],
) -> str:
    """生成一句话总结。"""
    summaries = {
        "A": f"诊断等级 A（适合建模），评分 {score}/100。数据质量较好，模型表现稳定，可继续做进一步解释和优化。",
        "B": f"诊断等级 B（可探索），评分 {score}/100。数据有一定可建模信号，但需关注上述风险点。",
        "C": f"诊断等级 C（高风险），评分 {score}/100。当前数据存在较多风险，模型结果不可靠。",
        "D": f"诊断等级 D（不建议），评分 {score}/100。当前数据不适合做机器学习建模。",
    }
    return summaries.get(level, "")
