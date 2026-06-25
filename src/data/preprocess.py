from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegressionData:
    """回归建模所需的数据结构"""

    model_data: pd.DataFrame  # 清洗后的完整数据
    x: pd.DataFrame  # 特征数据
    y: pd.Series  # 目标数据
    feature_columns: list[str]  # 特征列名
    missing_rate: pd.Series  # 缺失值比例
    removed_missing_rows: int  # 删除的缺失行数
    duplicate_rows: int  # 重复行数
    original_shape: tuple  # 原始数据形状
    target_column: str  # 目标列名
    encoded_categorical_info: dict  # 类别特征编码信息 {原列名: 类别数}


def prepare_regression_data(
    data: pd.DataFrame, target_column: str
) -> RegressionData:
    """
    准备回归建模数据（支持类别特征自动独热编码）。

    处理流程：
    1. 检测并自动编码类别型列（类别数≤10）
    2. 筛选数值型列
    3. 合并数值列和编码后的类别列
    4. 检查目标列是否为数值型
    5. 删除缺失样本
    6. 统计数据质量信息

    Args:
        data: 原始数据
        target_column: 目标列名

    Returns:
        RegressionData: 包含建模数据和质量信息的数据类

    Raises:
        ValueError: 目标列非数值型、特征列不足、样本量过少等错误
    """
    original_shape = data.shape
    encoded_categorical_info = {}

    # ========== 1. 处理类别型特征 ==========
    # 检测类别型列（object 和 category 类型）
    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # 排除目标列（如果目标列是类别型，后面会报错）
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)

    # 对类别特征进行独热编码
    encoded_dfs = []
    for col in categorical_columns:
        n_unique = data[col].nunique()

        if n_unique <= 10:
            # 自动独热编码
            encoded = pd.get_dummies(data[col], prefix=col, drop_first=False)
            encoded_dfs.append(encoded)
            encoded_categorical_info[col] = n_unique
        else:
            # 类别数过多，警告并跳过
            # 这里不抛出错误，只是记录警告，后续可以在界面提示
            pass

    # ========== 2. 筛选数值型列 ==========
    numeric_data = data.select_dtypes(include="number").copy()

    # ========== 3. 合并数值列和编码后的类别列 ==========
    if encoded_dfs:
        # 重置索引确保对齐
        numeric_data = numeric_data.reset_index(drop=True)
        for encoded_df in encoded_dfs:
            encoded_df = encoded_df.reset_index(drop=True)
            numeric_data = pd.concat([numeric_data, encoded_df], axis=1)

    # ========== 4. 检查目标列 ==========
    if target_column not in numeric_data.columns:
        raise ValueError(f"目标列 '{target_column}' 不是数值型，当前 Demo 只支持数值型目标列。")

    # 获取特征列
    feature_columns = [col for col in numeric_data.columns if col != target_column]

    if len(feature_columns) < 1:
        raise ValueError("至少需要 1 个数值特征列。")

    # ========== 5. 统计缺失值和重复行 ==========
    missing_rate = data.isna().mean().sort_values(ascending=False)
    duplicate_rows = int(data.duplicated().sum())

    # ========== 6. 删除缺失样本 ==========
    model_data = numeric_data[feature_columns + [target_column]].dropna()
    removed_missing_rows = len(numeric_data) - len(model_data)

    if len(model_data) < 10:
        raise ValueError(
            f"清洗缺失值后样本数仅 {len(model_data)} 条，少于 10，暂不建议建模。"
        )

    # 分离特征和目标
    x = model_data[feature_columns]
    y = model_data[target_column]

    return RegressionData(
        model_data=model_data,
        x=x,
        y=y,
        feature_columns=feature_columns,
        missing_rate=missing_rate,
        removed_missing_rows=removed_missing_rows,
        duplicate_rows=duplicate_rows,
        original_shape=original_shape,
        target_column=target_column,
        encoded_categorical_info=encoded_categorical_info,
    )


def normalize_features(
    x: pd.DataFrame, method: str = "zscore"
) -> tuple[pd.DataFrame, object | None]:
    """
    对数值特征进行归一化/标准化。

    Args:
        x: 特征 DataFrame
        method: "zscore"(默认,StandardScaler) / "minmax"(MinMaxScaler) /
                "robust"(RobustScaler) / "none"(跳过)

    Returns:
        (x_normalized, scaler_object)
    """
    if method is None or method == "none":
        return x, None

    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scaler_map = {
        "zscore": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    scaler = scaler_map.get(method)
    if scaler is None:
        return x, None

    x_norm = pd.DataFrame(
        scaler.fit_transform(x), columns=x.columns, index=x.index
    )
    return x_norm, scaler


def filter_features(
    x: pd.DataFrame,
    var_threshold: float = 0.0,
    corr_threshold: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    筛选特征：低方差过滤 + 高相关过滤。

    流程: 先剔除低于 var_threshold 的低方差特征,
          再剔除相关系数高于 corr_threshold 的冗余特征(保留第一个)。

    Args:
        x: 特征 DataFrame
        var_threshold: 方差阈值,<=0 时不进行方差过滤
        corr_threshold: 相关性阈值,>=1.0 时不进行相关过滤

    Returns:
        (x_filtered, info_dict)
        info_dict = {
            "kept_columns": [...],
            "dropped_low_variance": [...],
            "dropped_high_corr": [...],
        }
    """
    info: dict = {"kept_columns": [], "dropped_low_variance": [], "dropped_high_corr": []}
    x_out = x.copy()

    # 1. 方差过滤
    if var_threshold > 0:
        variances = x_out.var()
        low_var_cols = variances[variances < var_threshold].index.tolist()
        if low_var_cols:
            x_out = x_out.drop(columns=low_var_cols)
            info["dropped_low_variance"] = low_var_cols

    # 2. 高相关过滤
    if corr_threshold < 1.0 and x_out.shape[1] > 1:
        corr_matrix = x_out.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_cols = []
        for col in upper_tri.columns:
            if any(upper_tri[col] > corr_threshold):
                high_corr_cols.append(col)
        if high_corr_cols:
            x_out = x_out.drop(columns=high_corr_cols)
            info["dropped_high_corr"] = high_corr_cols

    info["kept_columns"] = x_out.columns.tolist()
    return x_out, info


def detect_outliers_iqr(
    data: pd.DataFrame, factor: float = 1.5
) -> tuple[dict, pd.Series]:
    """
    用 IQR 方法检测数值列中的异常值。

    对每列计算 Q1、Q3、IQR，超出 [Q1 - factor×IQR, Q3 + factor×IQR] 的记为异常。
    factor=1.5 为"适度异常"，factor=3 为"极端异常"。

    Args:
        data: 输入数据
        factor: IQR 倍数，默认 1.5

    Returns:
        (outlier_info, combined_mask)
        outlier_info = {
            "列名": {
                "count": 异常数,
                "lower": 下界,
                "upper": 上界,
                "indices": [异常行索引],
            }
        }
        combined_mask: bool Series，所有列合并后只要有任一列异常即为 True
    """
    numeric = data.select_dtypes(include="number")
    outlier_info = {}
    combined_mask = pd.Series(False, index=data.index)

    for col in numeric.columns:
        vals = numeric[col].dropna()
        if len(vals) < 4:
            continue  # 样本太少无法判断
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue  # 常数列跳过
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        col_mask = (vals < lower) | (vals > upper)
        n = int(col_mask.sum())
        if n > 0:
            outlier_info[col] = {
                "count": n,
                "lower": lower,
                "upper": upper,
                "indices": col_mask.index[col_mask].tolist(),
            }
            combined_mask = combined_mask | col_mask

    return outlier_info, combined_mask


def detect_near_duplicates(
    data: pd.DataFrame, numerical_columns: list[str], decimals: int = 2
) -> dict:
    """
    检测近重复样本：对数值列四舍五入到指定位数后，
    标记哪些行彼此成为"几乎重复"。

    Args:
        data: 输入数据
        numerical_columns: 用于比对的数值列名列表
        decimals: 四舍五入位数，默认 2 位小数

    Returns:
        {
            "groups": [{"rows": [行索引], "count": 行数, "values": {列名: 值}}, ...],
            "total_near_dup": 参与近重复的总行数,
        }
    """
    if not numerical_columns:
        return {"groups": [], "total_near_dup": 0}

    rounded = data[numerical_columns].round(decimals)
    dup_mask = rounded.duplicated(keep=False)
    total_near_dup = int(dup_mask.sum())

    if total_near_dup == 0:
        return {"groups": [], "total_near_dup": 0}

    # 按四舍五入后的值分组
    grouped = rounded[dup_mask].groupby(
        [c for c in rounded.columns], sort=False
    )
    groups = []
    for key, group in grouped:
        rows = group.index.tolist()
        if isinstance(key, tuple):
            vals = dict(zip(rounded.columns, key))
        else:
            vals = {rounded.columns[0]: key}
        groups.append({"rows": rows, "count": len(rows), "values": vals})

    groups.sort(key=lambda g: g["count"], reverse=True)
    return {"groups": groups, "total_near_dup": total_near_dup}
