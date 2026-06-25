"""
超参数优化(Hyperparameter Optimization, HPO)模块。

支持 4 种 HPO 算法:
- Bayesian (TPE) via Optuna
- Random Search via sklearn
- Hyperband (HalvingRandomSearchCV) via sklearn
- Grid Search via sklearn

所有方法返回统一格式的字典,便于上层调用和绘图。
"""

import importlib
import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  启用 HalvingRandomSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV

# optuna 是可选依赖(仅 Bayesian 方法需要),改成懒加载避免硬依赖
# 在 run_bayesian_hpo 中按需 import


def _is_module_available(module_name: str) -> bool:
    """检查 Python 包是否已安装(轻量检测,不导入命名空间)。"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_base_models() -> dict:
    """返回基础模型字典(自动检测已安装的包,未装则不包含)。"""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    # LightGBM
    if _is_module_available("lightgbm"):
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMRegressor(random_state=42, verbose=-1)

    # XGBoost
    if _is_module_available("xgboost"):
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBRegressor(random_state=42, verbosity=0)

    # CatBoost
    if _is_module_available("catboost"):
        import catboost as cb

        models["CatBoost"] = cb.CatBoostRegressor(random_state=42, verbose=0)

    return models


# 各模型的搜索空间(共享给所有 HPO 方法)
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "Linear Regression": {
        "fit_intercept": [True, False],
        "positive": [True, False],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.8, 0.9, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    "CatBoost": {
        "iterations": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
    },
}


def _model_supports_random_state(model_name: str) -> bool:
    """检查某模型是否支持 random_state 参数。"""
    cls = type(get_base_models()[model_name])
    return "random_state" in inspect.signature(cls.__init__).parameters


def _instantiate_model(model_name: str, params: dict, random_state: int = 42):
    """根据模型名 + 参数字典创建实例,自动处理 random_state。"""
    cls = type(get_base_models()[model_name])
    init_params: dict[str, Any] = dict(params)
    if _model_supports_random_state(model_name):
        init_params["random_state"] = random_state
    return cls(**init_params)


def _eval_cv_stats(model, x, y, cv) -> tuple[float, float]:
    """在训练集上做 K 折交叉验证,返回 (平均 R², 平均 RMSE)。"""
    from sklearn.model_selection import cross_validate

    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
    cv_out = cross_validate(model, x, y, cv=cv, scoring=scoring)
    r2_mean = float(cv_out["test_r2"].mean())
    rmse_mean = float((-cv_out["test_rmse"]).mean())
    return r2_mean, rmse_mean


def _make_history(
    trials: list[tuple[int, float, float, dict]],
) -> list[dict]:
    """统一 history 格式: 每项 {trial, score(R²), rmse, params}。"""
    return [
        {"trial": t, "score": r2, "rmse": rmse, "params": p}
        for t, r2, rmse, p in trials
    ]


def run_bayesian_hpo(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    cv: Any = 5,
    random_state: int = 42,
) -> dict:
    """Optuna TPE 贝叶斯优化。"""
    # 懒加载 optuna
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "使用 Bayesian HPO 需要安装 optuna,请运行: pip install optuna"
        ) from e

    # 静默 Optuna 日志,避免刷屏
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = SEARCH_SPACES[model_name]

    def objective(trial: "optuna.Trial") -> float:
        params = {
            name: trial.suggest_categorical(name, candidates)
            for name, candidates in space.items()
        }
        try:
            model = _instantiate_model(model_name, params, random_state)
            r2, rmse = _eval_cv_stats(model, x, y, cv)
            trial.set_user_attr("rmse", rmse)
            return r2
        except Exception:
            return -float("inf")  # 该 trial 失败，不干扰其他 trial

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_log = [
        (t.number, float(t.value), float(t.user_attrs.get("rmse", 0.0)), t.params)
        for t in study.trials
        if t.value is not None
    ]

    # 如果没有任何 trial 成功完成，返回默认参数
    if not trials_log:
        return {
            "method": "bayesian",
            "model_name": model_name,
            "best_params": {},
            "best_score": -float("inf"),
            "best_rmse": 0.0,
            "history": [],
            "n_trials": 0,
        }

    best_trials = [
        t for t in study.trials
        if t.value is not None and t.number == study.best_trial.number
    ]
    best_rmse = float(best_trials[0].user_attrs.get("rmse", 0.0)) if best_trials else 0.0

    return {
        "method": "bayesian",
        "model_name": model_name,
        "best_params": dict(study.best_params),
        "best_score": float(study.best_value),
        "best_rmse": best_rmse,
        "history": _make_history(trials_log),
        "n_trials": len(trials_log),
    }


def run_random_search_hpo(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    cv: Any = 5,
    random_state: int = 42,
) -> dict:
    """sklearn RandomizedSearchCV 随机搜索。"""
    space = SEARCH_SPACES[model_name]
    base = get_base_models()[model_name]
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            base,
            space,
            n_iter=n_trials,
            cv=cv,
            scoring=scoring,
            refit="r2",
            random_state=random_state,
            n_jobs=1,
        )
        search.fit(x, y)

    trials_log = [
        (i, float(r2), float(-rmse), params)
        for i, (params, r2, rmse) in enumerate(
            zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_r2"],
                search.cv_results_["mean_test_rmse"],
            )
        )
    ]

    best_idx = int(search.best_index_)
    best_rmse = float(-search.cv_results_["mean_test_rmse"][best_idx])

    return {
        "method": "random",
        "model_name": model_name,
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "best_rmse": best_rmse,
        "history": _make_history(trials_log),
        "n_trials": len(trials_log),
    }


def run_hyperband_hpo(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    cv: Any = 5,
    random_state: int = 42,
) -> dict:
    """sklearn HalvingRandomSearchCV 资源感知的 Hyperband。
    注: HalvingRandomSearchCV 不支持多指标 dict scoring,用单指标 r2 + 事后算 RMSE。"""
    space = SEARCH_SPACES[model_name]
    base = get_base_models()[model_name]
    n_candidates = max(n_trials, 20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = HalvingRandomSearchCV(
            base,
            space,
            n_candidates=n_candidates,
            cv=cv,
            scoring="r2",
            random_state=random_state,
            n_jobs=1,
            refit=True,
            min_resources="exhaust",
        )
        search.fit(x, y)

    # Hyperband 只记录 R²,回头单独算 RMSE
    trials_log = [
        (i, float(score), 0.0, params)
        for i, (params, score) in enumerate(
            zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
        )
    ]

    # 事后算最佳模型的 RMSE
    _, best_rmse = _eval_cv_stats(search.best_estimator_, x, y, cv)

    return {
        "method": "hyperband",
        "model_name": model_name,
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "best_rmse": best_rmse,
        "history": _make_history(trials_log),
        "n_trials": len(trials_log),
    }


def run_grid_hpo(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    cv: Any = 5,
    random_state: int = 42,
) -> dict:
    """sklearn GridSearchCV 穷举搜索。n_trials 在 grid 中无效,仅用于返回信息。"""
    space = SEARCH_SPACES[model_name]
    base = get_base_models()[model_name]
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = GridSearchCV(
            base,
            space,
            cv=cv,
            scoring=scoring,
            refit="r2",
            n_jobs=1,
        )
        search.fit(x, y)

    trials_log = [
        (i, float(r2), float(-rmse), params)
        for i, (params, r2, rmse) in enumerate(
            zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_r2"],
                search.cv_results_["mean_test_rmse"],
            )
        )
    ]

    best_idx = int(search.best_index_)
    best_rmse = float(-search.cv_results_["mean_test_rmse"][best_idx])

    return {
        "method": "grid",
        "model_name": model_name,
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "best_rmse": best_rmse,
        "history": _make_history(trials_log),
        "n_trials": len(trials_log),
    }


# 方法名 -> 函数的映射
HPO_METHODS: dict[str, callable] = {
    "bayesian": run_bayesian_hpo,
    "random": run_random_search_hpo,
    "hyperband": run_hyperband_hpo,
    "grid": run_grid_hpo,
}


def run_hpo(
    method: str,
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    cv: Any = 5,
    random_state: int = 42,
) -> dict:
    """
    统一 HPO 入口。

    Args:
        method: "bayesian" / "random" / "hyperband" / "grid"
        model_name: 模型名(在 get_base_models() 中)
        x, y: 训练数据
        n_trials: 试验次数(grid 会忽略,使用 param_grid 穷举)
        cv: 交叉验证折数或 splitter
        random_state: 随机种子

    Returns:
        dict: {method, model_name, best_params, best_score, best_rmse, history, n_trials}
    """
    if method not in HPO_METHODS:
        raise ValueError(
            f"未知 HPO 方法: {method}。可选: {list(HPO_METHODS.keys())}"
        )
    if model_name not in get_base_models():
        raise ValueError(
            f"未知模型: {model_name}。可选: {list(get_base_models().keys())}"
        )
    return HPO_METHODS[method](
        model_name=model_name,
        x=x,
        y=y,
        n_trials=n_trials,
        cv=cv,
        random_state=random_state,
    )


def get_search_space(model_name: str) -> dict:
    """获取指定模型的搜索空间(供 UI 展示)。"""
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"未知模型: {model_name}")
    return SEARCH_SPACES[model_name]
