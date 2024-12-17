import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from typing import Callable, Dict, Any, Tuple

# 禁用 Optuna 日誌輸出
optuna.logging.set_verbosity(optuna.logging.WARN)

def hyperparameter_optimization(
    model_fn: Callable[[Dict[str, Any]], BaseEstimator],
    param_space: Callable[[optuna.Trial], Dict[str, Any]],
    X_train: Any,
    y_train: Any,
    scoring: str = "recall",
    cv: int = 5,
    n_trials: int = 1000,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    使用 optuna 進行超參數優化。

    :param model_fn: 函式，用於創建模型實例，傳入參數為超參數字典。
    :param param_space: 函式，用於定義 optuna 的超參數搜索空間。
    :param X_train: 訓練集特徵。
    :param y_train: 訓練集標籤。
    :param scoring: 評估指標（默認為 "recall"）。
    :param cv: 交叉驗證的折數（默認為 5）。
    :param n_trials: 優化的迭代次數（默認為 1000）。
    :return: 最佳模型實例和對應的最佳超參數字典。
    """
    def objective(trial):
        # 獲取當前試驗的超參數
        params = param_space(trial)
        # 創建模型
        model = model_fn(params)
        # 使用交叉驗證評估模型
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
        return score

    # 創建 optuna 的研究對象
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # 獲取最佳參數
    best_params = study.best_params

    # 使用最佳參數創建最終模型
    best_model = model_fn(best_params)
    best_model.fit(X_train, y_train)

    return best_model, best_params

from sklearn.ensemble import RandomForestClassifier
from prepare import prepare

X_train, X_test, y_train, y_test = prepare().getTrainingData(
    year=1,
    test_size=0.8,
    data_type="tor"
)
# RF
# 定義模型函式
def rf_model_fn(params):
    return RandomForestClassifier(**params, random_state=42)

# 定義超參數空間
def rf_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),  # 擴展樹的數量範圍
        "max_depth": trial.suggest_int("max_depth", 2, 50),  # 擴展樹深度範圍
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),  # 增大分裂樣本數
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),  # 增大葉節點樣本數
    }

# 執行優化
best_rf_model, best_rf_params = hyperparameter_optimization(
    model_fn=rf_model_fn,
    param_space=rf_param_space,
    X_train=X_train,
    y_train=y_train,
    scoring="recall",
    cv=5,
    n_trials=1000,
)

print("最佳參數（RF）：", best_rf_params)
print("測試集準確率（RF）：", best_rf_model.score(X_test, y_test))

from sklearn.ensemble import AdaBoostClassifier

# Adaboost
# 定義模型函式
def adaboost_model_fn(params):
    return AdaBoostClassifier(**params, random_state=42)

# 定義超參數空間
def adaboost_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),  # 擴展樹的數量範圍
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 2.0, log=True),  # 擴展學習率範圍
    }

best_adaboost_model, best_adaboost_params = hyperparameter_optimization(
    model_fn=adaboost_model_fn,
    param_space=adaboost_param_space,
    X_train=X_train,
    y_train=y_train,
    scoring="recall",
    cv=5,
    n_trials=1000,
)

print("最佳參數（AdaBoost）：", best_adaboost_params)
print("測試集準確率（AdaBoost）：", best_adaboost_model.score(X_test, y_test))

# Xgboost
from xgboost import XGBClassifier

# 定義模型函式
def xgboost_model_fn(params):
    return XGBClassifier(
        **params,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

# 定義超參數空間
def xgboost_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),  # 擴展樹的數量範圍
        "max_depth": trial.suggest_int("max_depth", 2, 50),  # 擴展樹深度範圍
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),  # 擴展學習率範圍
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),  # 擴展列採樣比例
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),  # 擴展樣本子集比例
        "gamma": trial.suggest_float("gamma", 0, 10),  # 擴展分裂損失範圍
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 20),  # 擴展 L1 正則化範圍
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),  # 擴展 L2 正則化範圍
    }


best_xgboost_model, best_xgboost_params = hyperparameter_optimization(
    model_fn=xgboost_model_fn,
    param_space=xgboost_param_space,
    X_train=X_train,
    y_train=y_train,
    scoring="recall",
    cv=5,
    n_trials=1000,
)

print("最佳參數（XGBoost）：", best_xgboost_params)
print("測試集準確率（XGBoost）：", best_xgboost_model.score(X_test, y_test))
