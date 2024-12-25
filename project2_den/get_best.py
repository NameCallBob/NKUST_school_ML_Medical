import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from typing import Callable, Dict, Any, Tuple
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from multiprocessing import Pool
import concurrent.futures
import numpy as np

# 禁用 Optuna 日誌輸出
optuna.logging.set_verbosity(optuna.logging.WARN)

def hyperparameter_optimization(
    model_fn: Callable[[Dict[str, Any]], BaseEstimator],
    param_space: Callable[[optuna.Trial], Dict[str, Any]],
    X_train: Any,
    y_train: Any,
    classification_type: str = "binary",
    scoring: str = "recall",
    cv: int = 5,
    n_trials: int = 100,
    n_jobs: int = -1  # 並行運行所有試驗
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    使用 optuna 進行超參數優化，並進行並行計算。

    :param model_fn: 函式，用於創建模型實例，傳入參數為超參數字典。
    :param param_space: 函式，用於定義 optuna 的超參數搜索空間。
    :param X_train: 訓練集特徵。
    :param y_train: 訓練集標籤。
    :param classification_type: "binary" 或 "multiclass"，指示分類類型。
    :param scoring: 評估指標（默認為 "recall"）。
    :param cv: 交叉驗證的折數（默認為 5）。
    :param n_trials: 優化的迭代次數（默認為 100）。
    :param n_jobs: 用於並行化的工作數量（默認 -1，表示使用所有可用核）。
    :return: 最佳模型實例和對應的最佳超參數字典。
    """
    def objective(trial):
        params = param_space(trial)
        model = model_fn(params)

        # 創建交叉驗證策略
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scores = []
        for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X_train, y_train)):
            
            X_train_fold = X_train.iloc[train_idx]
            X_test_fold = X_train.iloc[test_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)

            if scoring == "recall":
                score = recall_score(y_test_fold, y_pred, average="macro" if classification_type == "multiclass" else "binary")
            elif scoring == "accuracy":
                score = accuracy_score(y_test_fold, y_pred)
            elif scoring == "f1":
                score = f1_score(y_test_fold, y_pred, average="macro" if classification_type == "multiclass" else "binary")
            else:
                raise ValueError("Unsupported scoring type")

            scores.append(score)

            # 記錄中間結果
            trial.report(score, fold)

            # 如果中間結果不理想，可提前終止
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # 創建 optuna 的研究對象
    study = optuna.create_study(direction="maximize")

    # 使用內建的 n_jobs 支援並行化
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # 獲取最佳參數
    best_params = study.best_params

    # 使用最佳參數創建最終模型
    best_model = model_fn(best_params)
    best_model.fit(X_train, y_train)

    return best_model, best_params

# 定義模型與超參數空間

# RandomForestClassifier

def rf_model_fn(params):
    return RandomForestClassifier(**params, random_state=42)

def rf_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
    }

# AdaBoostClassifier

def adaboost_model_fn(params):
    return AdaBoostClassifier(**params, random_state=42)

def adaboost_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 2.0, log=True),
    }

# XGBClassifier

def xgboost_model_fn(params):
    return XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric="mlogloss")

def xgboost_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),
    }

# 測試功能
from prepare import Prepare

X_train, X_test, y_train, y_test = Prepare().getTrainingData(binary_classification=True,target_class=1 ,test_size=0.2)

# RandomForestClassifier 測試
best_rf_model, best_rf_params = hyperparameter_optimization(
    model_fn=rf_model_fn,
    param_space=rf_param_space,
    X_train=X_train,
    y_train=y_train,
    classification_type="binary",
    scoring="recall",
    cv=5,
    n_trials=10,
    n_jobs=-1
)
print("最佳參數（RF）：", best_rf_params)
print("測試集準確率（RF）：", best_rf_model.score(X_test, y_test))

# AdaBoostClassifier 測試
best_adaboost_model, best_adaboost_params = hyperparameter_optimization(
    model_fn=adaboost_model_fn,
    param_space=adaboost_param_space,
    X_train=X_train,
    y_train=y_train,
    classification_type="binary",
    scoring="recall",
    cv=5,
    n_trials=10,
    n_jobs=-1
)
print("最佳參數（AdaBoost）：", best_adaboost_params)
print("測試集準確率（AdaBoost）：", best_adaboost_model.score(X_test, y_test))

# XGBClassifier 測試
best_xgboost_model, best_xgboost_params = hyperparameter_optimization(
    model_fn=xgboost_model_fn,
    param_space=xgboost_param_space,
    X_train=X_train,
    y_train=y_train,
    classification_type="binary",
    scoring="recall",
    cv=5,
    n_trials=10,
    n_jobs=-1
)
print("最佳參數（XGBoost）：", best_xgboost_params)
print("測試集準確率（XGBoost）：", best_xgboost_model.score(X_test, y_test))
