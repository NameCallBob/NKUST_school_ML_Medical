from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier



import matplotlib.pyplot as plt
import numpy as np


class Models:
    def __init__(self):
        self.models = {}

    def train_rf(self, X_train, y_train, n_estimators=100, max_depth=None):
        best_tree={'n_estimators': 70, 'max_depth': 19, 'min_samples_split': 7, 'min_samples_leaf': 1}
        rf = RandomForestClassifier(**best_tree)
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf

    def train_xgboost(self, X_train, y_train):
        best_xgb = {'n_estimators': 206, 'max_depth': 9, 'learning_rate': 0.20607789024546083, 'colsample_bytree': 0.8657151792491395, 'subsample': 0.9059667717354556, 'gamma': 0.501185342832905, 'reg_alpha': 0.005264955358300183, 'reg_lambda': 1.4121579985940274}
        xgb = XGBClassifier(**best_xgb)
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb

    def train_adaboost(self, X_train, y_train):
        best_ada = {'n_estimators': 112, 'learning_rate': 0.14906309146271576}
        adaboost = AdaBoostClassifier(**best_ada)
        adaboost.fit(X_train, y_train)
        self.models["AdaBoost"] = adaboost



if __name__ == "__main__":
    from train import prepare ; from evaluate_model import evaluate
    X_train, X_test, y_train, y_test = prepare().getTrainingData()

    # 初始化 Models 類別
    model_handler = Models()

    # 用來存放各模型的評估結果
    results = {}

    # 隨機森林
    print("訓練 RandomForest 模型...")
    model_handler.train_rf(X_train, y_train)
    results["RandomForest"] = evaluate.model("RandomForest",
                                            model_handler.models.get("RandomForest"),
                                            X_test,
                                            y_test)

    # XGBoost
    print("訓練 XGBoost 模型...")
    model_handler.train_xgboost(X_train, y_train)
    results["XGBoost"] = evaluate.model("XGBoost",
                                        model_handler.models.get("XGBoost"),
                                        X_test,
                                        y_test)

    # AdaBoost
    print("訓練 AdaBoost 模型...")
    model_handler.train_adaboost(X_train, y_train)
    results["AdaBoost"] = evaluate.model("AdaBoost",
                                         model_handler.models.get("AdaBoost"),
                                         X_test,
                                         y_test)

    # 輸出結果
    print("\n模型評估結果：")
    for model_name, metrics in results.items():
        print(f"\n{model_name} 模型的評估結果:")
        for metric, value in metrics.items():
            if metric == "Confusion Matrix":
                print(f"{metric}:\n{value}")
            elif value is None:
                print(f"{metric}: 無法計算")
            else:
                print(f"{metric}: {value:.4f}")