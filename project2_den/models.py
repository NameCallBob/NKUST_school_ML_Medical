from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import joblib
import numpy as np

class Models:
    def __init__(self):
        # 存放已訓練的模型
        self.models = {}

    def _detect_classification_type(self, y):
        """檢測是二元分類還是多類別分類"""
        num_classes = len(np.unique(y))
        if num_classes == 2:
            return "binary"
        elif num_classes > 2:
            return "multiclass"
        else:
            raise ValueError("y 中的類別數量無效，請檢查資料！")

    def train_rf(self, X_train, y_train, save_model=True):
        classification_type = self._detect_classification_type(y_train)

        # 設置模型
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' if classification_type == "binary" else None
        )
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        if save_model:
            joblib.dump(rf, './result/model/RF.joblib')

    def train_xgboost(self, X_train, y_train, save_model=True):
        classification_type = self._detect_classification_type(y_train)

        # 設置目標參數
        best_xgb = {
            "objective": "binary:logistic" if classification_type == "binary" else "multi:softmax",
            "num_class": len(np.unique(y_train)) if classification_type == "multiclass" else None,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 42,
        }
        xgb = XGBClassifier(**best_xgb)
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb
        if save_model:
            joblib.dump(xgb, './result/model/XGB.joblib')

    def train_adaboost(self, X_train, y_train, save_model=True):
        classification_type = self._detect_classification_type(y_train)

        # 設置模型
        ada = AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        )
        ada.fit(X_train, y_train)
        self.models["AdaBoost"] = ada
        if save_model:
            joblib.dump(ada, './result/model/ADA.joblib')

    def get_models(self):
        return self.models