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

        best_xgb = {'n_estimators': 957,\
                    'max_depth': 33,\
                    'learning_rate': 0.03995070033806279,\
                    'colsample_bytree': 0.7673016513030932,\
                    'subsample': 0.9158939353013769,\
                    'gamma': 2.245009613843046,\
                    'reg_alpha': 7.971891752929898,\
                    'reg_lambda': 6.603447890913207,
                    "objective": "binary:logistic" if classification_type == "binary" else "multi:softmax",
                    "num_class": len(np.unique(y_train)) if classification_type == "multiclass" else None,
                    "random_state": 42,
                    }

        xgb = XGBClassifier(**best_xgb)
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb
        if save_model:
            joblib.dump(xgb, './result/model/XGB.joblib')

    def train_adaboost(self, X_train, y_train, save_model=True):
        classification_type = self._detect_classification_type(y_train)
        best_ada = {'n_estimators': 735, 'learning_rate': 0.9756155617924089}
        # 設置模型
        ada = AdaBoostClassifier(
            **best_ada,
            random_state=42
        )
        ada.fit(X_train, y_train)
        self.models["AdaBoost"] = ada
        if save_model:
            joblib.dump(ada, './result/model/ADA.joblib')

    def get_models(self):
        return self.models