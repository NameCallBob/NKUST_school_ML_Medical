from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import joblib

class Models:
    def __init__(self):
        # 存放已訓練的模型
        self.models = {}

    def train_rf(self, X_train, y_train ,save_model=True ):
        best_tree = {}
        rf = RandomForestClassifier(**best_tree, random_state=42)
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        if save_model:
            joblib.dump(rf, f'./result/model/RF.joblib')

    def train_xgboost(self, X_train, y_train,save_model=True):
        best_xgb = {}
        xgb = XGBClassifier(**best_xgb, random_state=42)
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb
        if save_model:
            joblib.dump(xgb, f'./result/model/XGB.joblib')

    def train_adaboost(self, X_train, y_train,save_model=True):
        best_ada = {}
        ada = AdaBoostClassifier(**best_ada, random_state=42)
        ada.fit(X_train, y_train)
        self.models["AdaBoost"] = ada

        if save_model:
            joblib.dump(ada, f'./result/model/ADA.joblib')

    def get_models(self):
        return self.models
