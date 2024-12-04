from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import joblib

class Models:
    def __init__(self):
        # 存放已訓練的模型
        self.models = {}

    def train_rf(self, X_train, y_train ,save_model=True , data_type="ele"):
        best_tree = {'n_estimators': 50, 'max_depth': 41, 'min_samples_split': 50, 'min_samples_leaf': 2}
        rf = RandomForestClassifier(**best_tree, random_state=42)
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        if save_model:
            joblib.dump(rf, f'./result/model/RF_{data_type}.joblib')

    def train_xgboost(self, X_train, y_train,save_model=True, data_type="ele"):
        best_xgb = {
            'n_estimators': 206,
            'max_depth': 9,
            'learning_rate': 0.206,
            'colsample_bytree': 0.8657,
            'subsample': 0.906,
            'gamma': 0.501,
            'reg_alpha': 0.0053,
            'reg_lambda': 1.412
        }
        xgb = XGBClassifier(**best_xgb, random_state=42)
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb
        if save_model:
            joblib.dump(xgb, f'./result/model/XGB_{data_type}.joblib')

    def train_adaboost(self, X_train, y_train,save_model=True,data_type="ele"):
        best_ada = {'n_estimators': 72, 'learning_rate': 0.0018395156413243648}
        ada = AdaBoostClassifier(**best_ada, random_state=42)
        ada.fit(X_train, y_train)
        self.models["AdaBoost"] = ada
        if save_model:
            joblib.dump(ada, f'./result/model/ADA_{data_type}.joblib')

    def get_models(self):
        return self.models
