from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

class Models:
    def __init__(self):
        # 存放已訓練的模型
        self.models = {}

    def train_rf(self, X_train, y_train):
        best_tree = {
            'n_estimators': 70,
            'max_depth': 19,
            'min_samples_split': 7,
            'min_samples_leaf': 1
        }
        rf = RandomForestClassifier(**best_tree, random_state=42)
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf

    def train_xgboost(self, X_train, y_train):
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

    def train_adaboost(self, X_train, y_train):
        best_ada = {'n_estimators': 112, 'learning_rate': 0.149}
        ada = AdaBoostClassifier(**best_ada, random_state=42)
        ada.fit(X_train, y_train)
        self.models["AdaBoost"] = ada

    def get_models(self):
        return self.models
