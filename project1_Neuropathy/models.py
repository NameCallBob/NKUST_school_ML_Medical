from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import joblib

class Models:
    def __init__(self):
        # 存放已訓練的模型
        self.models = {}

    def train_rf(self, X_train, y_train ,save_model=True , data_type="ele",model_name=None):
        if data_type == "ele":
            # ele ; ACC:0.8205128205128205
            best_tree = {'n_estimators': 73, 'max_depth': 34, 'min_samples_split': 15, 'min_samples_leaf': 2}
        elif data_type == "tor":
            # tor ; ACC:
            best_tree = {'n_estimators': 808, 'max_depth': 32, 'min_samples_split': 2, 'min_samples_leaf': 1}

        rf = RandomForestClassifier(**best_tree, random_state=42)
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        if save_model:
            joblib.dump(rf, f'./result/model/RF_{model_name}.joblib')

    def train_xgboost(self, X_train, y_train,save_model=True, data_type="ele",model_name=None):
        if data_type == "ele":
            # ele ; ACC:0.8205128205128205
            best_xgb = {'n_estimators': 935, 'max_depth': 4, 'learning_rate': 0.024538977645993307, 'colsample_bytree': 0.9743703884931614, 'subsample': 0.67800206777262, 'gamma': 3.3372020239182567, 'reg_alpha': 0.01412867712047361, 'reg_lambda': 1.9186201283256892}
        elif data_type == "tor":
            # tor ; ACC:
            best_xgb = {'n_estimators': 905, 'max_depth': 8, 'learning_rate': 0.35994543231367176, 'colsample_bytree': 0.8873639342476397, 'subsample': 0.508282091586604, 'gamma': 4.080637049724824, 'reg_alpha': 0.9993551863456485, 'reg_lambda': 0.02667262528884007}

        xgb = XGBClassifier(**best_xgb, random_state=42)
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb
        if save_model:
            joblib.dump(xgb, f'./result/model/XGB_{model_name}.joblib')

    def train_adaboost(self, X_train, y_train,save_model=True,data_type="ele",model_name=None):
        if data_type == "tor":
            best_ada = {'n_estimators': 124, 'learning_rate': 0.001008430295687769}
        elif data_type == "ele":
            best_ada = {'n_estimators': 429, 'learning_rate': 1.9649821771621983}
        ada = AdaBoostClassifier(**best_ada, random_state=42)
        ada.fit(X_train, y_train)
        self.models["AdaBoost"] = ada

        if save_model:
            joblib.dump(ada, f'./result/model/ADA_{model_name}.joblib')

    def get_models(self):
        return self.models
