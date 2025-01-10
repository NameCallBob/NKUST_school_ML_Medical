from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import numpy as np

class Models:
    def __init__(self, verbose=True):
        # 存放已訓練的模型
        self.models = {}
        self.verbose = verbose

    def _detect_classification_type(self, y):
        """檢測是二元分類還是多類別分類"""
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        if num_classes == 2:
            return "binary"
        elif num_classes > 2:
            return "multiclass"
        else:
            raise ValueError("y 中的類別數量無效，請檢查資料！")

    def train_rf(self, X_train, y_train, save_model=True, save_path='./result/model/RF.joblib'):
        classification_type = self._detect_classification_type(y_train)

        # 設置模型
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' if classification_type == "binary" else 'balanced_subsample'
        )
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        if save_model:
            joblib.dump(rf, save_path)
        if self.verbose:
            print(f"RandomForest 已訓練完成並儲存於 {save_path}")

    def train_xgboost(self, X_train, y_train, save_model=True, save_path='./result/model/XGB.joblib'):
        classification_type = self._detect_classification_type(y_train)
        best_xgb ={'n_estimators': 486,
                    'max_depth': 26,
                    'learning_rate': 0.22307437151594256,
                    'colsample_bytree': 0.9429541973810297,
                    'subsample': 0.9249612441962007,
                    'gamma': 0.004748206429748825,
                    'reg_alpha': 1.4410927906583821,
                    'reg_lambda': 5.757800230534241,
                    'objective': "binary:logistic" if classification_type == "binary" else "multi:softprob",
                'num_class': len(np.unique(y_train)) if classification_type == "multiclass" else None,
                'random_state': 42,
                'use_label_encoder': False,  # 停用舊的編碼方式
                }
        # best_xgb = {
        #     'n_estimators': 957,
        #     'max_depth': 33,
        #     'learning_rate': 0.03995070033806279,
        #     'colsample_bytree': 0.7673016513030932,
        #     'subsample': 0.9158939353013769,
        #     'gamma': 2.245009613843046,
        #     'reg_alpha': 7.971891752929898,
        #     'reg_lambda': 6.603447890913207,
        # }

        xgb = XGBClassifier(**best_xgb)
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb
        if save_model:
            joblib.dump(xgb, save_path)
        if self.verbose:
            print(f"XGBoost 已訓練完成並儲存於 {save_path}")

    def train_adaboost(self, X_train, y_train, save_model=True, save_path='./result/model/ADA.joblib'):
        classification_type = self._detect_classification_type(y_train)

        best_ada ={'n_estimators': 811, 'learning_rate': 1.9851485234209303}
        ada = AdaBoostClassifier(
            **best_ada,
            random_state=42
        )
        ada.fit(X_train, y_train)
        self.models["AdaBoost"] = ada
        if save_model:
            joblib.dump(ada, save_path)
        if self.verbose:
            print(f"AdaBoost 已訓練完成並儲存於 {save_path}")

    def train_lightgbm(self, X_train, y_train, save_model=True, save_path='./result/model/LGBM.joblib'):
        classification_type = self._detect_classification_type(y_train)

        best_lgbm = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'class_weight': 'balanced' if classification_type == "binary" else None,
            'random_state': 42
        }

        lgbm = LGBMClassifier(**best_lgbm)
        lgbm.fit(X_train, y_train)
        self.models["LightGBM"] = lgbm
        if save_model:
            joblib.dump(lgbm, save_path)
        if self.verbose:
            print(f"LightGBM 已訓練完成並儲存於 {save_path}")

    def train_catboost(self, X_train, y_train, save_model=True, save_path='./result/model/CAT.joblib'):
        classification_type = self._detect_classification_type(y_train)

        best_catboost = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 10,
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': False  # 禁止訓練過程中的詳細輸出
        }

        catboost = CatBoostClassifier(**best_catboost)
        catboost.fit(X_train, y_train)
        self.models["CatBoost"] = catboost
        if save_model:
            joblib.dump(catboost, save_path)
        if self.verbose:
            print(f"CatBoost 已訓練完成並儲存於 {save_path}")

    def get_models(self):
        return self.models