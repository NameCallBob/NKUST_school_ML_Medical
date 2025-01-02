from data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np


class Prepare:

    def __init__(self):
        self.ob = Data()

    def getTrainingData(self, test_size=0.2, binary_classification=False, target_class=None):
        """
        導入測試資料並進行處理，最後返回訓練及測試資料
        """

        # 獲取多個陣列資料並合併
        data_arrays = self.ob.load_result()  # 假設返回的是一個包含多個 DataFrame 的 list

        if not isinstance(data_arrays, list):
            raise ValueError("load_result 必須返回一個包含 DataFrame 的列表")

        # 合併所有 DataFrame，根據 idcode 和 opdno 進行依值合併（直接堆疊）
        merged_data = pd.concat(data_arrays, ignore_index=True)

        feature = ['Hematocrit', 'Platelets', 'WBC']

        target = ['sick_type']

        # 驗證特徵與目標是否存在於合併後的資料中
        for col in feature + target:
            if col not in merged_data.columns:
                raise ValueError(f"合併後的資料缺少必要欄位: {col}")

        merged_data = merged_data.replace(0.0, np.nan)  # 不填值使用
        merged_data = merged_data.dropna(axis=0)

        X = merged_data[feature].copy()
        y = merged_data[target].copy()

        # 確保特徵欄位為浮點數
        X = X.astype(float)

        # 確保目標欄位為整數
        y = y.astype(int)

        # 檢查並轉換 object 欄位
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                except Exception as e:
                    raise ValueError(f"欄位 '{col}' 轉換失敗，錯誤訊息: {e}")

        # # 標準化特徵
        # scaler = StandardScaler()
        # X = pd.DataFrame(scaler.fit_transform(X), columns=feature)

        if binary_classification and target_class is not None:
            if target_class not in y['sick_type'].values:
                raise ValueError(f"指定的 target_class ({target_class}) 不存在於數據中。")
            y = y['sick_type'].apply(
                lambda x: 1 if x == target_class else 0).astype(int)
        else:
            y = y['sick_type']-1

        # 分割資料
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        from collections import Counter

        # 在 SMOTE 前檢查類別數量

        print("分割後 y_train 類別分佈:", Counter(y_train))
        print("分割後 y_test 類別分佈:", Counter(y_test))

        if len(Counter(y_train)) <= 1:
            raise ValueError("y_train 只包含一個類別，無法進行分類模型訓練。請檢查數據。")
        else:
            # from imblearn.over_sampling import RandomOverSampler
            # ros = RandomOverSampler(random_state=42)
            # X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

            # from imblearn.under_sampling import TomekLinks
            # tl = TomekLinks()
            # X_train_balanced, y_train_balanced = tl.fit_resample(X_train, y_train)

            from imblearn.combine import SMOTEENN
            # 使用 SMOTEENN 混合重採樣
            smote_enn = SMOTEENN(random_state=42)
            X_train_balanced, y_train_balanced = smote_enn.fit_resample(
                X_train, y_train)
            print("重採樣後 y_train_balanced 類別分佈:", Counter(y_train_balanced))

        print("訓練資料預處理完畢")
        return X_train_balanced, X_test , y_train_balanced, y_test

    def getTrainingDataWithFeatureEngineering(self, test_size=0.2, binary_classification=False, target_class=None):
        X_train, X_test, y_train, y_test = self.getTrainingData(
            test_size=test_size, binary_classification=binary_classification, target_class=target_class
        )

        # # 處理缺失值
        # X_train = self.handle_missing_values(X_train)
        # X_test = self.handle_missing_values(X_test)
        # print(
        #     f"After missing values: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        # print(
        #     f"After missing values: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        # 創造新特徵
        X_train = self.create_new_features(X_train)
        X_test = self.create_new_features(X_test)
        print(
            f"After creating new features: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        print(
            f"After creating new features: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        # 創造多項式特徵
        X_train = self.polynomial_features(X_train)
        X_test = self.polynomial_features(X_test)
        print(
            f"After polynomial features: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        print(
            f"After polynomial features: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        # 處理異常值，同時更新 y_train 和 y_test
        # X_train, y_train = self.handle_outliers(X_train, y_train)
        # X_test, y_test = self.handle_outliers(X_test, y_test)
        # print(f"After handling outliers: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        # print(f"After handling outliers: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        # 標準化特徵
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(
            X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(
            X_test), columns=X_test.columns, index=X_test.index)
        print(
            f"After scaling: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        print(
            f"After scaling: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        # 特徵選擇
        X_train, X_test = self.feature_selection(X_train, y_train, X_test, num_features=10)
        print(f"After feature selection: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        print(f"After feature selection: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        # 確保 X_train 和 y_train 及 X_test 和 y_test 的行數一致
        assert X_train.shape[0] == y_train.shape[0], "X_train 和 y_train 的行數不一致"
        assert X_test.shape[0] == y_test.shape[0], "X_test 和 y_test 的行數不一致"

        # 檢查特徵名稱唯一性
        assert X_train.columns.is_unique, "X_train 的特徵名稱不唯一"
        assert X_test.columns.is_unique, "X_test 的特徵名稱不唯一"

        return X_train, X_test, y_train, y_test
    # 檢查特徵與目標資料的類型和部分值

    def __check_dataframe_columns(self, X, y):
        print("特徵欄位資料類型與前 5 筆數據:")
        for col in X.columns:
            print(f"{col}: 類型 {X[col].dtype}, 前 5 筆值 {X[col].head().values}")
        print("\n目標變數資料類型與前 5 筆數據:")
        print(f"{y.columns[0]}: 類型 {y.dtypes[0]}, 前 5 筆值 {y.head().values}")

    def validate_concat(self, data_arrays):
        """
        驗證 `pd.concat` 的數據一致性
        :param data_arrays: List of DataFrame
        :return: None
        """
        import pandas as pd

        # 檢查是否為列表且包含 DataFrame
        if not isinstance(data_arrays, list):
            raise ValueError("data_arrays 必須是包含 DataFrame 的列表")
        if not all(isinstance(df, pd.DataFrame) for df in data_arrays):
            raise ValueError("data_arrays 中的所有元素必須是 DataFrame")

        # 檢查每個 DataFrame 的結構是否一致
        first_columns = data_arrays[0].columns if len(
            data_arrays) > 0 else None
        for idx, df in enumerate(data_arrays):
            if not list(df.columns) == list(first_columns):
                raise ValueError(f"第 {idx} 個 DataFrame 的欄位結構與其他 DataFrame 不一致")

        print("所有 DataFrame 的欄位結構一致。")

        # 合併後進一步檢查
        merged_data = pd.concat(data_arrays, ignore_index=True)

        # 確認合併結果是否有重複行
        duplicate_count = merged_data.duplicated().sum()
        if duplicate_count > 0:
            print(f"警告: 合併後的數據中發現 {duplicate_count} 行重複。")

        # 確認是否存在缺失值
        missing_values = merged_data.isnull().sum().sum()
        if missing_values > 0:
            print(f"警告: 合併後的數據中發現 {missing_values} 個缺失值。")

        print("合併驗證完成，返回的數據結構如下：")
        print(merged_data.info())
        print(merged_data.head())

        return merged_data

    def feature_importance_snap(self):
        """
        找尋最佳參數
        """
        from sklearn.ensemble import RandomForestClassifier
        import shap
        # print("目前為停用的狀態，發現他找了一天都沒有找到")
        # 分割資料
        X_train, X_test, y_train, y_test = self.getTrainingData(
            test_size=0.2, binary_classification=True, target_class=1
        )

        # 訓練模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 建立 SHAP 解釋器
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 畫出特徵重要性（summary plot）
        shap.summary_plot(shap_values, X_test, plot_type="bar")

    def feature_importance(self):
        """
        使用 LightGBM 訓練模型並顯示特徵重要性
        """
        import lightgbm as lgb
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        import pandas as pd

        # 獲取訓練數據
        X_train, X_test, y_train, y_test = self.getTrainingDataWithFeatureEngineering(
            test_size=0.2, binary_classification=False, target_class=0
        )
        print("y_train classes and counts:",
              np.unique(y_train, return_counts=True))

        # 設置 LightGBM 模型
        model = lgb.LGBMClassifier(
            objective='multiclass' if len(set(y_train)) > 2 else 'binary',
            num_class=len(set(y_train)) if len(set(y_train)) > 2 else None,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # 獲取特徵重要性
        feature_importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)

        # 顯示特徵重要性
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance["Feature"],
                 feature_importance["Importance"])
        plt.title("LightGBM Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.gca().invert_yaxis()
        plt.show()

    def handle_missing_values(self, X):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        return X

    def handle_outliers(self, X, y):
        from scipy import stats
        z_scores = np.abs(stats.zscore(X))
        threshold = 3
        mask = (z_scores < threshold).all(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]
        return X_clean, y_clean

    def create_new_features(self, X):
        X['Platelets_WBC_Ratio'] = X['Platelets'] / (X['WBC'] + 1e-5)
        X['Hematocrit_WBC_Ratio'] = X['Hematocrit'] / (X['WBC'] + 1e-5)
        X['Hematocrit_Platelets_Product'] = X['Hematocrit'] * X['Platelets']

        X['WBC_log'] = np.log1p(X['WBC'])

        X.columns = [col.replace('/', '_') for col in X.columns]

        return X

    def polynomial_features(self, X):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(X.columns)

        # 替換特徵名稱中的 '/' 為 '_'
        poly_feature_names = [name.replace('/', '_')
                              for name in poly_feature_names]

        X_poly = pd.DataFrame(
            poly_features, columns=poly_feature_names, index=X.index)

        # 合併原始特徵與多項式特徵
        X = pd.concat([X, X_poly], axis=1)

        # 移除重複的特徵名稱
        X = X.loc[:, ~X.columns.duplicated()]

        return X

    def feature_selection(self, X_train, y_train, X_test, num_features=10):
        import xgboost as xgb
        from sklearn.feature_selection import SelectFromModel

        model = xgb.XGBClassifier(
            n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        selector = SelectFromModel(
            model, max_features=num_features, prefit=True)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        selected_features = X_train.columns[selector.get_support()]
        print(f"選擇的特徵 ({num_features} 個): {list(selected_features)}")
        return pd.DataFrame(X_train_selected, columns=selected_features), pd.DataFrame(X_test_selected, columns=selected_features)


if __name__ == "__main__":
    ob = Prepare()
    # ob.getTrainingData()
    ob.feature_importance()
    # ob.feature_importance_snap()
