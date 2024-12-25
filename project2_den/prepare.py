from data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

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
        
        # 選取特徵與目標
        feature = ['ALT/GPT', 'CRP',
                    'Hematocrit', 
                   'Platelets', 'WBC']
        
        target = ['sick_type']

        # 驗證特徵與目標是否存在於合併後的資料中
        for col in feature + target:
            if col not in merged_data.columns:
                raise ValueError(f"合併後的資料缺少必要欄位: {col}")
        
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

        # 填補缺失值
        X.fillna(X.mean(), inplace=True)

        # 標準化特徵
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature)

        # 處理二元分類問題
        if binary_classification and target_class is not None:
            y = y['sick_type'].apply(lambda x: 1 if x == target_class else 0).astype(int)
        else:
            y = y['sick_type']

        # 分割資料
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        from collections import Counter

        # 在 SMOTE 前檢查類別數量
        if len(Counter(y_train)) > 1:  # 確保有多於一個類別
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            print("警告: y_train 只包含一個類別，跳過 SMOTE 處理。")

        print("訓練資料預處理完畢")
        return X_train, X_test, y_train, y_test


    # 檢查特徵與目標資料的類型和部分值
    def __check_dataframe_columns(self,X, y):
        print("特徵欄位資料類型與前 5 筆數據:")
        for col in X.columns:
            print(f"{col}: 類型 {X[col].dtype}, 前 5 筆值 {X[col].head().values}")
        print("\n目標變數資料類型與前 5 筆數據:")
        print(f"{y.columns[0]}: 類型 {y.dtypes[0]}, 前 5 筆值 {y.head().values}")
    
    def feature_importance_snap(self):
        """
        找尋最佳參數
        """
        from sklearn.ensemble import RandomForestClassifier
        import shap
        # print("目前為停用的狀態，發現他找了一天都沒有找到")
        # 分割資料
        X_train, X_test, y_train, y_test = self.getTrainingData(
            test_size=0.2,binary_classification=True,target_class=1
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
        找尋最佳參數
        """
        import matplotlib.pyplot as plt
        import xgboost as xgb

        X_train, X_test, y_train, y_test = self.getTrainingData(
            test_size=0.2,binary_classification=True,target_class=1
        )

        # 訓練 XGBoost 模型
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 顯示特徵重要性
        xgb.plot_importance(model)
        plt.title("XGBoost Feature Importance")
        plt.show()


if __name__ == "__main__":
    ob = Prepare()
    # ob.getTrainingData()
    # ob.feature_importance()
    ob.feature_importance_snap()