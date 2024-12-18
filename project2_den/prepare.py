from data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

class Prepare:

    def __init__(self):
        self.ob = Data()

    def getTrainingData(self,test_size=0.2,binary_classification=False, target_class=None):
        """
        導入測試資料並進行處理，最後返回訓練及測試資料
        """

        # 獲取多個陣列資料並合併
        data_arrays = self.ob.load_result()  # 假設返回的是一個包含多個 DataFrame 的 list

        if not isinstance(data_arrays, list):
            raise ValueError("load_result 必須返回一個包含 DataFrame 的列表")

        # 合併所有 DataFrame，根據 idcode 和 opdno 進行依值合併（直接堆疊）
        merged_data = pd.concat(data_arrays, ignore_index=True)


        # 移除 is_normal 欄位中含 NaN 的數據
        merged_data = merged_data.dropna(subset=['is_normal'])

        # 選取特徵與目標
        feature = ['sex', 'labit', 'labsh1it', 'labnmabv', 'labrefcval', 'labresuval', 'is_normal']
        target = ['sick_type']

        # 驗證特徵與目標是否存在於合併後的資料中
        for col in feature + target:
            if col not in merged_data.columns:
                raise ValueError(f"合併後的資料缺少必要欄位: {col}")

        X = merged_data[feature].copy()
        y = merged_data[target].copy()

        # 確保 labresuval 欄位為 float
        if 'labresuval' in X.columns:
            X['labresuval'] = pd.to_numeric(X['labresuval'], errors='coerce').fillna(0.0).astype(float)

        # 檢查並轉換 object 欄位
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # 使用 LabelEncoder 進行轉換
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

                    # 確保轉換後的類型為整數
                    X[col] = X[col].astype(int)

                except Exception as e:
                    print(f"欄位 '{col}' 轉換失敗，錯誤訊息: {e}")

        # 對 is_normal 欄位進行二元判斷處理並轉為 int
        if binary_classification and target_class is not None:
            X.loc[:, 'is_normal'] = X['is_normal'].apply(lambda x: 1 if x == target_class else 0)
        X['is_normal'] = X['is_normal'].astype(int)


        if binary_classification and target_class is not None:
            y = y['sick_type'].apply(lambda x: 1 if x == target_class else 0).astype(int)

        # self.__check_dataframe_columns(X,y)

        # 分割資料
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        # 使用 SMOTE 處理訓練資料的不平衡問題
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        print("訓練資料預處理完畢")
        return X_train, X_test, y_train, y_test

    # 檢查特徵與目標資料的類型和部分值
    def __check_dataframe_columns(self,X, y):
        print("特徵欄位資料類型與前 5 筆數據:")
        for col in X.columns:
            print(f"{col}: 類型 {X[col].dtype}, 前 5 筆值 {X[col].head().values}")
        print("\n目標變數資料類型與前 5 筆數據:")
        print(f"{y.columns[0]}: 類型 {y.dtypes[0]}, 前 5 筆值 {y.head().values}")
    
    def feature_importance(self):
        """
        找尋最佳參數
        """
        from sklearn.ensemble import RandomForestClassifier
        import shap
        # 分割資料
        X_train, X_test, y_train, y_test = self.getTrainingData(
            test_size=0.2,binary_classification=True,target_class=0
        )

        # 訓練模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 建立 SHAP 解釋器
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 畫出特徵重要性（summary plot）
        shap.summary_plot(shap_values, X_test, plot_type="bar")

if __name__ == "__main__":
    ob = Prepare()
    # ob.getTrainingData()
    ob.feature_importance()