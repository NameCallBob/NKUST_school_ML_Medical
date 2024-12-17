import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

"""
主要資料欄位：
電生理檢查
1.DML_Med_R_1,CMAP_Med_R_1,MNCV_Med_R_1,DML_Med_L_1,CMAP_Med_L_1,MNCV_Med_L_1,DML_Ula_R_1,CMAP_Ula_R_1,MNCV_Ula_R_1,DML_Ula_L_1,CMAP_Ula_L_1,MNCV_Ula_L_1,DML_Per_R_1,CMAP_Per_R_1,MNCV_Per_R_1,DML_Per_L_1,CMAP_Per_L_1,MNCV_Per_L_1,DML_Tib_R_1,CMAP_Tib_R_1,MNCV_Tib_R_1,DML_Tib_L_1,CMAP_Tib_L_1,MNCV_Tib_L_1,F_Med_R_1,F_Med_L_1,F_Ula_R_1,F_Ula_L_1,F_Per_R_1,F_Per_L_1,F_Tib_R_1,F_Tib_L_1,H_reflex_R_1,H_Reflex_L_1,SLO_Med_R_1,SNAP_Med_R_1,SNCV_Med_R_1,SLO_MP_R_1,SNAP_MP_R_1,SNCV_MP_R_1,SLO_Med_L_1,SNAP_Med_L_1,SNCV_Med_L_1,SLO_MP_L_1,SNAP_MP_L_1,SNCV_MP_L_1,SLO_Ula_R_1,SNAP_Ula_R_1,SNCV_Ula_R_1,SLO_Ula_L_1,SNAP_Ula_L_1,SNCV_Ula_L_1,SLO_Sur_R_1,SNAP_Sur_R_1,SNCV_Sur_R_1,SLO_Sur_L_1,SNAP_Sur_L_1,SNCV_Sur_L_1
多倫多量表
1.有主觀徵象_1,多倫多_痛_1,多倫多_麻_1,多倫多_刺_1,多倫多_無力_1,多倫多_協調_1,多倫多_UE_1,有客觀症狀_1,多倫多_刺鈍左_1,多倫多_刺鈍右_1,多倫多_溫鈍左_1,多倫多_溫鈍右_1,多倫多_碰鈍左_1,多倫多_碰鈍右_1,多倫多_振鈍左_1,多倫多_振鈍右_1,多倫多_JPS鈍左_1,多倫多_JPS鈍右_1,多倫多_knee左_1,多倫多_knee右_1,多倫多_ankle左_1,多倫多_ankle右_1
主要預測欄位：
是否有神經病變_確診神經病變
"""
class trainingData:
    """
    導入資料並開始預處理
    """
    
    def load_test(self):
        """導入測試資料"""
        file_path = "./data/test.xlsx"
        data = pd.read_excel(file_path)
        return data
    
    def load_data(self):
        """導入資料"""

        file_path = "./data/origin.xlsx"  # 替換為實際檔案路徑
        year1_feature , year2_feature  = self.string_trans_array()
        
        # 使用正確的方式合併列表
        # NOTE:feature[3]單純是字串，有關於最後要預測的結果
        year1_tor_combined = year1_feature[0] + [year1_feature[2]]
        year1_ele_combined = year1_feature[1] + [year1_feature[2]]
        year2_tor_combined = year2_feature[0] + [year2_feature[2]]
        year2_ele_combined = year2_feature[1] + [year2_feature[2]]

        data = pd.read_excel(file_path)
        year1_tor = data[year1_tor_combined]
        year1_ele = data[year1_ele_combined]
        year2_tor = data[year2_tor_combined]
        year2_ele = data[year2_ele_combined]
        
        return {'year1_tor':year1_tor, 'year1_ele':year1_ele, 'year2_tor':year2_tor, 'year2_ele':year2_ele}
    
    def is_imbalanced(self, data, label_column, threshold=0.2):
        """
        檢查資料是否不平衡
        """
        class_distribution = data[label_column].value_counts(normalize=True)
        min_class_proportion = class_distribution.min()
        return min_class_proportion < threshold

    def preprocess(self):
        """處理缺失值並選擇性地應用 SMOTE"""
        data_dict = self.load_data()
        year1_tor = data_dict['year1_tor']
        year1_ele = data_dict['year1_ele']
        year2_tor = data_dict['year2_tor']
        year2_ele = data_dict['year2_ele']

        # year1與year2皆有缺失值，由於資料非屬於可以填補的性質，將直接刪除整列資料
        year1_tor_clean = year1_tor.dropna()
        year1_ele_clean = year1_ele.dropna()
        year2_tor_clean = year2_tor.dropna()
        year2_ele_clean = year2_ele.dropna()

        # 判斷是否需要平衡 year1
        if self.is_imbalanced(year1_tor_clean, label_column="確診神經病變_1"):
            print("Year1 資料不平衡，應用 SMOTE")
            smote = SMOTE(random_state=42)
            X_year1_tor = year1_tor_clean.drop(columns=["確診神經病變_1"])
            y_year1_tor = year1_tor_clean["確診神經病變_1"]
            X_year1_tor_resampled, y_year1_tor_resampled = smote.fit_resample(X_year1_tor, y_year1_tor)
            year1_tor_clean = pd.concat([X_year1_tor_resampled, y_year1_tor_resampled], axis=1)

        if self.is_imbalanced(year1_ele_clean, label_column="確診神經病變_1"):
            print("Year1 資料不平衡，應用 SMOTE")
            smote = SMOTE(random_state=42)
            X_year1_ele = year1_ele_clean.drop(columns=["確診神經病變_1"])
            y_year1_ele = year1_ele_clean["確診神經病變_1"]
            X_year1_ele_resampled, y_year1_ele_resampled = smote.fit_resample(X_year1_ele, y_year1_ele)
            year1_ele_clean = pd.concat([X_year1_ele_resampled, y_year1_ele_resampled], axis=1)

        # 判斷是否需要平衡 year2
        if self.is_imbalanced(year2_tor_clean, label_column="確診神經病變_2"):
            print("Year2 資料不平衡，應用 SMOTE")
            smote = SMOTE(random_state=42)
            X_year2_tor = year2_tor_clean.drop(columns=["確診神經病變_2"])
            y_year2_tor = year2_tor_clean["確診神經病變_2"]
            X_year2_tor_resampled, y_year2_tor_resampled = smote.fit_resample(X_year2_tor, y_year2_tor)
            year2_tor_clean = pd.concat([X_year2_tor_resampled, y_year2_tor_resampled], axis=1)
        
        if self.is_imbalanced(year2_ele_clean, label_column="確診神經病變_2"):
            print("Year2 資料不平衡，應用 SMOTE")
            smote = SMOTE(random_state=42)
            X_year2_ele = year2_ele_clean.drop(columns=["確診神經病變_2"])
            y_year2_ele = year2_ele_clean["確診神經病變_2"]
            X_year2_ele_resampled, y_year2_ele_resampled = smote.fit_resample(X_year2_ele, y_year2_ele)
            year2_ele_clean = pd.concat([X_year2_ele_resampled, y_year2_ele_resampled], axis=1)

        return year1_tor_clean, year1_ele_clean, year2_tor_clean, year2_ele_clean

    def standardize(self):
        """標準化資料（跳過特定欄位）"""
        year1_tor_clean, year1_ele_clean, year2_tor_clean, year2_ele_clean = self.preprocess()

        # 不需要標準化的欄位
        skip_columns = ["確診神經病變_1", "確診神經病變_2"]

        # 初始化結果 DataFrame
        year1_tor_standardized = pd.DataFrame()
        year1_ele_standardized = pd.DataFrame()
        year2_tor_standardized = pd.DataFrame()
        year2_ele_standardized = pd.DataFrame()

        # Year1 的標準化處理
        scaler = StandardScaler()
        for col in year1_tor_clean.columns:
            if col in skip_columns:
                # 不進行標準化的欄位直接保留
                year1_tor_standardized[col] = year1_tor_clean[col]
            else:
                # 標準化處理
                year1_tor_standardized[col] = scaler.fit_transform(year1_tor_clean[[col]]).flatten()
        
        for col in year1_ele_clean.columns:
            if col in skip_columns:
                # 不進行標準化的欄位直接保留
                year1_ele_standardized[col] = year1_ele_clean[col]
            else:
                # 標準化處理
                year1_ele_standardized[col] = scaler.fit_transform(year1_ele_clean[[col]]).flatten()


        # Year2 的標準化處理
        for col in year2_tor_clean.columns:
            if col in skip_columns:
                # 不進行標準化的欄位直接保留
                year2_tor_standardized[col] = year2_tor_clean[col]
            else:
                # 標準化處理
                year2_tor_standardized[col] = scaler.fit_transform(year2_tor_clean[[col]]).flatten()

        for col in year2_ele_clean.columns:
            if col in skip_columns:
                # 不進行標準化的欄位直接保留
                year2_ele_standardized[col] = year2_ele_clean[col]
            else:
                # 標準化處理
                year2_ele_standardized[col] = scaler.fit_transform(year2_ele_clean[[col]]).flatten()

        return year1_tor_standardized, year1_ele_standardized, year2_tor_standardized, year2_ele_standardized

    def know(self):
        """
        對資料進行初步了解
        """
        # 讀取 Excel 檔案
        
        data = self.load_data()

        # 1. 資料基本資訊
        print("資料基本資訊:")
        print(data.info())

        # 2. 檢查空值
        print("\n每欄位空值總數:")
        print(data.isnull().sum())

        # 3. 檢查重複值
        print("\n重複列數量:", data.duplicated().sum())

        # 4. 基本統計資訊
        print("\n數值欄位的統計摘要:")
        print(data.describe())

        # 5. 檢查資料型別是否正確
        print("\n欄位資料型別:")
        print(data.dtypes)

        # 6. 異常值簡單檢查（舉例：檢查數值欄位是否有負數）
        numerical_columns = data.select_dtypes(include=["number"]).columns
        for col in numerical_columns:
            print(f"\n檢查數值欄位 {col} 是否有負值:")
            print(data[data[col] < 0])

        # 7. 日期格式檢查（如果有日期欄位）
        date_columns = data.select_dtypes(include=["datetime", "object"]).columns
        for col in date_columns:
            try:
                pd.to_datetime(data[col])  # 嘗試轉換為日期格式
                print(f"欄位 {col} 的日期格式檢查通過")
            except Exception as e:
                print(f"欄位 {col} 的日期格式可能有問題: {e}")

        # 8. 輸出報告（選擇性將結果存檔）
        output_path = "data_summary.xlsx"
        summary = pd.DataFrame({
            "欄位名稱": data.columns,
            "空值數量": data.isnull().sum(),
            "重複值": data.duplicated().sum(),
            "資料型別": data.dtypes
        })
        summary.to_excel(output_path, index=False)
        print(f"\n檢查報告已輸出到 {output_path}")
    
    def string_trans_array(self):
        """
        從Excel取得特定欄位的名稱，將輸出特定欄位的陣列
        利於後續進行處理
        
        """
        # 已經從Excel中確認欄位的名稱，複製並處理
        electrophysiological_data = "DML_Med_R_1,CMAP_Med_R_1,MNCV_Med_R_1,DML_Med_L_1,CMAP_Med_L_1,MNCV_Med_L_1,DML_Ula_R_1,CMAP_Ula_R_1,MNCV_Ula_R_1,DML_Ula_L_1,CMAP_Ula_L_1,MNCV_Ula_L_1,DML_Per_R_1,CMAP_Per_R_1,MNCV_Per_R_1,DML_Per_L_1,CMAP_Per_L_1,MNCV_Per_L_1,DML_Tib_R_1,CMAP_Tib_R_1,MNCV_Tib_R_1,DML_Tib_L_1,CMAP_Tib_L_1,MNCV_Tib_L_1,F_Med_R_1,F_Med_L_1,F_Ula_R_1,F_Ula_L_1,F_Per_R_1,F_Per_L_1,F_Tib_R_1,F_Tib_L_1,H_reflex_R_1,H_Reflex_L_1,SLO_Med_R_1,SNAP_Med_R_1,SNCV_Med_R_1,SLO_MP_R_1,SNAP_MP_R_1,SNCV_MP_R_1,SLO_Med_L_1,SNAP_Med_L_1,SNCV_Med_L_1,SLO_MP_L_1,SNAP_MP_L_1,SNCV_MP_L_1,SLO_Ula_R_1,SNAP_Ula_R_1,SNCV_Ula_R_1,SLO_Ula_L_1,SNAP_Ula_L_1,SNCV_Ula_L_1,SLO_Sur_R_1,SNAP_Sur_R_1,SNCV_Sur_R_1,SLO_Sur_L_1,SNAP_Sur_L_1,SNCV_Sur_L_1".split(",")
        toronto = "有主觀徵象_1,多倫多_痛_1,多倫多_麻_1,多倫多_刺_1,多倫多_無力_1,多倫多_協調_1,多倫多_UE_1,有客觀症狀_1,多倫多_刺鈍左_1,多倫多_刺鈍右_1,多倫多_溫鈍左_1,多倫多_溫鈍右_1,多倫多_碰鈍左_1,多倫多_碰鈍右_1,多倫多_振鈍左_1,多倫多_振鈍右_1,多倫多_JPS鈍左_1,多倫多_JPS鈍右_1,多倫多_knee左_1,多倫多_knee右_1,多倫多_ankle左_1,多倫多_ankle右_1".split(",")
        target = "確診神經病變_1"
        
        # 年欄位
        year1 = [electrophysiological_data,toronto,target]
        year2 = [[i.replace("_1","_2") for i in electrophysiological_data ],[ j.replace("_1","_2") for j in toronto],target.replace("_1","_2")]
        
        return year1 , year2
    
if __name__ == "__main__":
    # d = trainingData().standardize()
    d = trainingData().standardize()
    print(d)