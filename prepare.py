"""
主要目標如下：
第一年 預測 第一年
第二年 預測 第二年
第一年 預測 第二年

主要資料欄位：
電生理檢查
1.DML_Med_R_1,CMAP_Med_R_1,MNCV_Med_R_1,DML_Med_L_1,CMAP_Med_L_1,MNCV_Med_L_1,DML_Ula_R_1,CMAP_Ula_R_1,MNCV_Ula_R_1,DML_Ula_L_1,CMAP_Ula_L_1,MNCV_Ula_L_1,DML_Per_R_1,CMAP_Per_R_1,MNCV_Per_R_1,DML_Per_L_1,CMAP_Per_L_1,MNCV_Per_L_1,DML_Tib_R_1,CMAP_Tib_R_1,MNCV_Tib_R_1,DML_Tib_L_1,CMAP_Tib_L_1,MNCV_Tib_L_1,F_Med_R_1,F_Med_L_1,F_Ula_R_1,F_Ula_L_1,F_Per_R_1,F_Per_L_1,F_Tib_R_1,F_Tib_L_1,H_reflex_R_1,H_Reflex_L_1,SLO_Med_R_1,SNAP_Med_R_1,SNCV_Med_R_1,SLO_MP_R_1,SNAP_MP_R_1,SNCV_MP_R_1,SLO_Med_L_1,SNAP_Med_L_1,SNCV_Med_L_1,SLO_MP_L_1,SNAP_MP_L_1,SNCV_MP_L_1,SLO_Ula_R_1,SNAP_Ula_R_1,SNCV_Ula_R_1,SLO_Ula_L_1,SNAP_Ula_L_1,SNCV_Ula_L_1,SLO_Sur_R_1,SNAP_Sur_R_1,SNCV_Sur_R_1,SLO_Sur_L_1,SNAP_Sur_L_1,SNCV_Sur_L_1
多倫多量表
2.有主觀徵象_1,多倫多_痛_1,多倫多_麻_1,多倫多_刺_1,多倫多_無力_1,多倫多_協調_1,多倫多_UE_1,有客觀症狀_1,多倫多_刺鈍左_1,多倫多_刺鈍右_1,多倫多_溫鈍左_1,多倫多_溫鈍右_1,多倫多_碰鈍左_1,多倫多_碰鈍右_1,多倫多_振鈍左_1,多倫多_振鈍右_1,多倫多_JPS鈍左_1,多倫多_JPS鈍右_1,多倫多_knee左_1,多倫多_knee右_1,多倫多_ankle左_1,多倫多_ankle右_1
主要預測欄位：
是否有神經病變_確診神經病變
"""

import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data import trainingData


from matplotlib import rc

# NOTE:顏總你電腦室windows記得調整成Microsogt JhengHei
# rc('font', family='Heiti TC')  # "PingFang TC" 是繁體中文版本
rc('font', family='Microsoft JhengHei')

class prepare:
    """
    挑選特徵及分割訓練和驗證集資料
    """

    def __init__(self):
        # 未經過處理的資料
        ob = trainingData()
        self.origin_data =  ob.preprocess()
        self.standard_data = ob.standardize()
        self.origin_data = {
            "year1_ele": self.origin_data[0],
            "year1_tor": self.origin_data[1],
            "year2_ele": self.origin_data[2],
            "year2_tor": self.origin_data[3],
        }

    def getTrainingData(self,
                        use_all_feature = True,
                        standard=False,
                        year=1,
                        data_type="ele",  # 資料類型：'ele' 或 'tor'
                        test_size=0.2,
                        random_state=42,
                        common_features=None,):
        """
        取得訓練資料

        @params standard 是否標準化
        @params test_size 驗證集大小
        @params random_state 隨機種子
        @params year 有關資料集的年編號
        @params data_type: 選擇 'ele' 或 'tor' 資料
        @params common_features: 交集的特徵列表
        return X_train, X_test, y_train, y_test
        """


        # 選擇資料集
        data = self.standard_data if standard else self.origin_data
        # print(f'data:{data}')
        key = f'year{year}_{data_type}'

        if key not in data:
            raise ValueError(f"資料集中沒有鍵 {key}，請確認年份和資料類型參數是否正確！")

        selected_data = data[key]

        # 動態決定 '_1' 或 '_2' 的後綴
        def add_suffix_based_on_year(features, year):
            suffix = f"_{year}"  # 如果 year == 1，則加上 '_1'；如果是 year == 2，則加上 '_2'
            return [f"{feature}{suffix}" for feature in features]

        if common_features:     # 假設有加入交集特徵，就取得交集特徵的欄位
            common_features_with_suffix = add_suffix_based_on_year(common_features, year)
            features = selected_data[common_features_with_suffix]

        elif use_all_feature:
            features = selected_data.drop(columns=[f"確診神經病變_{year}"])
        else:
            if data_type == "ele":
                features_selected = ['H_Reflex_L', 'SNCV_Sur_R', 'F_Tib_L', 'MNCV_Tib_L', 'SLO_Sur_L', 'SLO_Sur_R', 'SNAP_Sur_L', 'CMAP_Tib_L', 'F_Tib_R', 'SNAP_Sur_R', 'SNAP_Ula_R']
            elif data_type == "tor":
                features_selected = ['有主觀徵象', '多倫多_knee左', '多倫多_碰鈍右', '多倫多_knee右', '多倫多_ankle左', '多倫多_刺鈍左', '多倫多_JPS鈍左', '多倫多_ankle右', '多倫多_刺', '多倫多_麻', '多倫多_JPS鈍右', '多倫多_振鈍左']
            # elif data_type == "combined":
            #     features_selected = ['有主觀徵象', '多倫多_knee左', '多倫多_碰鈍右', '多倫多_knee右', '多倫多_ankle左', '多倫多_刺鈍左', '多倫多_JPS鈍左', '多倫多_ankle右', '多倫多_刺', '多倫多_麻', '多倫多_JPS鈍右', '多倫多_振鈍左','H_Reflex_L', 'SNCV_Sur_R', 'F_Tib_L', 'MNCV_Tib_L', 'SLO_Sur_L', 'SLO_Sur_R', 'SNAP_Sur_L', 'CMAP_Tib_L', 'F_Tib_R', 'SNAP_Sur_R', 'SNAP_Ula_R']
            else:
                raise KeyError("請輸入ele或tor")
            features = selected_data[features_selected]

        target = selected_data[f"確診神經病變_{year}"]

        # 分割資料集
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def feature_importance(self, X_train, y_train, year, type):
        """
        計算特徵重要性，輸出所有特徵圖表、前 15 個重要特徵及高於 4% 的特徵。

        @params X_train 訓練集特徵
        @params y_train 訓練集目標變數

        return 高於 4% 特徵的重要性 DataFrame，及特徵名稱陣列
        """
        # 使用隨機森林模型進行特徵重要性分析
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        # 獲取特徵重要性
        importances = model.feature_importances_
        feature_names = X_train.columns

        # 創建特徵重要性 DataFrame
        importance_df = pd.DataFrame({
            "特徵名稱": feature_names,
            "重要性": importances
        }).sort_values(by="重要性", ascending=False)

         # 1. 顯示所有特徵的重要性圖表
        num_features = len(importance_df)
        plt.figure(figsize=(10, 0.3 * num_features))  # 動態調整圖表高度
        plt.barh(importance_df["特徵名稱"], importance_df["重要性"], color='skyblue')
        plt.xlabel("特徵重要性")
        plt.ylabel("特徵名稱")
        plt.title(f"所有特徵的重要性 ，第{year}年 {type}")
        plt.gca().invert_yaxis()  # 翻轉 Y 軸
        plt.tight_layout()  # 自動調整以避免文字重疊
        # plt.show()

        # 2. 輸出前 15 個重要特徵的圖表
        top_15_features = importance_df.head(15)
        plt.figure(figsize=(10, 6))
        plt.barh(top_15_features["特徵名稱"], top_15_features["重要性"], color='orange')
        plt.xlabel("特徵重要性")
        plt.ylabel("特徵名稱")
        plt.title(f"前 15 個特徵的重要性，第{year}年 {type}" )
        plt.gca().invert_yaxis()  # 翻轉 Y 軸
        plt.tight_layout()
        # plt.show()
        print("輸出前 15 個重要特徵 :")
        print(top_15_features["特徵名稱"])
        # 3. 篩選出高於 4% 的特徵（調整閾值以適應數據）
        threshold = 0.04
        filtered_features = importance_df[importance_df["重要性"] >= threshold]

        if not filtered_features.empty:
            # 繪製高於 4% 的特徵圖表
            plt.figure(figsize=(10, 0.3 * len(filtered_features)))
            plt.barh(filtered_features["特徵名稱"], filtered_features["重要性"], color='lightgreen')
            plt.xlabel("特徵重要性")
            plt.ylabel("特徵名稱")
            plt.title(f"特徵重要性 ≥ {threshold * 100}% ，第{year}年 {type}" )
            plt.gca().invert_yaxis()  # 翻轉 Y 軸
            plt.tight_layout()
            # plt.show()

            # 返回高於 4% 的特徵名稱列表
            high_importance_features = filtered_features["特徵名稱"].tolist()
        else:
            print(f"沒有特徵的重要性高於 {threshold * 100}%。")
            high_importance_features = []

        # 確認結果輸出
        print(f"\n高於 {threshold * 100}% 的重要特徵名稱：")
        print(high_importance_features)

        return filtered_features, high_importance_features,top_15_features['特徵名稱']

    def run(self):
        all_important_feature=[] # 儲存所有年分與量表的重要特徵 [year1_ele, year1_tor, year2_ele, year2_tor]
        top15_feature = []
        for year in [1, 2]:
            for data_type in ["ele", "tor"]:
                print(f"處理第 {year} 年的 {data_type} 資料")
                X_train, X_test, y_train, y_test = self.getTrainingData(year=year, data_type=data_type)
                feature = self.feature_importance(X_train, y_train, year, data_type)
                all_important_feature.append(feature[1])
                top15_feature.append(feature[2])

        # 清理特徵名稱，去掉 "_1" 和 "_2"
        def clean_feature_name(feature):
            return re.sub(r'(_1|_2)$', '', feature)

        # 將 all_important_feature 中的每個特徵名稱進行清理
        cleaned_all_important_feature = [
            [clean_feature_name(feature) for feature in all_important_feature[0]],  # 第一年的 ele
            [clean_feature_name(feature) for feature in all_important_feature[1]],  # 第一年的 tor
            [clean_feature_name(feature) for feature in all_important_feature[2]],  # 第二年的 ele
            [clean_feature_name(feature) for feature in all_important_feature[3]],  # 第二年的 tor
        ]

        # 使用 set 計算第一年和第二年 ele 特徵的交集
        ele_common = set(cleaned_all_important_feature[0]) & set(cleaned_all_important_feature[2])

        # 使用 set 計算第一年和第二年 tor 特徵的交集
        tor_common = set(cleaned_all_important_feature[1]) & set(cleaned_all_important_feature[3])

        # 輸出結果
        print("高相關:")
        print("共同的 ele 特徵：", ele_common)
        print("共同的 tor 特徵：", tor_common)

        # 將 top15_feature 中的每個特徵名稱進行清理
        cleaned_all_important_feature = [
            [clean_feature_name(feature) for feature in top15_feature[0]],  # 第一年的 ele
            [clean_feature_name(feature) for feature in top15_feature[1]],  # 第一年的 tor
            [clean_feature_name(feature) for feature in top15_feature[2]],  # 第二年的 ele
            [clean_feature_name(feature) for feature in top15_feature[3]],  # 第二年的 tor
        ]

        # 使用 set 計算第一年和第二年 ele 特徵的交集
        ele_common = set(cleaned_all_important_feature[0]) & set(cleaned_all_important_feature[2])

        # 使用 set 計算第一年和第二年 tor 特徵的交集
        tor_common = set(cleaned_all_important_feature[1]) & set(cleaned_all_important_feature[3])

        # 輸出結果
        print("前15個：")
        print("共同的 ele 特徵：", ele_common)
        print("共同的 tor 特徵：", tor_common)

        # 將交集特徵傳遞給 getTrainingData 方法
        # for year in [1, 2]:
        #     for data_type in ["ele", "tor"]:
        #         print(f"處理第 {year} 年的 {data_type} 資料，使用交集特徵")
        #         if data_type == "ele":
        #             common_features = list(ele_common)
        #         else:
        #             common_features = list(tor_common)

        #         X_train, X_test, y_train, y_test = self.getTrainingData(
        #             year=year,
        #             data_type=data_type,
        #             common_features=common_features  # 傳遞交集特徵
        #         )

if __name__ == "__main__":
    data= prepare().run()
