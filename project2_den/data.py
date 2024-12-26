"""
資料處理步驟:
先以idcode、opdno分組後，在找'labit','labsh1it','labnmabv','labrefcval','labresuval'

1.缺失值找尋 save_result(output_NaN = True)
2.缺失值補上一筆的資料 save_result()
"""

import pandas as pd
import numpy as np
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
class Data:

    def save_result(self, output_NaN=False):
        """
        導入基本資料，並進行處理。
        """

        data_den = pd.read_csv("./data/den0.csv")
        data_flu = pd.read_csv("./data/flu0.csv")
        data_sep = pd.read_csv("./data/sep0.csv")
        data_gen0 = pd.read_csv("./data/gen0.csv")
        data_gen1 = pd.read_csv("./data/gen1.csv")

        # 集合所有資料為一個陣列
        all_data = [data_den, data_sep, data_flu, data_gen0, data_gen1]
        all_data_name = ['den_data', 'flu_data', 'sep_data', 'gen_data_0', 'gen_data_1']
        global all_data_sex; all_data_sex = []

        if output_NaN:
            for data_count in range(len(all_data)):
                self.__missing_value_report(all_data[data_count], all_data_name[data_count])
            return

        def process_data(data, name):
            if name in ['den_data', 'flu_data']:
                data['rcvdat'] = pd.to_datetime(data['rcvdat'], format='%Y%m%d', errors='coerce')
                data['rcvtm'] = pd.to_datetime(data['rcvtm'], format='%H%M', errors='coerce').dt.time
                data = self.__fill_missing_values_direct(data, 'opdno', 'rcvdat', 'rcvtm')
            else:
                data['cltdat'] = pd.to_datetime(data['cltdat'], format='%Y%m%d', errors='coerce')
                data['clttm'] = pd.to_datetime(data['clttm'], format='%H%M', errors='coerce').dt.time
                data = self.__fill_missing_values_direct(data, 'opdno', 'cltdat', 'clttm')

            data['sick_type'] = name

            biomarkers = {
                '72A001': 'WBC', '72B703': 'WBC', '72A015': 'Segment', '72-547': 'CRP',
                '72C015': 'Lymphocyte', '72I001': 'Platelets', '72D001': 'Hematocrit',
                '72-517': 'AST/GOT', '72-360': 'AST/GOT', '72-361': 'ALT/GPT',
                '72-518': 'ALT/GPT', '72-333': 'Creatinine', '72-505': 'Creatinine'
            }

            data = data[data['labsh1it'].isin(biomarkers.keys())]
            data['labsh1it'] = data['labsh1it'].map(biomarkers)

            def clean_labresuval(value):
                try:
                    if isinstance(value, str):
                        if '>' in value:
                            return float(value.replace('>', '').strip()) + 0.01
                        elif '<' in value:
                            return float(value.replace('<', '').strip()) - 0.01
                        elif '-' in value:
                            return float(value.split('-')[0].strip())
                    return float(value) if pd.notna(value) else np.nan
                except:
                    return np.nan

            data['labresuval'] = data['labresuval'].apply(clean_labresuval)

            mean_value = data['labresuval'].mean(skipna=True)
            data['labresuval'] = data['labresuval'].fillna(mean_value if not np.isnan(mean_value) else 0)

            biomarker_filter = ['Hematocrit', 'Platelets', 'WBC']
            pivot = data.pivot_table(index=['idcode', 'opdno'], columns='labsh1it', \
                                     values='labresuval', aggfunc='first', fill_value=0)

            # 在樞紐分析後重新添加 sick_type
            pivot = pivot.reset_index()
            pivot['sick_type'] = all_data_name.index(name)

            for biomarker in biomarker_filter:
                if biomarker in pivot.columns:
                    pivot = pivot[pivot[biomarker] >= 2]

            output_path = f"./data/result/{name}.csv"
            pivot.to_csv(output_path)

        with ThreadPoolExecutor() as executor:
            for i in range(len(all_data)):
                executor.submit(process_data, all_data[i], all_data_name[i])

        print("數據處理完成並保存結果。")

    def __missing_value_report(self, dataframe, name):
        """
        尋找並報告缺失值。
        """
        missing_indices = []
        for (idcode, opdno), group in dataframe.groupby(['idcode', 'opdno']):
            for col in group.columns:
                if group[col].isnull().any():
                    missing_rows = group[group[col].isnull()].index.tolist()
                    for row in missing_rows:
                        missing_indices.append({"idcode": idcode, "opdno": opdno, "Column": col, "Row": row})

        missing_df = pd.DataFrame(missing_indices)
        output_path = f"./data/missing/missing_{name}.csv"
        missing_df.to_csv(output_path, index=False)

    def load_result(self):
        """
        導入以處理好的資料
        """
        data_den = pd.read_csv("./data/result/den_data.csv")
        data_flu = pd.read_csv("./data/result/flu_data.csv")
        data_sep = pd.read_csv("./data/result/sep_data.csv")
        data_gen0 = pd.read_csv("./data/result/gen_data_0.csv")
        data_gen1 = pd.read_csv("./data/result/gen_data_1.csv")

        # 集合所有資料為一個陣列
        all_data = [data_den,data_flu,
                    data_sep,data_gen0,
                    data_gen1]

        return all_data

    def __missing_value_report(self, dataframe, name):
        """
        尋找並報告缺失值。
        """
        missing_indices = []
        for (idcode, opdno), group in dataframe.groupby(['idcode', 'opdno']):
            for col in group.columns:
                if group[col].isnull().any():
                    missing_rows = group[group[col].isnull()].index.tolist()
                    for row in missing_rows:
                        missing_indices.append({"idcode": idcode, "opdno": opdno, "Column": col, "Row": row})

        missing_df = pd.DataFrame(missing_indices)
        output_path = f"./data/missing/missing_{name}.csv"
        missing_df.to_csv(output_path, index=False)

    def __fill_missing_values_direct(self, dataframe, group_column, date_column, time_column):
            """
            填補缺失值。
            """
            dataframe['datetime'] = pd.to_datetime(
                dataframe[date_column].astype(str) + ' ' + dataframe[time_column].astype(str),
                errors='coerce'
            )

            if dataframe['datetime'].isnull().any():
                def process_group(group):
                    group = group.sort_values(by='datetime')
                    filled_rows = []
                    prev_rows = None

                    for _, row in group.iterrows():
                        if pd.isna(row['datetime']):
                            if prev_rows is not None:
                                for prev_row in prev_rows:
                                    new_row = prev_row.copy()
                                    new_row.update(row.to_dict())
                                    for labresuval_key in ['labresuval']:
                                        if pd.isna(new_row.get(labresuval_key, np.nan)):
                                            new_row[labresuval_key] = 0
                                    filled_rows.append(pd.Series(new_row))
                        else:
                            prev_rows = group[group['datetime'] == row['datetime']].to_dict('records')
                            filled_rows.append(row)

                    return pd.DataFrame(filled_rows)

                result = dataframe.groupby(group_column).apply(process_group).reset_index(drop=True)
            else:
                result = dataframe.copy()

            result.drop(columns=['datetime'], inplace=True)
            return result


    def test_test(self, output_NaN=False):
        """
        測試學姊的程式碼邏輯於自身
        """
        import os
        data_files = [
            "./data/den0.csv",
            "./data/flu0.csv",
            "./data/sep0.csv",
            "./data/gen0.csv",
            "./data/gen1.csv"
        ]
        data_names = ["den_data", "flu_data", "sep_data", "gen_data_0", "gen_data_1"]

        output_dir = "./data/result_tmp/"
        os.makedirs(output_dir, exist_ok=True)

        if output_NaN:
            # 缺失值報告
            for i, file in enumerate(data_files):
                data = pd.read_csv(file)
                self.__missing_value_report(data, data_names[i])
            return

        # 處理所有檔案
        for i, file in enumerate(data_files):
            print(f"處理檔案: {file}")
            data = pd.read_csv(file)

            # 日期與時間處理
            if 'rcvdat' in data.columns:
                data['ipdat'] = pd.to_datetime(data['rcvdat'], format='%Y%m%d', errors='coerce')
                data['cltdat'] = pd.to_datetime(data['rcvdat'], format='%Y%m%d', errors='coerce')
                data['clttm'] = pd.to_datetime(data['rcvtm'], format='%H%M', errors='coerce').dt.time
            else:
                data['ipdat'] = pd.to_datetime(data['ipdat'], format='%Y%m%d', errors='coerce')
                data['cltdat'] = pd.to_datetime(data['cltdat'], format='%Y%m%d', errors='coerce')
                data['clttm'] = pd.to_datetime(data['clttm'], format='%H%M', errors='coerce').dt.time

            # 檢測時間過濾: 3 天內
            data['days_diff'] = (data['cltdat'] - data['ipdat']).dt.days
            data = data[data['days_diff'] <= 3]

            # 缺失值處理: 前後補值
            data = data.sort_values(by=['idcode', 'opdno', 'cltdat', 'clttm'])
            data = data.groupby(['idcode', 'opdno']).apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

            # 生成透視表 (Pivot Table)
            pivot_table = data.pivot_table(
                index=["idcode", "opdno", "ipdat"],
                columns="labnmabv",
                values="labresuval",
                aggfunc="count",
                fill_value=0
            ).reset_index()

            # 篩選測量次數大於等於 2 的病患
            pivot_table = pivot_table[(pivot_table["WBC"] >= 2) | (pivot_table["Hematocrit"] >= 2)]

            # 保存結果
            output_file = os.path.join(output_dir, f"{data_names[i]}_processed.csv")
            pivot_table.to_csv(output_file, index=False)
            print(f"結果已保存: {output_file}")

if __name__ == "__main__":
    Data().save_result()
    # Data().test_test()