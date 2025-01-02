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
        此程式碼是第一個測試，改用第二！
        導入原始資料，進行缺失值補前一行的資料並依照測量次數及天數輸出結果
        但與save_result_1的差別為，只會輸出一筆，代表若該次可能會有兩筆，另一筆將會被捨去。
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

    def save_result_1(self, output_NaN=False):
        """
        此程式碼是新的！
        導入原始資料，進行缺失值補前一行的資料並依照測量次數及天數輸出結果
        若有多值，將會用分號進行分隔，後續要在進行處理
        """
        from threading import Thread
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
            # 日期時間處理
            if name in ['den_data', 'flu_data']:
                data['rcvdat'] = pd.to_datetime(data['rcvdat'], format='%Y%m%d', errors='coerce')
                data['rcvtm'] = pd.to_datetime(data['rcvtm'], format='%H%M', errors='coerce').dt.time
                data = self.__fill_missing_values_direct(data, 'opdno', 'rcvdat', 'rcvtm')
            else:
                data['cltdat'] = pd.to_datetime(data['cltdat'], format='%Y%m%d', errors='coerce')
                data['clttm'] = pd.to_datetime(data['clttm'], format='%H%M', errors='coerce').dt.time
                data = self.__fill_missing_values_direct(data, 'opdno', 'cltdat', 'clttm')

            data['sick_type'] = name

            # 篩選 biomarkers
            biomarkers = {
                '72A001': 'WBC', '72B703': 'WBC', '72A015': 'Segment', '72-547': 'CRP',
                '72C015': 'Lymphocyte', '72I001': 'Platelets', '72D001': 'Hematocrit',
                '72-517': 'AST/GOT', '72-360': 'AST/GOT', '72-361': 'ALT/GPT',
                '72-518': 'ALT/GPT', '72-333': 'Creatinine', '72-505': 'Creatinine'
            }

            data = data[data['labsh1it'].isin(biomarkers.keys())]
            data['labsh1it'] = data['labsh1it'].map(biomarkers)

            # 數值清理
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

            # 過濾特定 biomarkers
            biomarker_filter = ['Hematocrit', 'Platelets', 'WBC']
            data_filtered = data[data['labsh1it'].isin(biomarker_filter)]

            # 重塑數據
            if data_filtered.duplicated(subset=['idcode', 'opdno', 'labsh1it']).any():
                data_filtered = data_filtered.drop_duplicates(subset=['idcode', 'opdno', 'labsh1it'])

            expanded_data = data_filtered.pivot_table(
                index=['idcode', 'opdno'],
                columns=['labsh1it'],
                values='labresuval',
                aggfunc=list  # 聚合重複值為列表
            ).reset_index()

            # 展開列表
            expanded_data = expanded_data.apply(
                lambda col: col.explode() if col.name in biomarker_filter and col.apply(type).eq(list).any() else col
            )

            # 增加 sick_type 欄位
            expanded_data['sick_type'] = all_data_name.index(name)

            output_path = f"./data/result/{name}_expanded.csv"
            expanded_data.to_csv(output_path, index=False)


        threads = []

        for i in range(len(all_data)):
            thread = Thread(target=process_data, args=(all_data[i], all_data_name[i]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

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
        導入已處理好的資料，並將含分號的欄位展開，同時將除 idcode, opdno, ipdat 之外的欄位都轉成 float。
        """
        data_den = pd.read_csv("./data/result_tmp/den_data_expanded.csv")
        data_flu = pd.read_csv("./data/result_tmp/flu_data_expanded.csv")
        data_sep = pd.read_csv("./data/result_tmp/sep_data_expanded.csv")
        data_gen0 = pd.read_csv("./data/result_tmp/gen_data_0_expanded.csv")
        data_gen1 = pd.read_csv("./data/result_tmp/gen_data_1_expanded.csv")

        # 集合所有資料為一個 list
        all_data = [data_den, data_flu, data_sep, data_gen0, data_gen1]

        # 對每個 DataFrame 做展開 + 型別轉換
        expanded_data_list = []

        for i in range(len(all_data)):
            df = all_data[i]
            # 1) 先展開含分號的欄位
            expanded_df = self.expand_semicolon_rows(df,i)

            # 2) 將除 idcode, opdno, ipdat 外的欄位全部轉為 float
            keep_cols = ["idcode", "opdno", "ipdat"]
            for col in expanded_df.columns:
                if col not in keep_cols:
                    expanded_df[col] = pd.to_numeric(expanded_df[col], errors="coerce")

            expanded_data_list.append(expanded_df)

        return expanded_data_list

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

    def teacher_example(self, output_NaN=False):
        """
        測試學姊的程式碼邏輯於自身
        計算每個biomarker測量次數
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

    def expand_semicolon_rows(self, df: pd.DataFrame, sick_type) -> pd.DataFrame:
        """
        將 df 中每一列如遇到分號 ';'，展開成多列。
        若某欄位不含分號(或分割後長度較少)，則以最後一個值補齊。
        """
        import pandas as pd

        expanded_rows = []

        for _, row in df.iterrows():
            split_values = {}
            max_len = 1  # 此列最大的分割長度 (要展開成幾列)

            # 先將每個欄位都 split(";")，並找出最長的分割長度
            for col in df.columns:
                val_str = str(row[col]) if pd.notnull(row[col]) else ""  # 處理缺失值
                parts = val_str.split(";")
                split_values[col] = parts
                if len(parts) > max_len:
                    max_len = len(parts)

            # 依照 max_len，展開此列
            for i in range(max_len):
                new_row = {}
                for col in df.columns:
                    col_parts = split_values[col]
                    if i < len(col_parts):
                        new_row[col] = col_parts[i]
                    else:
                        # 若不足 i+1 筆，使用該欄位的最後一筆值填補
                        new_row[col] = col_parts[-1]
                expanded_rows.append(new_row)

        # 組成新的 DataFrame，保持欄位順序一致
        expanded_df = pd.DataFrame(expanded_rows, columns=df.columns)

        # 添加 sick_type 欄位
        expanded_df['sick_type'] = sick_type

        return expanded_df

if __name__ == "__main__":
    ob = Data()
    ob.save_result_1()