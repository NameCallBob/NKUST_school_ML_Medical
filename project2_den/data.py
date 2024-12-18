"""
資料處理步驟:
先以idcode、opdno分組後，在找'labit','labsh1it','labnmabv','labrefcval','labresuval'

1.缺失值找尋 save_result(output_NaN = True)
2.缺失值補上一筆的資料 save_result()
"""

import pandas as pd
import re

class Data:

    def save_result(self , output_NaN = False):
        """
        導入基本資料
        """

        data_den = pd.read_csv("./data/den0.csv")
        data_flu = pd.read_csv("./data/flu0.csv")
        data_sep = pd.read_csv("./data/sep0.csv")
        data_gen0 = pd.read_csv("./data/gen0.csv")
        data_gen1 = pd.read_csv("./data/gen1.csv")

        # 集合所有資料為一個陣列
        all_data = [data_den,data_sep,
                    data_flu,data_gen0,
                    data_gen1]
        # name
        all_data_name = [
            'den_data','flu_data','sep_data',
            'gen_data_0','gen_data_1'
        ]
        global all_data_sex ; all_data_sex = []

        if output_NaN :
            for data_count in range(len(all_data)):
                self.__missing_value_report(all_data[data_count],
                                            all_data_name[data_count])
            return

        # den的資料與其他欄位不相同，在第一個迴圈需特殊處理
        for data_count in range(len(all_data)):
            if data_count in [0,1]  :
                all_data_sex.append(all_data[data_count].set_index("opdno")["sex"].to_dict())
                # 將日期時間轉為pandas可處理型態
                all_data[data_count]['rcvdat'] = pd.to_datetime(all_data[data_count]['rcvdat'], format='%Y%m%d', errors='coerce')
                all_data[data_count]['rcvtm'] = pd.to_datetime(all_data[data_count]['rcvtm'], format='%H%M', errors='coerce').dt.time
                # 若有缺失值則補此病患上一筆資料的紀錄
                all_data[data_count] =  self.__fill_missing_values_direct(
                                    all_data[data_count],
                                    'opdno',
                                    'rcvdat',
                                    'rcvtm',
                                    )


            else:
                # 將日期時間轉為pandas可處理型態
                all_data[data_count]['cltdat'] = pd.to_datetime(all_data[data_count]['cltdat'], format='%Y%m%d', errors='coerce')
                all_data[data_count]['clttm'] = pd.to_datetime(all_data[data_count]['clttm'], format='%H%M', errors='coerce').dt.time
                # 若有缺失值則補此病患上一筆資料的分組的所有紀錄
                all_data[data_count] =  self.__fill_missing_values_direct(
                                    all_data[data_count],
                                    'opdno',
                                    'cltdat',
                                    'clttm',
                                    )
                all_data[data_count] = self.add_gender_column(
                    all_data[data_count],'opdno'
                )

            # 添加疾病類別
            all_data[data_count]['sick_type'] = data_count

            # 增加是否正常的判別欄位
            all_data[data_count] = self.add_normal_flag(
                all_data[data_count], 'labrefcval', 'labresuval','sex')
            # save
            all_data[data_count].to_csv(f"./data/result/{all_data_name[data_count]}.csv")

        return all_data

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

    def add_normal_flag(self, data, condition_col, value_col, gender_col=None, age_col=None):
        """
        根據條件欄位、性別欄位與測量值欄位新增判別是否正常的欄位。
        :param data: 資料集 DataFrame
        :param condition_col: 條件欄位名稱
        :param value_col: 測量值欄位名稱
        :param gender_col: 性別欄位名稱 (可選)
        :param age_col: 年齡欄位名稱 (可選)
        :return: 每行的 is_normal 和 error_reason 結果列表
        """
        def is_normal(row):
            error_reason = None

            def clean_condition(condition):
                if pd.isna(condition):
                    return None  # 無效條件返回 None
                if isinstance(condition, str):
                    condition = condition.replace('�Ø', '').replace('Ø', '').strip()
                    condition = re.sub(r'[^\w\s<>\-:~,.]', '', condition)  # 移除無效字符
                return condition

            def clean_values(value):
                if isinstance(value, str):
                    value = value.strip()
                    if '<' in value:
                        try:
                            threshold = float(value.replace('<', '').strip())
                            return threshold * 0.99
                        except ValueError:
                            return None
                    elif '>' in value:
                        try:
                            threshold = float(value.replace('>', '').strip())
                            return threshold * 1.01
                        except ValueError:
                            return None
                try:
                    return float(value)
                except ValueError:
                    return None

            def parse_range_condition(condition , mark="-"):
                try:
                    lower, upper = map(float, condition.split(mark))
                    return lower, upper
                except ValueError:
                    return None, None

            def parse_gender_age_condition(condition):
                matches = re.findall(r'(M|F)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', condition)
                parsed_conditions = []
                for match in matches:
                    try:
                        gender = match[0]
                        lower = float(match[1])
                        upper = float(match[2])
                        parsed_conditions.append({'gender': gender, 'range': (lower, upper)})
                    except ValueError:
                        print(f"解析失敗：{match}")
                return parsed_conditions

            def parse_gender_value_condition(condition):
                """
                支援性別與範圍條件的解析，例如 "M:0.64~1.27,F:0.44~1.03"。
                返回格式：
                [{'gender': 'M', 'range': (0.64, 1.27)}, {'gender': 'F', 'range': (0.44, 1.03)}]
                """
                matches = re.findall(r'([MF]):([\d.]+)~([\d.]+)', condition)
                parsed_conditions = []
                for match in matches:
                    gender = match[0]
                    lower = float(match[1])
                    upper = float(match[2])
                    parsed_conditions.append({'gender': gender, 'range': (lower, upper)})
                return parsed_conditions

            def extract_numbers_from_condition(condition):
                condition = clean_condition(condition)
                try:
                    matches = re.findall(r'([<>])?\s*(\d+(\.\d+)?)', condition)
                    numbers = [float(match[1]) for match in matches]
                    return numbers
                except ValueError:
                    return []

            def parse_mixed_condition(condition):
                """
                提取混合條件中的數值。
                例如: "<5;CADrisk:<1.0Low,>3.0Hi." -> [{'operator': '<', 'value': 5.0}, {'operator': '<', 'value': 1.0}, {'operator': '>', 'value': 3.0}]
                """
                matches = re.findall(r'([<>])\s*(\d+(\.\d+)?)', condition)
                parsed_conditions = []
                for match in matches:
                    operator = match[0]
                    value = float(match[1])
                    parsed_conditions.append({'operator': operator, 'value': value})
                return parsed_conditions

            # 主邏輯
            condition = row[condition_col]
            value = row[value_col]
            gender = row[gender_col] if gender_col else None
            age = row[age_col] if age_col else None

            if pd.isna(condition) or pd.isna(value):
                error_reason = "缺失值錯誤"
                print(f"Row {row.name}: {error_reason} {condition}")
                return None

            condition = clean_condition(condition)
            value = clean_values(value)

            if value is None:
                error_reason = "測量值格式錯誤"
                print(f"Row {row.name}: {error_reason} {condition}")
                return None

            value = float(value)

            if gender_col and ':' in condition:
                gender_conditions = parse_gender_value_condition(condition)
                matched = False
                max_upper = None

                for cond in gender_conditions:
                    lower, upper = cond['range']
                    max_upper = max(max_upper or upper, upper)

                    if cond['gender'] == gender:
                        matched = True
                        if lower <= value <= upper:
                            return 1  # 正常

                    # 若性別不匹配，則比較最大值
                    if not matched and max_upper is not None:
                        return 1 if value <= max_upper else 0

            # 性別條件判斷
            if gender_col and any(char in condition for char in ['M', 'F']):
                gender_conditions = parse_gender_age_condition(condition)
                matched = False
                max_upper = None

                for cond in gender_conditions:
                    lower, upper = cond['range']
                    max_upper = max(max_upper or upper, upper)

                    if cond['gender'] == gender:
                        matched = True
                        if lower <= value <= upper:
                            return 1  # 正常

                # 若性別不匹配，則比較最大值
                if not matched and max_upper is not None:
                    return 1 if value <= max_upper else 0

            if '-' in condition:
                lower, upper = parse_range_condition(condition)
                if lower is None or upper is None:
                    error_reason = "範圍格式錯誤"
                    return None
                return 1 if lower <= value <= upper else 0
            elif '~' in condition:
                lower, upper = parse_range_condition(condition,'~')
                if lower is None or upper is None:
                    error_reason = "範圍格式錯誤"
                    return None
                return 1 if lower <= value <= upper else 0

            # 處理混合條件
            if '<' in condition and '>' in condition:
                parsed_conditions = parse_mixed_condition(condition)
                for cond in parsed_conditions:
                    if cond['operator'] == '<' and not value < cond['value']:
                        return 0  # 測量值不小於條件中的數值
                    if cond['operator'] == '>' and not value > cond['value']:
                        return 0  # 測量值不大於條件中的數值
                return 1  # 若所有條件均滿足，返回正常

            elif '<' in condition:
                try:
                    threshold = float(condition.replace('<', '').strip())
                    return 1 if value < threshold else 0
                except ValueError:
                    error_reason = "小於條件格式錯誤"
                    print(f"Row {row.name} {value}: {error_reason} {condition}")
                    return None

            elif '>' in condition:
                try:
                    threshold = float(condition.replace('>', '').strip())
                    return 1 if value > threshold else 0
                except ValueError:
                    error_reason = "大於條件格式錯誤"
                    print(f"Row {row.name} {value}: {error_reason} {condition}")
                    return None

            # 提取條件中的數值並判斷
            numbers = extract_numbers_from_condition(condition)
            if not numbers:
                print(f"Row {row.name} {value}: 無法提取條件中的數值{condition}")
                return None

            # 判斷邏輯：測量值是否大於提取的最大數值
            max_number = max(numbers)
            return 1 if value > max_number else 0

        # 遍歷行數據
        data['is_normal'] = data.apply(is_normal, axis=1)
        return data

    def __missing_value_report(self,dataframe,name):
        # 找出有缺失值的欄位及列
        missing_indices = []

        # 逐組檢查每個分組中的缺失值
        for (idcode, opdno), group in dataframe.groupby(['idcode', 'opdno']):
            for col in group.columns:
                if group[col].isnull().any():
                    missing_rows = group[group[col].isnull()].index.tolist()
                    for row in missing_rows:
                        missing_indices.append({
                            "idcode": idcode,
                            "opdno": opdno,
                            "Column": col,
                            "Row": row
                        })

        # 轉換為DataFrame並保存為CSV
        missing_df = pd.DataFrame(missing_indices)
        output_path = f"./data/missing/missing_{name}.csv"
        missing_df.to_csv(output_path, index=False)

    def analyze(self):
        pass

    def __fill_missing_values_direct(self, dataframe, group_column, date_column, time_column):
        """
        直接填補缺失值，若某時間點缺失，則用上一個時間的整組數據填補缺失處。

        Args:
            dataframe (pd.DataFrame): 數據框架。
            group_column (str): 分組的列名，例如 'opdno'。
            date_column (str): 日期列名，例如 'rcvdat'。
            time_column (str): 時間列名，例如 'rcvtm'。

        Returns:
            pd.DataFrame: 填補缺失值後的數據框架。
        """
        # 合併日期和時間為 datetime
        dataframe['datetime'] = pd.to_datetime(
            dataframe[date_column].astype(str) + ' ' + dataframe[time_column].astype(str),
            errors='coerce'
        )

        # 定義處理每組的邏輯
        def process_group(group):
            # 排序數據，確保按時間順序處理
            group = group.sort_values(by='datetime')
            filled_rows = []  # 存放補值後的結果
            prev_rows = None  # 保存上一個有效時間點的所有記錄

            for _, row in group.iterrows():
                if pd.isna(row['datetime']):
                    # 若時間缺失，複製上一個時間的所有數據
                    if prev_rows is not None:
                        for prev_row in prev_rows:
                            # 複製上一筆數據並更新當前行
                            new_row = prev_row.copy()
                            new_row.update(row.to_dict())  # 用當前行的其他列更新
                            filled_rows.append(pd.Series(new_row))
                else:
                    # 保存當前時間點的所有記錄作為補值來源
                    prev_rows = group[group['datetime'] == row['datetime']].to_dict('records')
                    filled_rows.append(row)

            # 返回補值後的 DataFrame
            return pd.DataFrame(filled_rows)

        # 分組應用補值邏輯
        result = dataframe.groupby(group_column).apply(process_group).reset_index(drop=True)

        # 刪除輔助列，還原結構
        result.drop(columns=['datetime'], inplace=True)
        return result

    def __fill_missing_values_by_hour(self,dataframe, group_column, date_column, time_column, target_columns):
        """
        生成每小時的時間序列，並根據分組填補缺失值。

        Args:
            dataframe (pd.DataFrame): 數據框架。
            group_column (str): 分組的列名，例如 'opdno'。
            date_column (str): 日期列名，例如 'rcvdat'。
            time_column (str): 時間列名，例如 'rcvtm'。
            target_columns (list): 需要補值的目標列名，例如 ['value']。

        Returns:
            pd.DataFrame: 補值後的完整數據框架。
        """
        # 合併日期和時間為 datetime
        dataframe['datetime'] = pd.to_datetime(
            dataframe[date_column].astype(str) + ' ' + dataframe[time_column].astype(str),
            errors='coerce'
        )

        # 定義處理每組的邏輯
        def process_group(group):
            # 排序數據，確保按時間處理，並移除重複的時間戳
            group = group.sort_values(by='datetime').drop_duplicates(subset='datetime')

            # 檢查是否可以生成時間範圍
            start_time = group['datetime'].min()
            end_time = group['datetime'].max()
            if pd.isna(start_time) or pd.isna(end_time):
                # 若無法生成時間範圍，直接返回空數據
                return pd.DataFrame(columns=group.columns)

            # 生成完整的每小時時間序列
            time_range = pd.date_range(start=start_time, end=end_time, freq='H')
            # 重建 DataFrame 以完整時間序列為索引
            group = group.set_index('datetime').reindex(time_range)
            group.index.name = 'datetime'
            # 填補分組名稱
            group[group_column] = group[group_column].iloc[0]
            # 填補目標列的缺失值
            group[target_columns] = group[target_columns].fillna(method='ffill')
            return group.reset_index()

        # 分組應用補值邏輯
        result = dataframe.groupby(group_column).apply(process_group).reset_index(drop=True)

        # 還原日期與時間列
        result[date_column] = result['datetime'].dt.date
        result[time_column] = result['datetime'].dt.time
        result.drop(columns=['datetime'], inplace=True)

        return result

    def add_gender_column(self , dataframe, opdno_column):
        """
        根據 opdno 添加性別欄位，並使用字典緩存性別數據。

        Args:
            dataframe (pd.DataFrame): 原始資料框。
            opdno_column (str): opdno 所在的欄位名稱。

        Returns:
            pd.DataFrame: 添加性別欄位後的資料框。
        """
        # 初始化性別欄位
        dataframe['sex'] = None

        # 遍歷所有 opdno
        for index, row in dataframe.iterrows():
            opdno = row[opdno_column]

            # 檢查性別是否已在緩存中
            for gender_cache in all_data_sex:
                if opdno in gender_cache:
                    gender = gender_cache[opdno]
                else:
                    gender_cache[opdno] = "O"
                    gender = "O"

                # 設置性別欄位
                dataframe.at[index, 'sex'] = gender

        return dataframe

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
    # Data().save_result()
    Data().test_test()