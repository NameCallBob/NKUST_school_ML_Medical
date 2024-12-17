if __name__ == "__main__":
    from prepare import prepare
    import joblib
    import os
    import pandas as pd

    # 模型儲存路徑
    model_save_dir = "./result/model/"

    data_type = 'ele'  # 資料類型（tor 或 ele）

    # 載入預測資料
    prepare_instance = prepare()
    testdata_selected = prepare_instance.getTestData(data_type)

    # 模型名稱
    model_name = "XGB"  # 你想使用的模型名稱，例如 "RF", "XGB", "ADA"
    model_file_path = os.path.join(model_save_dir, f"{model_name}_{data_type}_1.joblib")

    if os.path.exists(model_file_path):
        print(f"從 {model_file_path} 加載 {model_name} 模型...")
        # 加載儲存的模型
        model = joblib.load(model_file_path)

        # 預測
        print(f"使用 {model_name} 模型進行逐筆預測...")

        for index, row in testdata_selected.iterrows():
            # 進行單筆資料預測
            single_prediction = model.predict(row.values.reshape(1, -1))[0]

            # 顯示預測結果
            print(f"第 {index + 1} 筆資料預測結果: {single_prediction}")
    else:
        print(f"模型檔案 {model_file_path} 不存在，無法進行預測。")
