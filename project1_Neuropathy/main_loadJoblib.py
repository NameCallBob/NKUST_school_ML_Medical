if __name__ == "__main__":
    from prepare import prepare
    from evaluate import Evaluate
    import pandas as pd
    import joblib
    import os

    # 模型儲存路徑
    model_save_dir = "./result/model/"

    for i in range(1, 2):  # 循環年份
        for j in ['tor', 'ele']:  # 循環數據類型
            # 準備數據
            X_train, X_test, y_train, y_test = prepare().getTrainingData(
                year=2,
                test_size=0.3,
                data_type=j
            )

            # 將第二年的欄位名稱改為第一年利於之後預測
            # X_test.columns = [col.replace('_2', '_1') for col in X_test.columns]
            # print(X_test.columns)

            # 初始化評估器
            evaluator = Evaluate()

            # 模型名稱
            model_names = ["RF", "XGB", "ADA"]
            models_results = {}
            cross_val_results = {}

            for model_name in model_names:
                model_file_path = os.path.join(model_save_dir, f"{model_name}_{j}_{1}.joblib")

                if os.path.exists(model_file_path):
                    print(f"從 {model_file_path} 加載 {model_name} 模型...")
                    # 加載儲存的模型
                    model = joblib.load(model_file_path)

                    # 評估模型
                    print(f"評估 {model_name} 模型...")
                    models_results[model_name] = evaluator.model(model_name, model, X_test, y_test, i, j)
                    cross_val_results[model_name] = evaluator.cross_val(model, X_train, y_train, n_split=10)

                    # 預測示例
                    predictions = model.predict(X_test)
                    print(f"{model_name} 模型的部分預測結果: {predictions[:5]}")
                else:
                    print(f"模型檔案 {model_file_path} 不存在，跳過 {model_name} 評估。")

            # 繪製合併 ROC 曲線
            evaluator.plot_combined_roc(models_results, i, j)

            # 匯總結果
            test_data = {
                model_name: {
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1 Score"],
                    "AUC": metrics["AUC"],
                }
                for model_name, metrics in models_results.items()
            }

            cross_val_data = {
                model_name: {
                    f"{metric} (CV)": f"{mean:.4f} ± {std:.4f}"
                    for metric, (mean, std) in metrics.items()
                }
                for model_name, metrics in cross_val_results.items()
            }

            # 合併數據
            test_df = pd.DataFrame(test_data).T
            cross_val_df = pd.DataFrame(cross_val_data).T
            combined_df = pd.concat([test_df, cross_val_df], axis=1)

            # 導出結果到 Excel
            output_file = f"./result/model_evaluation_{j}_summary_{i}.xlsx"
            combined_df.to_excel(output_file)
            print(f"評估結果已導出至 {output_file}")
