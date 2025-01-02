if __name__ == "__main__":
    from models import Models
    from prepare import prepare
    from evaluate import Evaluate
    import pandas as pd
    for i in range(1,3):
        for j in ['tor','ele']:
            # 準備數據
            X_train, X_test, y_train, y_test = prepare().getTrainingData(
                year=i,
                test_size=0.3,
                data_type=j
                )

            # 初始化模型和評估器
            model_handler = Models()
            evaluator = Evaluate()

            # 訓練模型
            model_handler.train_rf(X_train, y_train,True,j,f"{j}_{i}")
            model_handler.train_xgboost(X_train, y_train,True,j,f"{j}_{i}")
            model_handler.train_adaboost(X_train, y_train,True,j,f"{j}_{i}")

            # 評估模型和交叉驗證
            models_results = {}
            cross_val_results = {}
            for model_name, model in model_handler.get_models().items():
                print(f"評估 {model_name} 模型...")
                models_results[model_name] = evaluator.model(model_name, model, X_train, y_train, X_test, y_test, i,j)
                cross_val_results[model_name] = evaluator.cross_val(model, X_train, y_train, n_split=10)
            # 繪製合併 ROC 曲線
            evaluator.plot_combined_roc(models_results,i,j)

            # 匯總結果
            test_data = {
                model_name: {
                    "Test Accuracy": metrics["Accuracy"],
                    "Test Precision": metrics["Precision"],
                    "Test Recall": metrics["Recall"],
                    "Test F1 Score": metrics["F1 Score"],
                    "Test AUC": metrics["AUC"],
                    "Train Accuracy": metrics_train["Accuracy"],
                    "Train Precision": metrics_train["Precision"],
                    "Train Recall": metrics_train["Recall"],
                    "Train F1 Score": metrics_train["F1 Score"],
                    "Train AUC": metrics_train["AUC"],
                }
                for model_name, result in models_results.items()
                for metrics in [result["Test Metrics"]]
                for metrics_train in [result["Train Metrics"]]
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
            output_file = f"./project1_Neuropathy/result/model_evaluation_{j}_summary_{i}.xlsx"
            combined_df.to_excel(output_file)
            print(f"評估結果已導出至 {output_file}")
