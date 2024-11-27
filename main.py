if __name__ == "__main__":
    from models import Models
    from prepare import prepare
    from evaluate import Evaluate
    import pandas as pd
    for i in range(1,3):
        # 準備數據
        X_train, X_test, y_train, y_test = prepare().getTrainingData(year=i,test_size=0.3)

        # 初始化模型和評估器
        model_handler = Models()
        evaluator = Evaluate()

        # 訓練模型
        model_handler.train_rf(X_train, y_train)
        model_handler.train_xgboost(X_train, y_train)
        model_handler.train_adaboost(X_train, y_train)

        # 評估模型和交叉驗證
        models_results = {}
        cross_val_results = {}
        for model_name, model in model_handler.get_models().items():
            print(f"評估 {model_name} 模型...")
            models_results[model_name] = evaluator.model(model_name, model, X_test, y_test,i)
            cross_val_results[model_name] = evaluator.cross_val(model, X_train, y_train, n_split=10)

        # 繪製合併 ROC 曲線
        evaluator.plot_combined_roc(models_results,i)

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
        output_file = f"./result/model_evaluation_summary_{i}.xlsx"
        combined_df.to_excel(output_file)
        print(f"評估結果已導出至 {output_file}")
