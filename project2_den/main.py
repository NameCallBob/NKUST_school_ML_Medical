if __name__ == "__main__":
    from models import Models
    from prepare import Prepare
    from evaluate import Evaluate
    import pandas as pd

    SAVE_MODEL = False
    """
    target class 有關於要判別的類別
    0 : den
    1 : flu
    2 : sep
    3 : gen
    """
    # 準備數據
    X_train, X_test, y_train, y_test = Prepare().getTrainingData(
        binary_classification=True, target_class=0, test_size=0.2
    )

    # 初始化模型和評估器
    model_handler = Models()
    evaluator = Evaluate()

    # 模型名稱和訓練方法
    models_to_train = {
        "RandomForest": model_handler.train_rf,
        "XGBoost": model_handler.train_xgboost,
        "AdaBoost": model_handler.train_adaboost,
    }

    # 清單確診和交叉驗證結果
    combined_results = []

    for model_name, train_func in models_to_train.items():
        print(f"\n--- 正在訓練 {model_name} 模型 ---")
        
        # 訓練模型
        train_func(X_train, y_train, SAVE_MODEL)
        model = model_handler.get_models()[model_name]

        # 評估模型
        print(f"\n正在評估 {model_name} 模型...")
        model_metrics = evaluator.model(model_name, model, X_train, X_test, y_train, y_test)

        # 評估交叉驗證
        cross_val_metrics = evaluator.cross_val(model, X_train, y_train, n_split=10)
        print(f"{model_name} 交叉驗證結果: {cross_val_metrics}\n")

        # 整合結果
        result_data = {
            "Model": model_name,
            "Accuracy": model_metrics["Accuracy"],
            "Precision": model_metrics["Precision"],
            "Recall": model_metrics["Recall"],
            "F1 Score": model_metrics["F1 Score"],
            "AUC": model_metrics["AUC"],
        }
        result_data.update(
            {
                f"{metric} (CV)": f"{mean:.4f} ± {std:.4f}"
                for metric, (mean, std) in cross_val_metrics.items()
            }
        )

        combined_results.append(result_data)
 
        # 自動導出結果
        output_file = f"./result/{model_name.lower()}_evaluation.xlsx"
        pd.DataFrame([result_data]).to_excel(output_file, index=False)
        print(f"{model_name} 模型結果已導出至 {output_file}")

    print("\n--- 所有模型訓練和評估結束 ---")
