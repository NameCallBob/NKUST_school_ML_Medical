import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.preprocessing import label_binarize

class Evaluate:
    @staticmethod
    def _ensure_1d(y):
        """
        確保目標變數 y 是 1D 陣列
        """
        if isinstance(y, pd.DataFrame) or isinstance(y, np.ndarray):
            if y.ndim == 2:
                return np.argmax(y, axis=1)  # 將 One-Hot 轉換回類別索引
        return y

    @staticmethod
    def _ensure_2d(X):
        """
        確保輸入 X 為 2D 結構
        """
        if isinstance(X, pd.Series):
            return X.to_frame()  # 將 Series 轉換成 DataFrame
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            return X.reshape(-1, 1)
        return X

        # 此處新增一個判斷，如果 X 是純量，將其轉成 2D
        if isinstance(X, (int, float)):
            return np.array([[X]])
        return X

    @staticmethod
    def model(model_name, model, X_train, y_train, X_test, y_test, i, j):
        """
        評估模型，支援二分類與多類別分類
        """

        y_train = Evaluate._ensure_1d(y_train)
        y_test = Evaluate._ensure_1d(y_test)

        # 在此處先確保 X_train 和 X_test 是 2D 結構
        X_train = Evaluate._ensure_2d(X_train)
        X_test = Evaluate._ensure_2d(X_test)

        if not hasattr(model, "predict_proba"):
            print(f"模型 {model_name} 不支援 predict_proba，無法計算 ROC 曲線或 AUC。")
            return {"Error": "Model does not support predict_proba"}

        # 預測機率與類別
        y_pred_probs_test = model.predict_proba(X_test)
        y_pred_test = model.predict(X_test)
        y_pred_probs_train = model.predict_proba(X_train)
        y_pred_train = model.predict(X_train)

        # 多類別檢測
        classes = np.unique(y_train)
        n_classes = len(classes)

        # 檢查 y_probs 維度是否與類別數量一致
        if y_pred_probs_test.shape[1] != n_classes:
            raise ValueError("y_probs 的列數與類別數量不匹配，請檢查輸出。")

        # 指標計算 (使用 macro average 適應多類別)
        metrics = {
            "Train": Evaluate._calculate_metrics(y_train, y_pred_train, y_pred_probs_train, classes),
            "Test": Evaluate._calculate_metrics(y_test, y_pred_test, y_pred_probs_test, classes),
        }

        # 過擬合檢測
        overfit_status = Evaluate._check_overfitting(
            metrics["Train"]["Accuracy"], metrics["Test"]["Accuracy"]
        )

        # ROC 曲線繪製
        Evaluate._plot_roc(y_test, y_pred_probs_test, model_name, classes, i, j)

        # 混淆矩陣繪製
        Evaluate.plot_confusion_matrix(y_test, y_pred_test, classes, model_name, i, j)

        results = {
            "Train Metrics": metrics["Train"],
            "Test Metrics": metrics["Test"],
            "Overfitting": overfit_status,
            "y_test": y_test,
            "y_pred_probs_test": y_pred_probs_test,
        }
        return results

    @staticmethod
    def _calculate_metrics(y_true, y_pred, y_probs, classes):
        """
        計算多類別或二分類的指標
        """

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # 假設為二元分類時的處理
        if len(classes) == 2:
            # y_probs為 (n_samples, 2), 選取對正類的機率
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            # 多類別情境
            y_true_bin = label_binarize(y_true, classes=classes)
            auc = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")

        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "AUC": auc}

    @staticmethod
    def _check_overfitting(train_score, test_score, threshold=0.05):
        """
        檢查是否過擬合
        """
        diff = train_score - test_score
        if diff > threshold:
            return f"可能存在過擬合 (差異: {diff:.2f})"
        return f"未檢測到過擬合 (差異: {diff:.2f})"

    @staticmethod
    def _plot_roc(y_true, y_probs, model_name, classes, i, j):
        """
        繪製 ROC 曲線 (支援多類別 One-vs-Rest)
        """
        # 檢查 y_probs 維度是否與類別數量一致
        if len(classes) == 2:  # 二分類特殊處理
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])  # 只取正類別機率
            auc_score = roc_auc_score(y_true, y_probs[:, 1])

            plt.figure()
            plt.plot(fpr, tpr, label=f"Class {classes[1]} (AUC = {auc_score:.2f})")
        else:
            # 多類別情況
            y_true_bin = label_binarize(y_true, classes=classes)
            n_classes = len(classes)

            plt.figure()
            for k in range(n_classes):
                if y_probs.shape[1] <= k:
                    raise ValueError("y_probs 的維度與指定的 classes 不匹配")

                fpr, tpr, _ = roc_curve(y_true_bin[:, k], y_probs[:, k])
                auc_score = roc_auc_score(y_true_bin[:, k], y_probs[:, k])
                plt.plot(fpr, tpr, label=f"Class {classes[k]} (AUC = {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {model_name}")
        plt.legend(loc="best")
        plt.savefig(f"./project1_Neuropathy/result/pic/{model_name}_roc_curve_{j}_{i}.png")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, model_name, i, j):
        """
        繪製混淆矩陣
        """
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.savefig(f"./project1_Neuropathy/result/pic/{model_name}_confusion_matrix_{j}_{i}.png")
        plt.close()

    @staticmethod
    def cross_val(model, X, y, n_split):
        """
        執行交叉驗證 (支援多類別)
        """
        scoring = {
            "Accuracy": make_scorer(accuracy_score),
            "Precision": make_scorer(precision_score, average="macro", zero_division=0),
            "Recall": make_scorer(recall_score, average="macro", zero_division=0),
            "F1": make_scorer(f1_score, average="macro", zero_division=0),
        }
        kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)

        results = {
            metric: (cv_results[f'test_{metric}'].mean(), cv_results[f'test_{metric}'].std())
            for metric in scoring.keys()
        }

        # 繪製交叉驗證結果
        Evaluate.plot_cross_val_results(cv_results, scoring.keys(), model.__class__.__name__)

        return results

    @staticmethod
    def plot_cross_val_results(cv_results, metrics, model_name):
        """
        繪製交叉驗證結果的箱形圖
        """
        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics, start=1):
            plt.subplot(1, len(metrics), i)
            plt.boxplot(cv_results[f'test_{metric}'])
            plt.title(metric)
        plt.savefig(f"./project1_Neuropathy/result/pic/{model_name}_cross_val.png")
        plt.close()
    @staticmethod
    def plot_combined_roc(models_results,i,j):
        plt.figure(figsize=(10, 6))
        for model_name, result in models_results.items():
            y_test = result["y_test"]
            y_pred_probs_test = result["y_pred_probs_test"]
            classes = np.unique(y_test)
            
            if len(classes) == 2:  # 二分類特殊處理
                fpr, tpr, _ = roc_curve(y_test, y_pred_probs_test[:, 1])  # 只取正類別機率
                auc_score = roc_auc_score(y_test, y_pred_probs_test[:, 1])
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
            else:
            # 多類別情況
                y_true_bin = label_binarize(y_test, classes=classes)
                n_classes = len(classes)

                for k in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, k], y_pred_probs_test[:, k])
                    auc_score = roc_auc_score(y_true_bin[:, k], y_pred_probs_test[:, k])
                    plt.plot(fpr, tpr, label=f"{model_name} Class {classes[k]} (AUC = {auc_score:.2f})")
        
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Combined ROC Curve")
        plt.legend(loc="best")
        plt.savefig(f"./project1_Neuropathy/result/pic/Combined_ROC_Curve_{j}_{i}.png")
        plt.close()