import matplotlib.pyplot as plt
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

    @staticmethod
    def model(model_name, model, X_train, y_train, X_test, y_test):
        """
        評估模型，支援二分類與多類別分類
        """

        y_train = Evaluate._ensure_1d(y_train)
        y_test = Evaluate._ensure_1d(y_test)

        # 預測機率與類別
        y_pred_probs_test = model.predict_proba(X_test)
        y_pred_test = model.predict(X_test)
        y_pred_probs_train = model.predict_proba(X_train)
        y_pred_train = model.predict(X_train)

        # 多類別檢測
        classes = np.unique(y_train)
        n_classes = len(classes)

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
        Evaluate._plot_roc(y_test, y_pred_probs_test, model_name, classes)

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
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

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
    def _plot_roc(y_true, y_probs, model_name, classes):
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
            for i in range(n_classes):
                if y_probs.shape[1] <= i:
                    raise ValueError("y_probs 的維度與指定的 classes 不匹配")

                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                auc_score = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {model_name}")
        plt.legend(loc="best")
        plt.savefig(f"./result/pic/{model_name}_roc_curve.png")
        plt.close()

    @staticmethod
    def cross_val(model, X, y, n_split):
        """
        執行交叉驗證 (支援多類別)
        """
        scoring = {
            "Accuracy": make_scorer(accuracy_score),
            "Precision": make_scorer(precision_score, average="macro"),
            "Recall": make_scorer(recall_score, average="macro"),
            "F1": make_scorer(f1_score, average="macro"),
        }
        kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)

        results = {
            metric: (cv_results[f'test_{metric}'].mean(), cv_results[f'test_{metric}'].std())
            for metric in scoring.keys()
        }

        return results
