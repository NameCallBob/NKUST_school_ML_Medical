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
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer

class Evaluate:
    @staticmethod
    def model(model_name, model, X_test, y_test,year):
        """
        評估模型
        """
        y_pred_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # 計算評估指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        auc = roc_auc_score(y_test, y_pred_probs)
        cm = confusion_matrix(y_test, y_pred)

        # 個別 ROC 曲線繪製
        Evaluate._plot_individual_roc(y_test, y_pred_probs, model_name,year)

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc,
            "Confusion Matrix": cm,
            "y_test": y_test,
            "y_pred_probs": y_pred_probs,
        }

    @staticmethod
    def cross_val(model, X, y, n_split):
        """
        執行交叉驗證
        """
        scoring = {
            "Accuracy": make_scorer(accuracy_score),
            "Precision": make_scorer(precision_score, average="binary"),
            "Recall": make_scorer(recall_score, average="binary"),
            "F1": make_scorer(f1_score, average="binary"),
            "AUC": make_scorer(roc_auc_score),
        }
        kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)

        # 整理結果
        results = {
            metric: (cv_results[f'test_{metric}'].mean(), cv_results[f'test_{metric}'].std())
            for metric in scoring.keys()
        }

        return results

    @staticmethod
    def _plot_individual_roc(y_test, y_pred_probs, model_name,year):
        """
        繪製單一模型的 ROC 曲線
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_pred_probs):.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {model_name}")
        plt.legend(loc="best")
        plt.savefig(f"./result/{model_name}_roc_curve_{year}.png")
        plt.close()

    @staticmethod
    def plot_combined_roc(models_results,year):
        """
        繪製所有模型的合併 ROC 曲線
        """
        plt.figure()
        for model_name, result in models_results.items():
            fpr, tpr, _ = roc_curve(result["y_test"], result["y_pred_probs"])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {result['AUC']:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Combined ROC Curve")
        plt.legend(loc="best")
        plt.savefig(f"./result/combined_roc_curve_{year}.png")
        plt.close()
