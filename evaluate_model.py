import numpy as np 

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import matplotlib.pyplot as plt
import numpy as np 

class evaluate:
    """
    模型評測
    """
    @staticmethod
    def model(model_name,model, X_test, y_test, num_classes=None):
        """
        對模型進行評測
        """
        if model_name == "CNN":
            # CNN 模型預測概率
            y_pred_probs = model.predict(X_test)
            # 預測類別
            y_pred = np.argmax(y_pred_probs, axis=1)

            # 確保 y_test 是 1D 數據
            if len(y_test.shape) > 1:
                y_test = np.argmax(y_test, axis=1)

        else:
            # 非 CNN 模型處理
            y_pred_probs = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

        # 計算評估指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # 處理 AUC
        auc = None
        try:
            if num_classes == 2 or len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_probs)
                evaluate._plot_roc_curve(y_test, y_pred_probs, model_name)
            else:
                auc = roc_auc_score(
                    y_test,
                    y_pred_probs,
                    multi_class="ovr",  # One-vs-Rest
                    average="weighted",
                )
        except ValueError as e:
            print(f"AUC 計算失敗：{e}")

        cm = confusion_matrix(y_test, y_pred)

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc,
            "Confusion Matrix": cm,
        }
    
    @staticmethod
    def _plot_roc_curve(y_test, y_pred_probs, model_name):
        """
        繪畫個別ROC圖
        """
        if len(np.unique(y_test)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_pred_probs):.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve for {model_name}")
            plt.legend(loc="best")
            plt.show()