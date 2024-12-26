**文件：`medical-ml/NKUST_school_ML_Medical/project1_Neuropathy/` 程式碼修改記錄**

**1. `eval2.py` 檔案修改**

   **1.1 `_ensure_2d(X)` 函數修改**

   *   **原始碼:**
        ```python
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
        ```
   *   **修改後程式碼:**
        ```python
        @staticmethod
        def _ensure_2d(X):
            """
            確保輸入 X 為 2D 結構
            """
            if isinstance(X, pd.Series):
                return X.to_frame()  # 將 Series 轉換成 DataFrame
            elif isinstance(X, np.ndarray) and X.ndim == 1:
                return X.reshape(-1, 1)
            # 此處新增一個判斷，如果 X 是純量，將其轉成 2D
            if isinstance(X, (int, float)):
                return np.array([[X]])
            return X
        ```
   *   **修改說明:**
        *   在原本的程式碼中，`_ensure_2d` 函數僅能處理 `pd.Series` 和一維 `np.ndarray` 數據，將它們轉換成二維結構。但當 `X` 是純量（如 `int` 或 `float`）時，原本程式碼會直接返回，而模型需要二維輸入，導致錯誤。
        *   新增的 `if isinstance(X, (int, float)): return np.array([[X]])` 判斷式，能將純量轉換為二維的 `np.ndarray`，例如原本輸入的是 `1.0`，輸出會是 `np.array([[1.0]])`。這使得 `_ensure_2d` 函數可以處理更廣泛的輸入數據類型，避免了因為純量輸入而導致的維度錯誤。

   **1.2 `model` 函數修改**

   *   **原始碼:** (僅顯示修改部分)
        ```python
         def model(model_name, model, X_train, y_train, X_test, y_test):
            # ...
            if not hasattr(model, "predict_proba"):
               # ...
        ```
   *   **修改後程式碼:**
        ```python
         def model(model_name, model, X_train, y_train, X_test, y_test, i, j):
             # ...
             # 在此處先確保 X_train 和 X_test 是 2D 結構
             X_train = Evaluate._ensure_2d(X_train)
             X_test = Evaluate._ensure_2d(X_test)
             if not hasattr(model, "predict_proba"):
                # ...
        ```
   *   **修改說明:**
      *   在 `model` 函數中，新增了在預測之前對 `X_train` 和 `X_test` 進行 `_ensure_2d` 處理的步驟。這確保了無論 `X_train` 和 `X_test` 的原始數據類型是什麼，它們在餵給模型時都是 2D 的。
      *   新增 `i` 和 `j` 參數，以便可以讀取 `main_saveJoblib.py` 的參數。

   **1.3 `_plot_roc` 函數修改**

    *   **原始碼:**
         ```python
         def _plot_roc(y_true, y_probs, model_name, classes):
            # ...
            plt.savefig(f"./project1_Neuropathy/result/pic/{model_name}_roc_curve.png")
         ```
    *   **修改後程式碼:**
         ```python
         def _plot_roc(y_true, y_probs, model_name, classes, i, j):
             # ...
             plt.savefig(f"./project1_Neuropathy/result/pic/{model_name}_roc_curve_{j}_{i}.png")
         ```
    *  **修改說明:**
        *  新增 `i` 和 `j` 參數，以便可以讀取 `main_saveJoblib.py` 的參數。
        *   修改 `savefig` 的路徑，讓產出的圖片可以正確儲存到路徑。

   **1.4 `plot_confusion_matrix` 函數修改**

    *   **原始碼:**
         ```python
         def plot_confusion_matrix(y_true, y_pred, classes, model_name):
            # ...
            plt.savefig(f"./project1_Neuropathy/result/pic/{model_name}_confusion_matrix.png")
         ```
    *   **修改後程式碼:**
         ```python
         def plot_confusion_matrix(y_true, y_pred, classes, model_name, i, j):
            # ...
            plt.savefig(f"./project1_Neuropathy/result/pic/{model_name}_confusion_matrix_{j}_{i}.png")
         ```
    *  **修改說明:**
        *  新增 `i` 和 `j` 參數，以便可以讀取 `main_saveJoblib.py` 的參數。
        *   修改 `savefig` 的路徑，讓產出的圖片可以正確儲存到路徑。

    **1.5 `cross_val` 函數修改**
    *  **原始碼:**
        ```python
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
             # ...
        ```
    *   **修改後程式碼:**
        ```python
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
            # ...
        ```
    *   **修改說明:**
        *   在 `cross_val` 函數中，我們修改了 `scoring` 字典，將 `zero_division=0` 參數傳遞給 `precision_score`、`recall_score` 和 `f1_score`。這確保了在交叉驗證過程中，即使某些類別沒有任何預測樣本，這些指標的計算也會被設置為 0，而不會產生警告。

    **1.6 `_calculate_metrics` 函數修改**

    *  **原始碼:**
        ```python
        def _calculate_metrics(y_true, y_pred, y_probs, classes):
            """
            計算多類別或二分類的指標
            """
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")
            # ...
        ```
    *   **修改後程式碼:**
         ```python
        def _calculate_metrics(y_true, y_pred, y_probs, classes):
            """
            計算多類別或二分類的指標
            """
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            # ...
         ```
    *   **修改說明:**
       *  在 `_calculate_metrics` 函數中，新增了 `zero_division=0` 參數傳遞給 `precision_score`、`recall_score` 和 `f1_score`。這確保了在計算測試指標時，即使某些類別沒有任何預測樣本，這些指標的計算也會被設置為 0，而不會產生警告。

    **1.7 `plot_combined_roc` 函數修改**

    *   **原始碼:**
         ```python
         def plot_combined_roc(models_results):
             # ...
             plt.savefig(f"./project1_Neuropathy/result/pic/Combined_ROC_Curve.png")
         ```
    *   **修改後程式碼:**
         ```python
         def plot_combined_roc(models_results, i, j):
             # ...
             plt.savefig(f"./project1_Neuropathy/result/pic/Combined_ROC_Curve_{j}_{i}.png")
         ```
    *  **修改說明:**
        *  新增 `i` 和 `j` 參數，以便可以讀取 `main_saveJoblib.py` 的參數。
        *   修改 `savefig` 的路徑，讓產出的圖片可以正確儲存到路徑。

**2. `main_saveJoblib.py` 檔案修改**

   *   **原始碼:**
        ```python
         for model_name, model in model_handler.get_models().items():
            # ...
            models_results[model_name] = evaluator.model(model_name, model, X_test, y_test)
            # ...
            test_data = {
                model_name: {
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1 Score"],
                    "AUC": metrics["AUC"],
                }
                for model_name, metrics in models_results.items()["Test Metrics"]
            }
        ```

   *   **修改後程式碼:**
        ```python
         for model_name, model in model_handler.get_models().items():
            # ...
            models_results[model_name] = evaluator.model(model_name, model, X_train, y_train, X_test, y_test, i,j)
            # ...
            test_data = {
                model_name: {
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1 Score"],
                    "AUC": metrics["AUC"],
                }
                for model_name, result in models_results.items()
                for metrics in [result["Test Metrics"]]
            }
        ```
    *   **修改說明:**
       *  修改 `model` 函數的調用，新增 `i` 和 `j` 參數。
       *  修正 `test_data` 資料的獲取方式，使用兩層迴圈來正確讀取模型指標。
       *  使用 `for model_name, result in models_results.items() for metrics in [result["Test Metrics"]]` 來正確地從每個模型的結果中提取 "Test Metrics" 字典，並獲取對應的指標數據。

**3. 修改效果總結**

*   **`eval2.py` 修改：**
    *   `_ensure_2d` 函數變得更通用，可以處理純量輸入。
    *   `model` 函數確保輸入是 2D 結構
    *   `_plot_roc` 和 `plot_confusion_matrix` 函數產出的圖片