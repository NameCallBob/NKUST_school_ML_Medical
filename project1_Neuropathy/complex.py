from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import numpy as np 

class Complex_model:
    """
    主要訓練較複雜的模型
    """
    def __init__(self):
        self.models = {}

    def train_cnn(self,
        X_train,
        y_train,
        input_shape,
        num_classes,
        epochs=16,
        batch_size=8,
        learning_rate=0.001,
        dropout_rate=0.5,
        optimizer_type="adam",):

        
        # 確認輸入形狀
        height, width = input_shape[:2]

        # 自動調整卷積核大小
        kernel_size = (min(3, height), min(3, width))  # 確保卷積核不超過輸入尺寸
        pool_size = (min(2, height), min(2, width))  # 確保池化層不超過輸入尺寸

        # 將標籤轉為 one-hot 編碼
        y_train_categorical = to_categorical(y_train, num_classes=num_classes)

        # 設置優化器
        if optimizer_type.lower() == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"不支援的優化器類型：{optimizer_type}")

        # CNN 架構
        cnn = Sequential(
            [
                Conv2D(32, kernel_size, activation="relu", input_shape=input_shape),
                BatchNormalization(),  # 添加批次正規化
                Conv2D(64, kernel_size, activation="relu"),
                MaxPooling2D(pool_size=pool_size),
                Dropout(dropout_rate),  # 添加 Dropout
                Flatten(),
                Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(dropout_rate),  # 添加 Dropout
                Dense(num_classes, activation="softmax"),
            ]
        )

        # 編譯模型
        cnn.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # 設置 EarlyStopping 和 Learning Rate Scheduler
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

        # 訓練模型
        cnn.fit(
            X_train,
            y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.2,  # 設置驗證集
            callbacks=[early_stopping, lr_scheduler],
        )

        # 保存模型
        self.models["CNN"] = cnn


if __name__ == "__main__":
    from prepare import prepare
    from evaluate import Evaluate
    import pandas as pd
    import numpy as np

    for i in range(1,3):
        # 準備數據
        X_train, X_test, y_train, y_test = prepare().getTrainingData(year=i)

        # 初始化 Models 類別
        model_handler = Complex_model()
        
        # 確保 X_train 和 X_test 是 Numpy 陣列
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # 用來存放各模型的評估結果
        results = {}

        # CNN 模型訓練與評估
        try:
            print("開始訓練 CNN 模型...")

            # 判斷數據形狀
            if len(X_train.shape) == 2:  # 一維特徵數據需要 reshape
                feature_size = X_train.shape[1]
                height = int(np.sqrt(feature_size))  # 嘗試將特徵數轉換為矩陣形式
                width = feature_size // height
                X_train = X_train.reshape(-1, height, width, 1)  # (樣本數, 高度, 寬度, 通道數)
                X_test = X_test.reshape(-1, height, width, 1)
            elif len(X_train.shape) == 3:  # 如果已有 (樣本數, 高度, 寬度)
                X_train = np.expand_dims(X_train, axis=-1)  # 添加通道維度
                X_test = np.expand_dims(X_test, axis=-1)

            # 確定輸入形狀
            input_shape = X_train.shape[1:]  # (高度, 寬度, 通道)
            num_classes = len(np.unique(y_train))

            # 訓練 CNN
            model_handler.train_cnn(X_train, y_train, input_shape, num_classes)

            # 評估 CNN 模型
            results["CNN"] = Evaluate.model(
                "CNN", model_handler.models.get("CNN"), X_test, y_test,i
            )
            print("CNN 模型訓練完成並評估成功！")
        except Exception as e:
            print(f"CNN 模型訓練或評估時出現錯誤：{e}")

        # 結果輸出
        print("\n模型評估結果：")
        for model_name, metrics in results.items():
            print(f"\n{model_name} 模型的評估結果:")
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    print(f"{metric}:\n{value}")
                elif value is None:
                    print(f"{metric}: 無法計算")
                else:
                    print(f"{metric}: {value:.4f}")

        # 將結果轉為 DataFrame 並保存到 Excel
        output_file = f"./result/cnn_model_evaluation_{i}.xlsx"
        data = {
            metric: {model_name: metrics.get(metric, "無法計算") for model_name, metrics in results.items()}
            for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
        }
        df = pd.DataFrame(data).T
        df.to_excel(output_file)
        print(f"所有模型的評估結果已保存至 {output_file}")
