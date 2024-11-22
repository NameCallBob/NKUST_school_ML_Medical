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
    from train import prepare ; from evaluate_model import  evaluate
    X_train, X_test, y_train, y_test = prepare().getTrainingData()

    # 初始化 Models 類別
    model_handler = Complex_model()

    # 用來存放各模型的評估結果
    results = {}

    # 卷積神經網路 (假設數據需要 reshape)
    if len(X_train.shape) == 2:  # CNN 需要多維輸入
        num_classes = len(np.unique(y_train))
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)  # 調整為 (樣本數, 特徵數, 高度, 寬度)
        num_classes = len(np.unique(y_train))
        # 調整數據形狀以符合 CNN 要求
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

        # 將標籤轉換為 one-hot 編碼
        y_train_categorical = to_categorical(y_train, num_classes=num_classes)
        y_test_categorical = to_categorical(y_test, num_classes=num_classes)
        
        print("訓練 CNN 模型...")
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)  # 調整為 (樣本數, 特徵數, 高度, 寬度)
        input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3])        
        num_classes = len(np.unique(y_train))
        model_handler.train_cnn(X_train_cnn, y_train, input_shape, num_classes)
        results["CNN"] = evaluate.model("CNN",model_handler.models.get("CNN"),X_test_cnn, y_test)

    # 輸出結果
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