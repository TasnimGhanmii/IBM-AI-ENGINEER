import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.utils import to_categorical

# 1. load & preprocess ---------------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to NHWC + normalize
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255
X_test  = X_test.reshape(-1,  28, 28, 1).astype("float32")  / 255

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)
num_classes = y_test.shape[1]

# 2. model: two Conv+Pool blocks -----------------------------------------------
def cnn_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (5, 5), activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(8, (2, 2), activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# 3. train & evaluate ----------------------------------------------------------
model = cnn_model()
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=200,
          verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {scores[1]:.4f}  Error: {100*(1-scores[1]):.2f}%")