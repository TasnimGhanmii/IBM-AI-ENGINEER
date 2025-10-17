import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. flatten & normalise
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype("float32") / 255
X_test  = X_test.reshape(X_test.shape[0], num_pixels).astype("float32") / 255

# 3. one-hot targets
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)
num_classes = y_test.shape[1]

# 4. build model
model = Sequential([
    Dense(num_pixels, activation="relu", input_shape=(num_pixels,)),
    Dense(100, activation="relu"),
    Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5. train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# 6. evaluate
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {scores[1]*100:.1f}%  Error: {1-scores[1]:.3f}")

# 7. save
model.save("classification_model.h5")