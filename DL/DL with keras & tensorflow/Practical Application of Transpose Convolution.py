
# 1. Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model

print("TensorFlow", tf.__version__)

# 2. Dummy data (28×28 grayscale)
X_train = np.random.rand(1000, 28, 28, 1).astype("float32")
y_train = X_train.copy()  # auto-encoder target
X_test  = np.random.rand(200, 28, 28, 1).astype("float32")
y_test  = X_test.copy()

# 3. Build baseline model (Step 3 + 4)
inp = Input(shape=(28, 28, 1))
x   = Conv2D(32, 3, activation="relu", padding="same")(inp)
out = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
model = Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# 4. Train (Step 6)
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# 5. Evaluate (Step 7)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MSE: {test_loss:.4f}")

# 6. Visualise 10 reconstructions (Step 8)
y_pred = model.predict(X_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original"); plt.axis("off")
    # reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(y_pred[i].reshape(28, 28), cmap="gray")
    plt.title("Recon"); plt.axis("off")
plt.suptitle("Baseline – 3×3 kernels", fontsize=16)
plt.show()

# ==========================================================
# EXERCISE 1 – bigger kernels (5×5)
# ==========================================================
print("\n=== Exercise 1: 5×5 kernels ===")
ex1_inp = Input(shape=(28, 28, 1))
ex1_x   = Conv2D(32, (5,5), activation="relu", padding="same")(ex1_inp)
ex1_out = Conv2DTranspose(1, (5,5), activation="sigmoid", padding="same")(ex1_x)
ex1_model = Model(ex1_inp, ex1_out)
ex1_model.compile(optimizer="adam", loss="mse")
ex1_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
print("Test MSE:", ex1_model.evaluate(X_test, y_test, verbose=0))

# ==========================================================
# EXERCISE 2 – add Dropout(0.5)
# ==========================================================
print("\n=== Exercise 2: + Dropout(0.5) ===")
ex2_inp = Input(shape=(28, 28, 1))
ex2_x   = Conv2D(32, 3, activation="relu", padding="same")(ex2_inp)
ex2_x   = Dropout(0.5)(ex2_x)
ex2_out = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(ex2_x)
ex2_model = Model(ex2_inp, ex2_out)
ex2_model.compile(optimizer="adam", loss="mse")
ex2_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
print("Test MSE:", ex2_model.evaluate(X_test, y_test, verbose=0))

# ==========================================================
# EXERCISE 3 – tanh activation
# ==========================================================
print("\n=== Exercise 3: tanh activation ===")
ex3_inp = Input(shape=(28, 28, 1))
ex3_x   = Conv2D(32, 3, activation="tanh", padding="same")(ex3_inp)
ex3_out = Conv2DTranspose(1, 3, activation="tanh", padding="same")(ex3_x)
ex3_model = Model(ex3_inp, ex3_out)
ex3_model.compile(optimizer="adam", loss="mse")
ex3_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
print("Test MSE:", ex3_model.evaluate(X_test, y_test, verbose=0))