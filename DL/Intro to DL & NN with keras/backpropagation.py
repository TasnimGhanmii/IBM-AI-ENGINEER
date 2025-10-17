import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def sigmoid(x): return 1 / (1 + np.exp(-x))
def dsigmoid(x): return x * (1 - x)  # derivative w.r.t. activated output

# ---------- data ----------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2×4
d = np.array([0, 1, 1, 0])  # XOR truth table

# ---------- hyper-parameters ----------
inputSize, hiddenSize, outputSize = 2, 2, 1
lr, epochs = 0.1, 180_000

# ---------- weights & biases ----------
np.random.seed(42)
w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1  # 2×2
b1 = np.random.rand(hiddenSize, 1) * 2 - 1          # 2×1
w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1  # 1×2
b2 = np.random.rand(outputSize, 1) * 2 - 1           # 1×1

# ---------- training ----------
errors = []
for epoch in range(epochs):
    # forward
    z1 = w1 @ X + b1
    a1 = sigmoid(z1)
    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)

    # error
    error = d - a2
    if (epoch + 1) % 10_000 == 0:
        errors.append(np.mean(np.abs(error)))
        print(f"Epoch {epoch+1:>6},  avg error {errors[-1]:.5f}")

    # backward
    da2 = error * dsigmoid(a2)
    dz2 = da2
    da1 = w2.T @ dz2
    dz1 = da1 * dsigmoid(a1)

    # updates
    w2 += lr * (dz2 @ a1.T)
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)
    w1 += lr * (dz1 @ X.T)
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)

# ---------- testing ----------
z1 = w1 @ X + b1; a1 = sigmoid(z1)
z2 = w2 @ a1 + b2; a2 = sigmoid(z2)
print("\nFinal output:", a2.ravel())
print("Ground truth:", d)
print("Final avg error:", np.mean(np.abs(d - a2)))

# ---------- error curve ----------
plt.plot(range(0, len(errors) * 10_000, 10_000), errors, marker="o")
plt.title("XOR learning curve"); plt.xlabel("epochs"); plt.ylabel("mean abs error")
plt.savefig("xor_learning_curve.png"); plt.show()