import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1.  activation & derivatives
# -----------------------------
sigmoid   = lambda z: 1 / (1 + np.exp(-z))
sig_grad  = lambda z: sigmoid(z) * (1 - sigmoid(z))

relu      = lambda z: np.maximum(0, z)
relu_grad = lambda z: (z > 0).astype(float)

# -----------------------------
# 2.  synthetic input range
# -----------------------------
z = np.linspace(-10, 10, 400)
sig_grad_vals = sig_grad(z)
relu_grad_vals = relu_grad(z)

# -----------------------------
# 3.  plot sigmoid & ReLU
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid', color='b')
plt.plot(z, sig_grad_vals, label='Sigmoid derivative', color='r', linestyle='--')
plt.title('Sigmoid Activation & Gradient')
plt.xlabel('Input (z)'); plt.ylabel('Output'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU', color='g')
plt.plot(z, relu_grad_vals, label='ReLU derivative', color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input (z)'); plt.ylabel('Output'); plt.legend()

plt.tight_layout()
plt.savefig('activation_gradients.png')
plt.show()

# -----------------------------
# 4.  tanh exercise (practice)
# -----------------------------
tanh   = lambda z: np.tanh(z)
tanh_grad = lambda z: 1 - tanh(z)**2

# -----------------------------
# 5.  compare tanh vs ReLU
# -----------------------------
z_small = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(z_small, relu(z_small), label='ReLU', color='g')
plt.plot(z_small, relu_grad(z_small), label='ReLU grad', color='r', linestyle='--')
plt.title('ReLU'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(z_small, tanh(z_small), label='tanh', color='b')
plt.plot(z_small, tanh_grad(z_small), label='tanh grad', color='r', linestyle='--')
plt.title('tanh'); plt.legend()

plt.tight_layout()
plt.savefig('relu_vs_tanh.png')
plt.show()