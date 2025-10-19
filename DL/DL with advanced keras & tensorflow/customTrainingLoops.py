
# 1. Imports & environment
import os, warnings, tensorflow as tf, numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import Callback

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 2. Data – MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 3. Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10)
])

# 4. Loss & optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# 5. Basic custom loop (2 epochs)
print("=== Basic custom loop ===")
for epoch in range(2):
    for step, (x, y) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if step % 200 == 0:
            print(f"Epoch {epoch+1}  Step {step}: loss={loss.numpy():.4f}")

# 6. Loop + accuracy metric
print("\n=== Loop with accuracy ===")
for epoch in range(2):
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        accuracy_metric.update_state(y, logits)
    print(f"Epoch {epoch+1}: loss={loss.numpy():.4f}  acc={accuracy_metric.result():.4f}")
    accuracy_metric.reset_state()

# 7. Custom callback
class CustomCB(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"[CB] End epoch {epoch+1} – loss={logs['loss']:.4f}  acc={logs['accuracy']:.4f}")

print("\n=== Loop with custom callback ===")
cb = CustomCB()
for epoch in range(2):
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        accuracy_metric.update_state(y, logits)
    cb.on_epoch_end(epoch, logs={"loss": loss.numpy(), "accuracy": accuracy_metric.result()})
    accuracy_metric.reset_state()

# 8. Functional API – binary classifier on synthetic 20-D data
print("\n=== Functional API – binary clf ===")
X = np.random.rand(1000, 20).astype("float32")
y = np.random.randint(0, 2, size=(1000, 1))

inp = Input(shape=(20,))
h1 = Dense(64, activation="relu")(inp)
h2 = Dense(64, activation="relu")(h1)
out = Dense(1, activation="sigmoid")(h2)
model2 = Model(inp, out)
model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model2.fit(X, y, epochs=10, batch_size=32, verbose=0)

# 9. Evaluate
X_test, y_test = np.random.rand(200, 20).astype("float32"), np.random.randint(0, 2, (200, 1))
loss, acc = model2.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}  accuracy: {acc:.4f}")