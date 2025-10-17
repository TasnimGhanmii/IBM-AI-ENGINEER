import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# ---------- custom ReLU-dense layer ----------
class CustomDenseLayer(Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

# ---------- build & compile ----------
model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10),
    Softmax()
])
model.compile(optimizer='adam', loss='categorical_crossentropy')

# ---------- data ----------
x_train = np.random.random((1000, 20))
y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

x_test = np.random.random((200, 20))
y_test = tf.keras.utils.to_categorical(np.random.randint(10, size=(200, 1)), num_classes=10)

# ---------- train ----------
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)

# ---------- evaluate ----------
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {loss:.4f}")

# ---------- optional extras ----------
# 1. visualise
plot_model(model, to_file='custom_layer_model.png', show_shapes=True)

# 2. add Dropout
model_drop = Sequential([
    CustomDenseLayer(64),
    Dropout(0.5),
    CustomDenseLayer(10),
    Softmax()
])
model_drop.compile(optimizer='adam', loss='categorical_crossentropy')
model_drop.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
print("With Dropout:", model_drop.evaluate(x_test, y_test, verbose=0))

# 3. change units
model_wide = Sequential([
    CustomDenseLayer(256),
    CustomDenseLayer(10),
    Softmax()
])
model_wide.compile(optimizer='adam', loss='categorical_crossentropy')
model_wide.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
print("256-unit hidden:", model_wide.evaluate(x_test, y_test, verbose=0))