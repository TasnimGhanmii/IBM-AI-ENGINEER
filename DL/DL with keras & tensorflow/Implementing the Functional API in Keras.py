
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

# -----------------------------------------------------------
# 1.  build a functional network (binary classification)
# -----------------------------------------------------------
input_layer = Input(shape=(20,))
x = Dense(64, activation='relu')(input_layer)
x = Dense(64, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# -----------------------------------------------------------
# 2.  dummy data & train / test
# -----------------------------------------------------------
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))
X_test = np.random.rand(200, 20)
y_test = np.random.randint(2, size=(200, 1))

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {loss:.4f}  |  Test accuracy: {acc:.4f}")

# -----------------------------------------------------------
# 3.  variants (dropout, tanh, batch-norm) – quick builders
# -----------------------------------------------------------
def with_dropout():
    inp = Input(shape=(20,))
    x = Dense(64, activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def with_tanh():
    inp = Input(shape=(20,))
    x = Dense(64, activation='tanh')(inp)
    x = Dense(64, activation='tanh')(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def with_batchnorm():
    inp = Input(shape=(20,))
    x = Dense(64, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ---- (un-comment to test any variant) -----------------------------------
# model = with_dropout()
# model = with_tanh()
# model = with_batchnorm()
# model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
# print("Variant evaluated:", model.evaluate(X_test, y_test, verbose=0))