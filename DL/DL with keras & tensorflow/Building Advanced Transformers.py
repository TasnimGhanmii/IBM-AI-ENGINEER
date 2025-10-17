
# 1. Imports
import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout

print("TensorFlow", tf.__version__)

# 2. Synthetic stock-price data ------------------------------------------------
np.random.seed(42)
data_len = 2_000
trend = np.linspace(100, 200, data_len)
noise = np.random.normal(0, 2, data_len)
synthetic = trend + noise
df = pd.DataFrame(synthetic, columns=["Close"])
df.to_csv("stock_prices.csv", index=False)  # save once
data = df[["Close"]].values.astype("float32")

# 3. Min-max scaling -----------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# 4. Windowing utility ---------------------------------------------------------
def create_dataset(series, time_step=100):
    X, Y = [], []
    for i in range(len(series) - time_step - 1):
        X.append(series[i : i + time_step, 0])
        Y.append(series[i + time_step, 0])
    return np.array(X), np.array(Y)

TIME_STEP = 100
X, Y = create_dataset(data, TIME_STEP)
X = X.reshape(X.shape[0], TIME_STEP, 1)  # (samples, timesteps, features)
print("X shape:", X.shape, "Y shape:", Y.shape)

# 5. Multi-Head Self-Attention -------------------------------------------------
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense   = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, q, k, v):
        score = tf.matmul(q, k, transpose_b=True)
        dim_key = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled, axis=-1)
        return tf.matmul(weights, v), weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        q = self.split_heads(q, B)
        k = self.split_heads(k, B)
        v = self.split_heads(v, B)
        attn, _ = self.attention(q, k, v)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        concat = tf.reshape(attn, (B, -1, self.embed_dim))
        return self.combine_heads(concat)

# 6. Transformer Block ---------------------------------------------------------
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_out = self.att(inputs)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(inputs + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(out1 + ffn_out)

# 7. Transformer Encoder -------------------------------------------------------
class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, rate)
                       for _ in range(num_layers)]

    def call(self, inputs, training=False):
        x = inputs
        for blk in self.blocks:
            x = blk(x, training=training)
        return x

# 8. Build full model ----------------------------------------------------------
EMBED_DIM  = 128
NUM_HEADS  = 8
FF_DIM     = 512
NUM_LAYERS = 4

inputs = tf.keras.Input(shape=(TIME_STEP, 1))
x = Dense(EMBED_DIM)(inputs)  # project to embed_dim
encoded = TransformerEncoder(NUM_LAYERS, EMBED_DIM, NUM_HEADS, FF_DIM)(encoded)
encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)  # simpler than Flatten
outputs = Dense(1)(encoded)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")
model.summary()

# 9. Train ----------------------------------------------------------------------
model.fit(X, Y, epochs=20, batch_size=32, verbose=2)

# 10. Evaluate & visualise -----------------------------------------------------
pred = model.predict(X).squeeze()
pred = scaler.inverse_transform(pred.reshape(-1, 1))
true = scaler.inverse_transform(Y.reshape(-1, 1))

plt.figure(figsize=(10, 4))
plt.plot(true, label="True")
plt.plot(np.arange(TIME_STEP, TIME_STEP + len(pred)), pred, label="Pred")
plt.title("Transformer – stock price forecast")
plt.legend(); plt.show()

# ------------------------------------------------------------------
# PRACTICE EXERCISES (quick inline versions)
# ------------------------------------------------------------------

# EX-1: Add Dropout before final Dense
print("\nExercise 1 – Dropout(0.5) before output")
x = Dropout(0.5)(encoded)
outputs = Dense(1)(x)
model1 = tf.keras.Model(inputs, outputs)
model1.compile("adam", "mse")
model1.fit(X, Y, epochs=10, batch_size=32, verbose=0)
print("MSE with dropout:", model1.evaluate(X, Y, verbose=0))

# EX-2: Different batch sizes
print("\nExercise 2 – batch size 16 vs 64")
for bs in (16, 64):
    tf.keras.backend.clear_session()
    m = tf.keras.models.clone_model(model)
    m.compile("adam", "mse")
    m.fit(X, Y, epochs=5, batch_size=bs, verbose=0)
    print(f"Batch {bs:2d} – MSE:", m.evaluate(X, Y, verbose=0))

# EX-3: tanh activation on output layer
print("\nExercise 3 – tanh activation")
outputs = Dense(1, activation="tanh")(encoded)
model3 = tf.keras.Model(inputs, outputs)
model3.compile("adam", "mse")
model3.fit(X, Y, epochs=10, batch_size=32, verbose=0)
print("MSE with tanh:", model3.evaluate(X, Y, verbose=0))