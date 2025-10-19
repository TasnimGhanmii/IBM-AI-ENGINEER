
# 1. Imports
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.layers import TextVectorization, Embedding, MultiHeadAttention
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow", tf.__version__)

# 2. Load Shakespeare text
path = get_file("shakespeare.txt",
                "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path, "rb").read().decode("utf-8")
print("Corpus length:", len(text))
print(text[:300])

# 3. Text-vectorisation layer
vocab_size = 10_000
seq_length = 100
vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="int")
vectorizer.adapt(tf.data.Dataset.from_tensor_slices([text]).batch(1))
vectorized_text = vectorizer([text])[0]  # drop batch dim
print("Vectorised shape:", vectorized_text.shape)

# 4. Create input / target sequences
def create_sequences(vec, seq_len):
    inputs, targets = [], []
    for i in range(len(vec) - seq_len):
        inputs.append(vec[i : i + seq_len])
        targets.append(vec[i + 1 : i + seq_len + 1])
    return np.array(inputs), np.array(targets)

X, Y = create_sequences(vectorized_text.numpy(), seq_length)
X = tf.convert_to_tensor(X[:10_000])  # truncate for speed
Y = tf.convert_to_tensor(Y[:10_000])
print("X / Y shapes:", X.shape, Y.shape)

# 5. Transformer components ----------------------------------------------------
class TransformerBlock(Model):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_out = self.att(inputs, inputs)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(inputs + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(out1 + ffn_out)

class TransformerModel(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_len):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dense = Dense(vocab_size)

    def positional_encoding(self, seq_len, embed_dim):
        def get_angles(pos, i):
            return pos / np.power(10000, 2 * (i // 2) / embed_dim)
        pos = np.arange(seq_len)[:, np.newaxis]
        angle = get_angles(pos, np.arange(embed_dim)[np.newaxis, :])
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        return tf.cast(angle[np.newaxis, ...], tf.float32)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs) + self.pos_encoding[:, :seq_len, :]
        for blk in self.blocks:
            x = blk(x, training=training)
        return self.dense(x)

# 6. Build & compile -----------------------------------------------------------
embed_dim = 256
num_heads = 4
ff_dim    = 512
num_layers = 4

model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length)
_ = model(tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32))  # build
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

# 7. Train (2 epochs quick demo) ----------------------------------------------
early_stop = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
history = model.fit(X, Y, epochs=2, batch_size=32, callbacks=[early_stop], verbose=1)

plt.plot(history.history["loss"])
plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

# 8. Text-generation helper ----------------------------------------------------
def generate_text(model, start_str, num_gen=100, temperature=1.0):
    inp = vectorizer([start_str]).numpy()
    if inp.shape[1] < seq_length:
        inp = np.hstack([np.zeros((1, seq_length - inp.shape[1])), inp])
    elif inp.shape[1] > seq_length:
        inp = inp[:, -seq_length:]
    inp = tf.convert_to_tensor(inp, dtype=tf.int32)

    tokens = []
    for _ in range(num_gen):
        preds = model(inp)  # (1, seq_len, vocab)
        preds = preds[0, -1, :] / temperature  # last timestep logits
        pred_id = tf.random.categorical(preds[None, :], 1)[0, 0].numpy()
        tokens.append(vectorizer.get_vocabulary()[pred_id])
        # slide window
        inp = tf.concat([inp[:, 1:], [[pred_id]]], axis=1)
    return start_str + " " + " ".join(tokens)

# 9. Generate sample -----------------------------------------------------------
print("\nGenerated text:")
print(generate_text(model, "To be, or not to be", num_gen=120, temperature=0.8))

# ------------------------------------------------------------------
# PRACTICE EXERCISES (inline)
# ------------------------------------------------------------------

# EX-1: seq_length = 50
print("\nExercise 1 – seq_length=50")
seq_length = 50
X50, Y50 = create_sequences(vectorized_text.numpy(), seq_length)
X50 = tf.convert_to_tensor(X50[:10_000])
Y50 = tf.convert_to_tensor(Y50[:10_000])
model50 = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length)
_ = model50(tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32))
model50.compile("adam", "sparse_categorical_crossentropy")
hist50 = model50.fit(X50, Y50, epochs=2, batch_size=32, verbose=0)
print("Final loss (seq=50):", hist50.history["loss"][-1])

# EX-2: learning-rate scheduler
print("\nExercise 2 – LR scheduler")
def scheduler(epoch, lr):
    return lr * 0.5 if epoch and epoch % 10 == 0 else lr
lr_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
hist_lr = model.fit(X, Y, epochs=2, batch_size=32, callbacks=[lr_cb], verbose=0)
print("Final LR-scheduled loss:", hist_lr.history["loss"][-1])

# EX-3: generate longer text
print("\nExercise 3 – 200 tokens")
long_text = generate_text(model, "The king said", num_gen=200, temperature=0.8)
print(long_text[:500] + " ...")