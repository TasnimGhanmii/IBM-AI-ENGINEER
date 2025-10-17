import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, AdditiveAttention, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------
# 1.  toy parallel corpus
# -----------------------------------------------------------
input_texts = [
    "Hello.", "How are you?", "I am learning machine translation.",
    "What is your name?", "I love programming."
]
target_texts = [
    "Hola.", "¿Cómo estás?", "Estoy aprendiendo traducción automática.",
    "¿Cuál es tu nombre?", "Me encanta programar."
]
target_texts = ["startseq " + t + " endseq" for t in target_texts]

# -----------------------------------------------------------
# 2.  tokenise & pad
# -----------------------------------------------------------
inp_tok = Tokenizer(); inp_tok.fit_on_texts(input_texts)
tar_tok = Tokenizer(); tar_tok.fit_on_texts(target_texts)

inp_seq = inp_tok.texts_to_sequences(input_texts)
tar_seq = tar_tok.texts_to_sequences(target_texts)

inp_vocab = len(inp_tok.word_index) + 1
tar_vocab = len(tar_tok.word_index) + 1

max_inp = max(len(s) for s in inp_seq)
max_tar = max(len(s) for s in tar_seq)

inp_seq = pad_sequences(inp_seq, maxlen=max_inp, padding='post')
tar_seq = pad_sequences(tar_seq, maxlen=max_tar, padding='post')

# decoder inputs / outputs (teacher forcing)
decoder_in  = tar_seq[:, :-1]
decoder_out = tar_seq[:, 1:]
decoder_out = tf.one_hot(decoder_out, tar_vocab)

# -----------------------------------------------------------
# 3.  encoder-decoder with additive attention
# -----------------------------------------------------------
# encoder
enc_in = Input(shape=(max_inp,))
enc_emb = Embedding(inp_vocab, 256, mask_zero=True)(enc_in)
enc_lstm = LSTM(256, return_sequences=True, return_state=True)
enc_out, state_h, state_c = enc_lstm(enc_emb)
enc_states = [state_h, state_c]

# decoder
dec_in = Input(shape=(max_tar - 1,))
dec_emb = Embedding(tar_vocab, 256, mask_zero=True)(dec_in)
dec_lstm = LSTM(256, return_sequences=True, return_state=True)
dec_out, _, _ = dec_lstm(dec_emb, initial_state=enc_states)

# attention
attn = AdditiveAttention()
attn_out = attn([dec_out, enc_out])  # query=dec_out, value=enc_out

# merge & predict
merge = Concatenate()([dec_out, attn_out])
dense = Dense(tar_vocab, activation='softmax')
output = dense(merge)

model = Model([enc_in, dec_in], output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------------------------------------
# 4.  train
# -----------------------------------------------------------
hist = model.fit([inp_seq, decoder_in], decoder_out,
                 epochs=100, batch_size=16, verbose=0)

# -----------------------------------------------------------
# 5.  plot loss curve
# -----------------------------------------------------------
plt.plot(hist.history['loss'])
plt.title('Transformer Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.savefig('transformer_loss.png'); plt.show()