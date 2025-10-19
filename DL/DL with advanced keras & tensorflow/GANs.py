# 1.  Imports & environment
import os, warnings, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 2.  Load & preprocess MNIST
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 127.5 - 1.0  # scale to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)        # (60000, 28, 28, 1)

# 3.  Generator
def build_generator():
    model = Sequential([
        Dense(256, input_dim=100),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(28 * 28 * 1, activation="tanh"),
        Reshape((28, 28, 1))
    ])
    return model

# 4.  Discriminator
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-4), metrics=["accuracy"])
    return model

# 5.  GAN
generator = build_generator()
discriminator = build_discriminator()

discriminator.trainable = False
gan_input = Input(shape=(100,))
gan_out = discriminator(generator(gan_input))
gan = Model(gan_input, gan_out)
gan.compile(loss="binary_crossentropy", optimizer=Adam(1e-4))

# 6.  Training
BATCH = 64
EPOCHS = 50
real_lbl = np.ones((BATCH, 1))
fake_lbl = np.zeros((BATCH, 1))

d_losses, g_losses = [], []

for epoch in range(EPOCHS):
    # ---- train discriminator ----
    idx = np.random.randint(0, x_train.shape[0], BATCH)
    real_imgs = x_train[idx]
    noise = np.random.normal(0, 1, (BATCH, 100))
    fake_imgs = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_imgs, real_lbl)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_lbl)
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

    # ---- train generator ----
    noise = np.random.normal(0, 1, (BATCH, 100))
    g_loss = gan.train_on_batch(noise, real_lbl)

    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if epoch % 10 == 0:
        print(f"{epoch:03d}  D-loss: {d_loss:.4f}  G-loss: {g_loss:.4f}")

# 7.  Visualise generated images
def show_generated(generator, n=25):
    noise = np.random.normal(0, 1, (n, 100))
    imgs = generator.predict(noise, verbose=0)
    imgs = 0.5 * imgs + 0.5  # rescale [0, 1]

    fig, axs = plt.subplots(5, 5, figsize=(6, 6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(imgs[i, :, :, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("Generated MNIST digits", fontsize=16)
    plt.show()

show_generated(generator)

# 8.  Discriminator accuracy on fresh batch
noise = np.random.normal(0, 1, (BATCH, 100))
fake = generator.predict(noise, verbose=0)
real = x_train[np.random.randint(0, x_train.shape[0], BATCH)]

acc_real = discriminator.evaluate(real, real_lbl, verbose=0)[1]
acc_fake = discriminator.evaluate(fake, fake_lbl, verbose=0)[1]
print(f"\nDiscriminator accuracy – real: {acc_real*100:.2f}%  fake: {acc_fake*100:.2f}%")

# 9.  Plot training curves
plt.figure(figsize=(7, 4))
plt.plot(d_losses, label="Discriminator")
plt.plot(g_losses, label="Generator")
plt.title("GAN training loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()