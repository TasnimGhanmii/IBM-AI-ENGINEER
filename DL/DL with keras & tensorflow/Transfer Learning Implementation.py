
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt, os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

print("TensorFlow", tf.__version__)

# 2. Load ImageNet-pre-trained VGG16 (no top)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze all

# 3. Build transfer model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")  # binary classifier
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 4. Create dummy dataset (black vs white images)
os.makedirs("sample_data/class_a", exist_ok=True)
os.makedirs("sample_data/class_b", exist_ok=True)
for i in range(10):
    Image.fromarray(np.ones((224,224,3), dtype=np.uint8)*255).save(f"sample_data/class_a/img_{i}.jpg")
    Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)).save(f"sample_data/class_b/img_{i}.jpg")

# 5. Train from frozen features
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
    "sample_data", target_size=(224,224), batch_size=8,
    class_mode="binary", shuffle=True
)
print("Found", train_gen.samples, "images")
hist1 = model.fit(train_gen, epochs=5, verbose=2)

# 6. Fine-tune last 4 conv blocks
for layer in base_model.layers[-4:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR
              loss="binary_crossentropy", metrics=["accuracy"])
hist2 = model.fit(train_gen, epochs=5, verbose=2)

# 7. EXERCISE 1 – plot train/val loss with validation split
train_datagen_val = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen_val = train_datagen_val.flow_from_directory(
    "sample_data", target_size=(224,224), batch_size=8,
    class_mode="binary", subset="training"
)
val_gen_val = train_datagen_val.flow_from_directory(
    "sample_data", target_size=(224,224), batch_size=8,
    class_mode="binary", subset="validation"
)
hist_val = model.fit(train_gen_val, validation_data=val_gen_val, epochs=8, verbose=0)

plt.plot(hist_val.history["loss"], label="train")
plt.plot(hist_val.history["val_loss"], label="val")
plt.title("Exercise 1 – Loss curves"); plt.legend(); plt.show()

# 8. EXERCISE 2 – compare optimisers (SGD vs RMSprop)
def reset_weights(m):
    for l in m.layers:
        if hasattr(l, "kernel_initializer"):
            l.set_weights([l.kernel_initializer(tf.shape(l.kernel).numpy()),
                           l.bias_initializer(tf.shape(l.bias).numpy())])
    return m

for name, opt in [("SGD", tf.keras.optimizers.SGD(1e-4)),
                  ("RMSprop", tf.keras.optimizers.RMSprop(1e-4))]:
    m = tf.keras.models.clone_model(model)
    m.set_weights(model.get_weights())
    m.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    h = m.fit(train_gen_val, validation_data=val_gen_val, epochs=5, verbose=0)
    print(f"{name:8s} – final val-acc: {max(h.history['val_accuracy']):.3f}")

# 9. EXERCISE 3 – evaluate on “unseen” test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    "sample_data", target_size=(224,224), batch_size=8,
    class_mode="binary", shuffle=False
)
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"\nExercise 3 – Test loss: {loss:.4f}  Test accuracy: {acc:.3f}")