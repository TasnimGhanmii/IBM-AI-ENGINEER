# ------------------------------------------------------------
# Advanced Data Augmentation with Keras – full standalone code
# ------------------------------------------------------------

# 1. Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array, array_to_img   # <= added array_to_img
)
from PIL import Image, ImageDraw
import os, glob, zipfile, urllib.request

# 2. Load CIFAR-10
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# 3. Quick peek
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1); plt.imshow(x_train[i]); plt.axis("off")
plt.suptitle("CIFAR-10 samples"); plt.show()

# 4. Create local sample.jpg
os.makedirs("assets", exist_ok=True)
img = Image.new("RGB", (224,224), (255,255,255))
draw = ImageDraw.Draw(img)
draw.rectangle([(50,50),(174,174)], fill=(255,0,0))
img.save("assets/sample.jpg")

# 5. Load sample into 4-D tensor
x = np.expand_dims(img_to_array(load_img("assets/sample.jpg")), axis=0)

# 6. Basic augmentations
basic = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
plt.figure(figsize=(8,8))
for i,batch in enumerate(basic.flow(x, batch_size=1)):
    plt.subplot(2,2,i+1)
    plt.imshow(array_to_img(batch[0])); plt.title(f"aug {i+1}")
    if i==3: break
plt.tight_layout(); plt.show()

# 7. Feature-wise / sample-wise norm
norm = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    samplewise_center=True,
    samplewise_std_normalization=True
)
norm.fit(x)  # (on real data fit on full train set)
plt.figure(figsize=(8,8))
for i,batch in enumerate(norm.flow(x, batch_size=1)):
    plt.subplot(2,2,i+1)
    plt.imshow(array_to_img(batch[0])); plt.title(f"norm {i+1}")
    if i==3: break
plt.tight_layout(); plt.show()

# 8. Custom noise injection
def add_noise(img):
    return img + np.random.normal(0, 0.1, img.shape)

noise_gen = ImageDataGenerator(preprocessing_function=add_noise)
plt.figure(figsize=(8,8))
for i,batch in enumerate(noise_gen.flow(x, batch_size=1)):
    plt.subplot(2,2,i+1)
    plt.imshow(array_to_img(batch[0])); plt.title(f"noise {i+1}")
    if i==3: break
plt.tight_layout(); plt.show()

# 9. Download practice images (works in plain Python too)
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/RgP3JFNtPTZA34UmG3KZaA/sample-images.zip"
zip_path = "sample-images.zip"
if not os.path.exists("sample_images"):
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall()
    os.remove(zip_path)

# 10. Build mini dataset
paths = glob.glob("sample_images/*.jpg")
train_imgs = np.array([img_to_array(load_img(p, target_size=(224,224))) for p in paths])

# 11. Exercise 1 – basic aug on new images
plt.figure(figsize=(10,10))
for i,batch in enumerate(basic.flow(train_imgs, batch_size=1)):
    plt.subplot(3,3,i+1)
    plt.imshow(array_to_img(batch[0])); plt.title(f"ex1-{i+1}")
    if i==8: break
plt.suptitle("Exercise 1 – basic augmentations"); plt.tight_layout(); plt.show()

# 12. Exercise 2 – normalization
norm.fit(train_imgs)
plt.figure(figsize=(10,10))
for i,batch in enumerate(norm.flow(train_imgs, batch_size=1)):
    plt.subplot(3,3,i+1)
    plt.imshow(array_to_img(batch[0])); plt.title(f"ex2-{i+1}")
    if i==8: break
plt.suptitle("Exercise 2 – normalization"); plt.tight_layout(); plt.show()

# 13. Exercise 3 – custom noise
noise_gen2 = ImageDataGenerator(preprocessing_function=add_noise)
plt.figure(figsize=(10,10))
for i,batch in enumerate(noise_gen2.flow(train_imgs, batch_size=1)):
    plt.subplot(3,3,i+1)
    plt.imshow(array_to_img(batch[0])); plt.title(f"ex3-{i+1}")
    if i==8: break
plt.suptitle("Exercise 3 – custom noise"); plt.tight_layout(); plt.show()