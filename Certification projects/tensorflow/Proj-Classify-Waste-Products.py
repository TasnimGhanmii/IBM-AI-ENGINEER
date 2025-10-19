# =========================  ONE-CELL COMPLETE LAB  =========================
import numpy as np, os, glob, requests, zipfile, warnings, tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
from sklearn import metrics
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# download & extract dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
zip_path = "o-vs-r-split-reduced-1200.zip"
print("Downloading …")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
with zipfile.ZipFile(zip_path, 'r') as zf:
    for member in tqdm(zf.infolist(), unit='file'):
        zf.extract(member)
os.remove(zip_path)

# config
IMG_ROWS, IMG_COLS, BATCH_SIZE, N_EPOCHS, VAL_SPLIT, SEED = 150, 150, 32, 10, 0.2, 42
PATH_TRAIN, PATH_TEST = 'o-vs-r-split/train/', 'o-vs-r-split/test/'
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)

# generators
train_datagen = ImageDataGenerator(rescale=1./255., validation_split=VAL_SPLIT, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_datagen   = ImageDataGenerator(rescale=1./255., validation_split=VAL_SPLIT)
test_datagen  = ImageDataGenerator(rescale=1./255.)
train_gen = train_datagen.flow_from_directory(PATH_TRAIN, target_size=(IMG_ROWS,IMG_COLS), batch_size=BATCH_SIZE, class_mode='binary', shuffle=True, seed=SEED, subset='training')
val_gen   = val_datagen.flow_from_directory(PATH_TRAIN, target_size=(IMG_ROWS,IMG_COLS), batch_size=BATCH_SIZE, class_mode='binary', shuffle=True, seed=SEED, subset='validation')
test_gen  = test_datagen.flow_from_directory(PATH_TEST, target_size=(IMG_ROWS,IMG_COLS), batch_size=BATCH_SIZE, class_mode='binary', shuffle=False, seed=SEED)

# base model (VGG16 frozen)
base = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
x = Flatten()(base.output)
basemodel = Model(base.input, x)
for layer in basemodel.layers: layer.trainable = False

# head
model = Sequential([basemodel, Dense(512,activation='relu'), Dropout(0.3), Dense(512,activation='relu'), Dropout(0.3), Dense(1,activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(1e-4), metrics=['accuracy'])

# callbacks
def exp_decay(epoch): return 1e-4*np.exp(-0.1*epoch)
ckpt1, ckpt2 = 'O_R_tlearn_vgg16.keras', 'O_R_tlearn_fine_tune_vgg16.keras'
cbs = [LearningRateScheduler(exp_decay), EarlyStopping(monitor='val_loss',patience=4,mode='min',min_delta=0.01), ModelCheckpoint(ckpt1,monitor='val_loss',save_best_only=True,mode='min')]

# train feature-extraction
hist1 = model.fit(train_gen, steps_per_epoch=5, epochs=N_EPOCHS, validation_data=val_gen, validation_steps=val_gen.samples//BATCH_SIZE, callbacks=cbs, verbose=1)
plt.plot(hist1.history['accuracy'],label='Train Acc'); plt.plot(hist1.history['val_accuracy'],label='Val Acc'); plt.title('Acc Curve'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend(); plt.show()

# fine-tune: unfreeze block5_conv3 onwards
base = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
x = Flatten()(base.output)
basemodel = Model(base.input, x)
set_trainable = False
for layer in basemodel.layers:
    if layer.name=='block5_conv3': set_trainable=True
    layer.trainable = set_trainable
model = Sequential([basemodel, Dense(512,activation='relu'), Dropout(0.3), Dense(512,activation='relu'), Dropout(0.3), Dense(1,activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(1e-4), metrics=['accuracy'])
cbs = [LearningRateScheduler(exp_decay), EarlyStopping(monitor='val_loss',patience=4,mode='min',min_delta=0.01), ModelCheckpoint(ckpt2,monitor='val_loss',save_best_only=True,mode='min')]
hist2 = model.fit(train_gen, steps_per_epoch=5, epochs=N_EPOCHS, validation_data=val_gen, validation_steps=val_gen.samples//BATCH_SIZE, callbacks=cbs, verbose=1)
plt.plot(hist2.history['loss'],label='Train Loss'); plt.plot(hist2.history['val_loss'],label='Val Loss'); plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()
plt.plot(hist2.history['accuracy'],label='Train Acc'); plt.plot(hist2.history['val_accuracy'],label='Val Acc'); plt.title('Acc Curve'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend(); plt.show()

# evaluate on test
m1 = tf.keras.models.load_model(ckpt1)
m2 = tf.keras.models.load_model(ckpt2)
test_O = glob.glob('./o-vs-r-split/test/O/*')[:50]
test_R = glob.glob('./o-vs-r-split/test/R/*')[:50]
test_files = test_O + test_R
test_imgs = np.array([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(f,target_size=(150,150))) for f in test_files])/255.
test_lbl = [0 if Path(f).parent.name=='O' else 1 for f in test_files]
pred1 = ['O' if p<0.5 else 'R' for p in m1.predict(test_imgs,verbose=0)]
pred2 = ['O' if p<0.5 else 'R' for p in m2.predict(test_imgs,verbose=0)]
true_lbl = ['O' if l==0 else 'R' for l in test_lbl]
print('=== Extract Features ===\n', metrics.classification_report(true_lbl,pred1),'\n=== Fine-Tuned ===\n', metrics.classification_report(true_lbl,pred2))

# plot sample
def plot_img(image, model_name, actual, predicted):
    plt.imshow(image.astype('uint8')); plt.title(f'{model_name} — Actual: {actual}, Predicted: {predicted}'); plt.axis('off'); plt.show()
idx=1
plot_img(test_imgs[idx]*255, 'Extract Features', true_lbl[idx], pred1[idx])
plot_img(test_imgs[idx]*255, 'Fine-Tuned', true_lbl[idx], pred2[idx])
# ==========================================================================