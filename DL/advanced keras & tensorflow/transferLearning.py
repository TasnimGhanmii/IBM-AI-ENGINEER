import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#for directory ops
import os
#for creating & saving imagess
from PIL import Image

# Load the VGG16 model pre-trained on ImageNet
#using the weightes trained on ImageNet (source of the weights), execluding the top classification layers 
# (aka fully connected layers at the end of the model responsible for classification) of the original model
#execluding the top layers allows me to add my own custom layers on top of the pre-trained model=>adapt to my need (#classification tasks, # nbs of classes)
#& setting the input shape to 224*224 piwels with 3 color channels (RGB)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
#by setting all layers non trainable=>preserving the weights
for layer in base_model.layers:
    layer.trainable = False


# Create a new model and add the base model and new layers
model = Sequential([
    base_model,
    #flattens the output of VGG16 to 1D array
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Change to the number of classes you have
])

# Compile the model
#to specify key components that define how the model gonna be trained and evaluated
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Create directories if they don't exist
os.makedirs('sample_data/class_a', exist_ok=True)
os.makedirs('sample_data/class_b', exist_ok=True)

# Create 10 sample images for each class
for i in range(10):
    # Create a blank white image for class_a
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
    img.save(f'sample_data/class_a/img_{i}.jpg')

    # Create a blank black image for class_b
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    img.save(f'sample_data/class_b/img_{i}.jpg')

print("Sample images created in 'sample_data/'")


# Load and preprocess the dataset
#rescaling pixels
train_datagen = ImageDataGenerator(rescale=1./255)
#load images from the directory
train_generator = train_datagen.flow_from_directory(
    'sample_data',
    #resiza img
    target_size=(224, 224),
    #batch of 32 imds
    batch_size=32,
    #set class mode for binary classification
    class_mode='binary'
)

# Verify if the generator has loaded images correctly
print(f"Found {train_generator.samples} images belonging to {train_generator.num_classes} classes.")

# Train the model
if train_generator.samples > 0:
    model.fit(train_generator, epochs=10)


#fine-tuning 
# Unfreeze the top layers of the base model 
for layer in base_model.layers[-4:]:
    layer.trainable = True 

# Compile the model again 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

# Train the model again 
model.fit(train_generator, epochs=10) 