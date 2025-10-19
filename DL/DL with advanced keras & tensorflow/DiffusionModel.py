import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


(x_train,_),(x_test,_)=mnist.load_data()
x_train=x_train.astype('float32')/255.
x_train=x_test.astype('float32')/255.

#expand dims to match input shape (28,28,1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Expand dimensions to match the input shape (28, 28, 1)  
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Add noise to the data
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip the values to be within the range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#define encoder
input=Input(shape=(28,28,1))
conv1=Conv2D(filters=32,kernel=(3,3),activation='relu',padding='same')(input)
conv2=Conv2D(filters=64,kernel=(3,3),activation='relu',padding='same')(conv1)
#flatten
encoded=Flatten()(conv2)

#bottelneck 
#256 closest power of 2 number to 28*28=784
bottleneck=Dense(256,activation='relu')(encoded)

#decoder
#to expand bottleneck representation
dense=Dense(256,activation='relu')(bottleneck)

# Reshape to (7, 7, 64) to begin upsampling back to 28x28
reshape = Reshape((7, 7, 64))(dense)

# First Conv2DTranspose layer: increase spatial dimensions and reduce filters
conv_transpose1 = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(reshape)

# Second Conv2DTranspose layer: go from (14,14,64) to (28,28,32)
conv_transpose2 = Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(conv_transpose1)

# Final layer to reconstruct the input image (1 channel, sigmoid for [0,1] range)
output = Conv2DTranspose(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid')(conv_transpose2)

# Build the model
autoencoder = Model(input, output)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Define the diffusion model architecture with reduced complexity
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)  # Reduced filters
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # Reduced filters
x = Flatten()(x)
x = Dense(64, activation='relu')(x)  # Reduced size
x = Dense(28*28*32, activation='relu')(x)  # Reduced size
x = Reshape((28, 28, 32))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)  # Reduced filters
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)  # Reduced filters
output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
diffusion_model = Model(input_layer, output_layer)

# Compile the model with mixed precision and a different loss function
diffusion_model.compile(optimizer='adam', loss='mean_squared_error')  # Using MSE for regression tasks







