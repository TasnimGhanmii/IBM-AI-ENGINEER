import tensorflow as tf 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D 
import numpy as np 
import matplotlib.pyplot as plt 

#greyscale 28*28 img
input_layer=Input(28,28,1)

#adding convo layer with 32 filters(kernels) and a 3*3 matrix kernel (filter)
#padding='same' so the output has the same dims as the input
#applies convo ops (encoder)
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer) 
#adding transpose layer for reconstructing the image (decoder)
transpose_convo_layer=Conv2DTranspose(filters=1,kernel_size=(3,3),activation='sigmoid',padding='same')(conv_layer)

#model creation
model=Model(inputs=input_layer,outputs=transpose_convo_layer)
#compile
model.compile(optimizer='adams',loss='mean_squared_error')

#train
#gen training data
X_train = np.random.rand(1000, 28, 28, 1) 
y_train = X_train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2) 

#test
X_test = np.random.rand(200, 28, 28, 1) 
y_test = X_test 
loss = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}')

# Predict on test data 
y_pred = model.predict(X_test) 

# Plot some sample images 

n = 10 # Number of samples to display 

plt.figure(figsize=(20, 4))

for i in range(n): 

    # Display original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original") 
    plt.axis('off') 
    # Display reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show() 