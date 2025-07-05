import tensorflow as tf 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense,Dropout,BatchNormalization 
import numpy as np 

#creating layers
#input layer, vector of length 20
#it has 20 inputs/features/neurons
input_layer = Input(shape=(20,))

#hidden layers
#fully connected layer connected with input layer ,it has 64 units/neurons
hidden_layer1 = Dense(64, activation='relu')(input_layer) 
#adding drop layer to prevent overfitting
dropout_layer = Dropout(rate=0.5)(hidden_layer1)
#adding batch normalization to stabilize and speed up model
batch_norm_layer = BatchNormalization()(hidden_layer1)

hidden_layer2 = Dense(64, activation='relu')(hidden_layer2)
dropout_layer = Dropout(rate=0.5)(hidden_layer2)
batch_norm_layer = BatchNormalization()(hidden_layer2)

#output layer with 1 neuron
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)  

#creating the model
#connecting the output and input layers
model = Model(inputs=input_layer, outputs=output_layer)
#gives summary visualizations etc..
model.summary()

#compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training 
#data of 1000 samples (rows) & 20 features (cols)
X_train = np.random.rand(1000, 20) 
#data of 1 col(target) & 1000 samples
y_train = np.random.randint(2, size=(1000, 1)) 

#fitting the model 1000/32 sample per epoch
model.fit(X_train, y_train, epochs=10, batch_size=32) 

#testing
X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 
loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}') 
print(f'Test accuracy: {accuracy}') 