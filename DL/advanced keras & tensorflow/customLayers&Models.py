import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential 
import numpy as np 


#defining layers
#input that expects a vector of length 20
input_layer = Input(shape=(20,))

#creates fully connected layer with 64 units and ReLU activation function
hidden_layer1 = Dense(64, activation='relu')(input_layer) 
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

#output layer ,dense layer, with 1 unit and sigmoid activation fct
output_layer = Dense(1, activation='sigmoid')(hidden_layer2) 

#creating the model
#creates keras model connecting the input and output through hidden layers
model = Model(inputs=input_layer, outputs=output_layer)
#provides a summary of the model, showing the layers, their shapes, and the number of parameters
model.summary()
#compiling model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#training the model
X_train = np.random.rand(1000, 20) 
y_train = np.random.randint(2, size=(1000, 1)) 
model.fit(X_train, y_train, epochs=10, batch_size=32) 

#test & evaluate
X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 
loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}') 
print(f'Test accuracy: {accuracy}') 