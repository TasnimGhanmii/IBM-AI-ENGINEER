import tensorflow as tf
from tensorflow.keras.layers import Layer,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax
from tensorflow.keras.utils import plot_model
import numpy as np 

#defining a custom layer that ineherits from Layer 
class CustomDenseLayer(Layer):
    #defines number of units/neurons in a layer if not provided the deault is 32
    def __init__(self, units=128):
        #to ensure any inits performed by the base class Layer is executed
        super(CustomDenseLayer, self).__init__()
        self.units = units
    
    #defines the weights and biases of the layer
    #called auto when the layer is FIRST used
    def build(self, input_shape):
        #weight matrix
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        #bias vector
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    #defines the computation performed by the layer when the layer is called on some input data    
    def call(self, inputs):
               #applies the rlu activation fct to the result
                          #calculates matrix mult of the input and weights
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
    

#integrating the custom layer into a model
model = Sequential([
    CustomDenseLayer(128),
    Dropout(rate=0.5),
    CustomDenseLayer(10),  # Hidden layer with ReLU activation
    Softmax()              # Output layer with Softmax activation for multi-class classification
])

#compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building:")
model.summary()

# Build the model to show parameters
model.build((1000, 20))
print("\nModel summary after building:")
model.summary()

# Generate random data 
x_train = np.random.random((1000, 20)) 
y_train = np.random.randint(10, size=(1000, 1)) 

# Convert labels to categorical one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
                                       #number of samples 
model.fit(x_train, y_train, epochs=10, batch_size=32)

#visualisation
#plot_model(model,to_file='model_archi.png',show_shapes=True,show_layer_names=True)