import numpy as np 
import pandas as pd 
import tensorflow as tf 
import requests
#used for scaling
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


# Create a synthetic stock price dataset
np.random.seed(42)
data_length = 2000
#generates evenly spaced numbers over a specified range 100 start ands at 200 & data_length is nb of samples
trend = np.linspace(100, 200, data_length)
#Adds Gaussian noise with a mean of 0 and standard deviation of 2
noise = np.random.normal(0, 2, data_length)
synthetic_data = trend + noise

# Create a DataFrame and save the synthetic data  as 'stock_prices.csv'
data = pd.DataFrame(synthetic_data, columns=['Close'])
data.to_csv('stock_prices.csv', index=False)
print("Synthetic stock_prices.csv created and loaded.")

# Load the dataset 
data = pd.read_csv('stock_prices.csv') 
#Extracts the 'Close' column as a NumPy array
data = data[['Close']].values 

# Normalize the data
#created scaler obj scaling data to the range of [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
#fits the scaler to the data and transforms the data to the specified range.
data = scaler.fit_transform(data)

# Prepare the data for training
def create_dataset(data, time_step=1):
    X, Y = [], []

    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X, Y = create_dataset(data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

print("Shape of X:", X.shape) 
print("Shape of Y:", Y.shape) 
