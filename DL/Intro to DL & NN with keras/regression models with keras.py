import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input

# 1. load & clean -------------------------------------------------------------
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv"
df = pd.read_csv(url)
predictors = df.drop(columns=["Strength"])
target = df["Strength"]

# 2. normalise ----------------------------------------------------------------
pred_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = pred_norm.shape[1]

# 3. build model --------------------------------------------------------------
def regression_model():
    model = Sequential([
        Input(shape=(n_cols,)),
        Dense(50, activation="relu"),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

model = regression_model()

# 4. train --------------------------------------------------------------------
history = model.fit(pred_norm, target,
                    validation_split=0.3,
                    epochs=100,
                    verbose=2)

# 5. quick plot ---------------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE")
plt.savefig("keras_regression_loss.png"); plt.show()