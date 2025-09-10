import requests
import zipfile
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.layers import Dropout,MaxPooling2D,GlobalAveragePooling2D, Input
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


