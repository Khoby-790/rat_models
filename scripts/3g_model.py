import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from lib import Lib
import numpy as np
from decimal import Decimal
import h5py
import datetime

xlib = Lib()

four_g_data = pd.read_csv("../data/4g_data.csv")

four_g_data = pd.DataFrame(four_g_data)

four_g_data.loc[(four_g_data.Test_type == "Upload"), "Test_type"] = 1
four_g_data.loc[(four_g_data.Test_type == "Download"), "Test_type"] = 2


four_g_data["status"] = np.where(np.absolute(four_g_data["Signal_strength"]) > 85, 0, np.where(
    np.absolute(four_g_data["Signal_strength"]) > 59, 1, np.where(np.absolute(four_g_data["Signal_strength"]) > 50,
                                                                  2, np.where(np.absolute(
                                                                      four_g_data["Signal_strength"]) > 40, 3, 4)
                                                                  )
))


print(four_g_data)

train_data_4g, test_data_4g = xlib.split_by_fractions(four_g_data, [0.8, 0.2])
train_labels_4g = train_data_4g["status"]
test_labels_4g = test_data_4g["status"]

normalize_4g = preprocessing.Normalization()
normalize_4g.adapt(train_data_4g)


print("Creating new 4G model")
model_4g = keras.Sequential([
    normalize_4g,
    layers.Dense(16, activation="relu"),
    layers.Dense(18, activation="softmax"),
    layers.Dense(3, activation="softmax"),
])
model_4g.compile(optimizer="adam",
                 loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])