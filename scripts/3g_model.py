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

four_g_data["Test_type"] = np.where(
    (four_g_data.Test_type == 'Upload'), 1, four_g_data.Test_type)
four_g_data["Technology"] = np.where(
    (four_g_data.Test_type == '4G'), 1, four_g_data.Test_type)
four_g_data["Test_type"] = np.where(
    (four_g_data.Test_type == 'Download'), 2, four_g_data.Test_type)
four_g_data["Technology"] = np.where(
    (four_g_data.Test_type == '3G'), 2, four_g_data.Test_type)

# print(four_g_data)

four_g_data["status"] = np.where(np.absolute(four_g_data["Signal_strength"]) > 85, 0, np.where(
    np.absolute(four_g_data["Signal_strength"]) > 59, 1, np.where(np.absolute(four_g_data["Signal_strength"]) > 50,
                                                                  2, np.where(np.absolute(
                                                                      four_g_data["Signal_strength"]) > 40, 3, 4)
                                                                  )
))


head = len(four_g_data) * 0.8
tail = len(four_g_data) * 0.2

X = four_g_data[["Signal_strength", "Test_type", "Data Speed(Mbps)"]]
Y = four_g_data[["status"]]
train_data_4g = X.head(int(head))
test_data_4g = X.tail(int(tail))
train_labels_4g = Y.head(int(head))
test_labels_4g = Y.tail(int(tail))
# train_data_4g =

# train_data_4g = pd.DataFrame(train_data_4g)
train_data_4g = np.array(train_data_4g)

print("After Array")
print(train_data_4g)

normalize = preprocessing.Normalization()
normalize.adapt(train_data_4g)

print("After normalize")
print(train_labels_4g)

print("Creating new 4G model")
model_4g = keras.Sequential([
    normalize,
    layers.Dense(16, activation="relu"),
    layers.Dense(18, activation="softmax"),
    layers.Dense(3, activation="softmax"),
])
model_4g.compile(optimizer="adam",
                 loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])


model_4g.fit(train_data_4g, train_labels_4g, epochs=10)
