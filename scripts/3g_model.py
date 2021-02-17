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

four_g_data = pd.read_csv("../data/3g_data.csv")

four_g_data = pd.DataFrame(four_g_data)

four_g_data["Test_type"] = np.where(
    (four_g_data.Test_type == 'Upload'), 1, four_g_data.Test_type)
four_g_data["Technology"] = np.where(
    (four_g_data.Technology == '4G'), 1, four_g_data.Technology)
four_g_data["Test_type"] = np.where(
    (four_g_data.Test_type == 'Download'), 2, four_g_data.Test_type)
four_g_data["Technology"] = np.where(
    (four_g_data.Technology == '3G'), 2, four_g_data.Technology)

# upload = 1 = "call"
# Download = 2 = "Streaming"

# print(four_g_data)

four_g_data["status"] = np.where(four_g_data["Signal_strength"] >= -85, 1, 0)


head = len(four_g_data) * 0.7
tail = len(four_g_data) * 0.3

X = four_g_data[["Signal_strength", "Test_type", "Data Speed(Mbps)"]]
Y = four_g_data[["status"]]
train_data_4g = X.head(int(head))
test_data_4g = X.tail(int(tail))
train_labels_4g = Y.head(int(head))
test_labels_4g = Y.tail(int(tail))

print("before Array")
print(train_data_4g)

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

train_data_4g = np.asarray(train_data_4g).astype(np.int)
train_labels_4g = np.asarray(train_labels_4g).astype(np.int)

model_4g.fit(train_data_4g, train_labels_4g, epochs=10)

test_data_4g = np.asarray(test_data_4g).astype(np.int)
test_labels_4g = np.asarray(test_labels_4g).astype(np.int)

loss, accuracy = model_4g.evaluate(test_data_4g, test_labels_4g)