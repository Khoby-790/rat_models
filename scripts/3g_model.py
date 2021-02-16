import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import TensorBoard
from lib import Lib
import numpy as np
import h5py
import datetime

data_3g = pd.read_csv("../data/3g_data.csv", usecols=[
    "Technoology", "Test_type", "Data Speed(Mbps)", "Signal_strength"])


data = pd.DataFrame(data_3g)
data["status"] = np.where(data["Signal_strength"] > 30, 0, np.where(
    data["Signal_strength"] > 15, 1, 2
))


xlib = Lib()


X = data[["Technoology", "Test_type", "Data Speed(Mbps)", "Signal_strength"]]
Y = data[["status"]]


train_data, test_data = xlib.split_by_fractions(data, [0.7, 0.3])
train_labels = train_data["status"]
test_labels = test_data["status"]

# train_data =>> array
train_data = np.array(train_data)

# normalization
normalize = preprocessing.Normalization()
normalize.adapt(train_data)



# File Paths
modelPath = "../lte_model/3g_model.json"
weightsPath = "../lte_model/3g_model.h5"


print("Creating Models")
print("---- 3G Models -----")
model_3g = keras.Sequential([
    normalize,
    layers.Dense(16, activation="relu"),
    layers.Dense(18, activation="softmax"),
    layers.Dense(3, activation="softmax"),
])
print("---- 4G Models -----")
model_4g = keras.Sequential([
    normalize,
    layers.Dense(16, activation="relu"),
    layers.Dense(18, activation="softmax"),
    layers.Dense(3, activation="softmax"),
])

model_3g.compile(optimizer="adam",
                 loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])
model_4g.compile(optimizer="adam",
                 loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])

print("---- TRAINING ZONE -----")
model_3g.fit(train_data, train_labels, epochs=10)
model_4g.fit(train_data, train_labels, epochs=10)
test_loss, test_acc = model_3g.evaluate(test_data, test_labels)

# save model
xlib.save_model(model_3g, modelpath=modelPath, weightspath=weightsPath)

print("Test Accuracy: ", test_acc)
print("Test loss: ", test_loss)
