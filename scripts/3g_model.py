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

data = pd.read_csv("../data/3g_data.csv", usecols=[
                   "Technoology", "Test_type", "Data Speed(Mbps)", "Signal_strength"])


data = pd.DataFrame(data)
# data["status"] = np.random.randint(1, 3, size=data.shape[0])
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


# input_shape = X.shape

# File Paths
modelPath = "../lte_model/3g_model.json"
weightsPath = "../lte_model/3g_model.h5"


# # print(Y)
model = xlib.load_model(modelPath, weightsPath)

# Creating the model
if(model):
    print("model Found")
    pass
else:
    print("Creating new model")
    model = keras.Sequential([
        normalize,
        layers.Dense(16, activation="relu"),
        layers.Dense(18, activation="softmax"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])


model.fit(train_data, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_data, test_labels)

# model.save_weights()

# save model
xlib.save_model(model, modelpath=modelPath, weightspath=weightsPath)


print("Test Accuracy: ", test_acc)
print("Test loss: ", test_loss)
