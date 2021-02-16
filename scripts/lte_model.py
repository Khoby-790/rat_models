import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from lib import Lib
import numpy as np
import h5py
import datetime

data = pd.read_csv("../data/LTE_data.csv", usecols=[
                   "Lat", "Long", "Number_of_Users", "Radio_CQI_Distribution"])
four_g_data = pd.read_csv("../data/4g_data.csv")

data = pd.DataFrame(data)
four_g_data = pd.DataFrame(four_g_data)
# data["status"] = np.random.randint(1, 3, size=data.shape[0])
data["status"] = np.where(data["Radio_CQI_Distribution"] / 10000 > 30, 0, np.where(
    data["Radio_CQI_Distribution"] / 10000 > 15, 1, 2
))


xlib = Lib()

# status = [1, 0, 2]
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


X = data[["Lat", "Long", "Number_of_Users", "Radio_CQI_Distribution"]]
Y = data[["status"]]
train_data = X.head(1109)
test_data = X.tail(476)
train_labels = Y.head(1109)
test_labels = Y.tail(476)

# print(data.sort_values(['Radio_CQI_Distribution'], ascending=True))

# train_data =>> array
train_data = np.array(train_data)

# normalization
normalize = preprocessing.Normalization()
normalize.adapt(train_data)


# input_shape = X.shape

# File Paths
modelPath = "../lte_model/lte_model.json"
weightsPath = "../lte_model/lte_model.h5"


# # print(Y)
# model = xlib.load_model(modelPath, weightsPath)

# # Creating the model
# if(model):
#     print("model Found")
#     pass
# else:
print("Creating new model")
model_lte = keras.Sequential([
    normalize,
    layers.Dense(16, activation="relu"),
    layers.Dense(18, activation="softmax"),
    layers.Dense(3, activation="softmax"),
])
model_lte.compile(optimizer="adam",
                  loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])



print(train_data)

model_lte.fit(train_data, train_labels, epochs=10)
test_loss, test_acc = model_lte.evaluate(test_data, test_labels)

# model.save_weights()

# save model
xlib.save_model(model_lte, modelpath=modelPath, weightspath=weightsPath)


print("Test Accuracy: ", test_acc)
print("Test loss: ", test_loss)
