import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from lib import Lib
import numpy as np
import datetime

data = pd.read_csv("LTE_data.csv", usecols=[
                   "Lat", "Long", "Number_of_Users", "Radio_CQI_Distribution"])


data = pd.DataFrame(data)
# data["status"] = np.random.randint(1, 3, size=data.shape[0])
data["status"] = np.where(data["Radio_CQI_Distribution"] / 10000 > 30, 0, np.where(
    data["Radio_CQI_Distribution"] / 10000 > 15, 1, 2
))

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

# # print(Y)

# Creating the model
model = keras.models.load_model("lte_model.h5")
if(model):
    print("model Found")
    pass
else:
    print("Creating new model")
    model = keras.Sequential([
        normalize,
        layers.Dense(5, activation="relu"),
        layers.Dense(5, activation="softmax"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


model.fit(train_data, train_labels, epochs=1000)

test_loss, test_acc = model.evaluate(test_data, test_labels)


print("Test Accuracy: ", test_acc)
print("Test loss: ", test_loss)

model.save("lte_model.h5")
