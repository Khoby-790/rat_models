import tensorflow as tf  # tehsorflow
from tensorflow import keras  # keras
import matplotlib.pyplot as plt  # pyplot
# from matplotlib.image import imread # imread that reads images
import pandas as pd

# lte_data = pd.read_csv("LTE_data.csv")

# print(lte_data)

data = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


# plt.imshow(train_images[7])
# plt.show()

model = keras.models.load_model("model.h5")

if(model):
    print("model Found")
    pass
else:
    print("Creating new model")
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])


model.fit(train_images, train_labels, epochs=5)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy: ", test_acc)
print("Test loss: ", test_loss)

model.save("model.h5")
