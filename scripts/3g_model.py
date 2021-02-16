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


four_g_data = pd.read_csv("../data/4g_data.csv")

four_g_data = four_g_data[[""]]

four_g_data.loc[(four_g_data.Test_type == "Upload"),"Test_type"] = 1
four_g_data.loc[(four_g_data.Test_type == "Download"),"Test_type"] = 2

print(four_g_data)