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