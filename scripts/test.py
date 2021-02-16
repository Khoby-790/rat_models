import pandas as pd
import numpy as np


excel_data = pd.read_csv(
    r'C:\\Users\\Khoby\Downloads\\march18_myspeed.csv\\march18_myspeed.csv')
# print(excel_data);


actual_set = pd.DataFrame(excel_data, columns=["Technology", "Test_type",
                                               "Data Speed(Mbps)", "Signal_strength"])


fourG_data = actual_set[actual_set["Technology"] != "3G"]
threeG_data = actual_set[actual_set["Technology"] == "3G"]
fourG_data = fourG_data[fourG_data["Signal_strength"] != "na"]
threeG_data = actual_set[actual_set["Signal_strength"] != "na"]

fourG_data["status"] = np.where(np.absolute(fourG_data["Signal_strength"]) > 30, 0, np.where(
    np.absolute(fourG_data["Signal_strength"]) > 15, 1, 2
))


threeG_data["status"] = np.where(np.absolute(threeG_data["Signal_strength"]) > 30, 0, np.where(
    np.absolute(threeG_data["Signal_strength"]) > 15, 1, 2
))

threeG_data.to_csv("../data/3g_data.csv")
fourG_data.to_csv("../data/4g_data.csv")

# print(fourG_data)
