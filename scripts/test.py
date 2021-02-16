import pandas as pd
import numpy as np


excel_data = pd.read_csv(
    r'C:\\Users\\Khoby\Downloads\\march18_myspeed.csv\\march18_myspeed.csv')
# print(excel_data);


actual_set = pd.DataFrame(excel_data, columns=["Technology", "Test_type",
                                               "Data Speed(Mbps)", "Signal_strength"]).reset_index(drop=True)


actual_set = actual_set[actual_set["Signal_strength"] != "na"]

fourG_data = actual_set[actual_set["Technology"] != "3G"]
threeG_data = actual_set[actual_set["Technology"] == "3G"]

# data_4g = np.array(fourG_data["Signal_strength"])

# fourG_data["Signal_strength"] = pd.Series(fourG_data["Signal_strength"]).abs()

print(fourG_data["Signal_strength"])

# threeG_data.to_csv("../data/3g_data.csv")
# fourG_data.to_csv("../data/4g_data.csv")

# print(fourG_data)
