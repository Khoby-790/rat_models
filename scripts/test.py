import pandas as pd


excel_data = pd.read_csv(
    r'C:\\Users\\Khoby\Downloads\\march18_myspeed.csv\\march18_myspeed.csv')
# print(excel_data);


actual_set = pd.DataFrame(excel_data, columns=["Technology", "Test_type",
                                               "Data Speed(Mbps)", "Signal_strength"])

print(actual_set)

fourG_data = actual_set[actual_set["Technology"] == "4G"]

print(fourG_data)