import pandas as pd


class Lib:
    def load_data(self):
        pass

    def prepareData(self, status):
        lte_data = pd.read_csv("LTE_data.csv")
        data = pd.DataFrame(lte_data)
        for i, row in data.iterrows():
            print(row, i)
