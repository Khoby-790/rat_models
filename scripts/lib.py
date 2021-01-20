import pandas as pd
from tensorflow.keras.models import model_from_json
import os


class Lib:
    def load_data(self):
        pass

    def prepareData(self, status):
        lte_data = pd.read_csv("LTE_data.csv")
        data = pd.DataFrame(lte_data)
        for i, row in data.iterrows():
            print(row, i)

    def save_model(self, model, modelpath, weightspath):
        # Convert modelto json
        model.save_weights(weightspath)
        model = model.to_json()
        json_file = open(modelpath, "w+")
        json_file.write(model)
        # serialize weights to HDF5
        print("Model Saved")

    def load_model(self, modelpath, weightspath):
        if(os.path.exists(modelpath) and os.path.exists(weightspath)):
            json_file = open(modelpath, 'r')
            loaded_model_json = json_file.read()
            print(loaded_model_json)
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(weightspath)
            print("Loaded model from disk")
            loaded_model.summary()
            return loaded_model
        else:
            return None
