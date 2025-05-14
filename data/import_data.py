from kaggle_loader import kaggleDataLoader
import os
import pandas as pd #type: ignore

def importData():
    dataset_path = kaggleDataLoader()

    csv_file_path = os.path.join (dataset_path, "training.1600000.processed.noemoticon.csv")
    column_name = ["target", "id", "date", "flag", "user", "tweet"]
    dat = pd.read_csv(csv_file_path, encoding='latin-1', header=None, names = column_name)

    return dat