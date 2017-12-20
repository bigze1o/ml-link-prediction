import pandas as pd
import numpy as np

def ReadData(pathFile):
    Data = pd.read_csv(pathFile)
    DataFull = Data.iloc[:, :].values
    return DataFull

def ReadDataNoWeight(pathFile):
    Data = pd.read_csv(pathFile)
    DataNoWeight = Data.iloc[:, :-1].values
    return DataNoWeight