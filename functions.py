from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from model import LSTMModel
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd

def search_null_value(df):
    columns_with_null = df.columns[df.isnull().any()]
    print('columns_with_null:',columns_with_null)
    for column in columns_with_null:
        print(f"Null values in column '{column}':")
        print(df[df[column].isnull()])
    
def prepare_data_forlstm(data):
    data = data.dropna(how = "any", ignore_index = True)
    data = data.sort_values(["Date"]).reset_index(drop=True)
    return np.array(data['Close']), np.array(data['rsi_14'])

def detect_outliers_iqr(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    return len(np.where((data < lower_bound) | (data > upper_bound))[0])


def predict_rsi(data, model_path):
    if pd.isna(data['rsi_14']) or data['rsi_14'] in [None,np.nan,""]:
        close = float(data['Close'])
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        x = np.array([[[close]]])
        x = torch.tensor(x)
        x = x.to(torch.float32)
        return model(x).item()
    else:
        return data['rsi_14']