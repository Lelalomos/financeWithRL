import pandas as pd
import os
from functions import predict_lstm
import numpy as np
import torch
from utils import normalization_data

df = pd.read_parquet(os.path.join(os.getcwd(),'dataset','original_dataset.parquet'))
normalize = normalization_data()

# rmse: 0.14743329345924763
def benchmark_rsi():
    # split data
    df_bench = df.copy()
    df_bench = df_bench[['Open', 'High', 'Low', 'Close', 'Volume','rsi_14']]
    df_bench['Close_norm'] = normalize.normalize_minmax_1d_data(df_bench['Close'].to_numpy())
    df_bench['rsi_14_norm'] = normalize.normalize_minmax_1d_data(df_bench['rsi_14'].to_numpy())
    df_bench = df_bench.iloc[-200000:,:]
    rsi_model_path = os.path.join(os.getcwd(),'saved_model','rsi_14_model.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device:",device)
    df_bench['rsi_pred'] = df_bench.apply(lambda x :predict_lstm(x['Close_norm'], rsi_model_path, device),axis=1)
    # df_bench['rsi_distance']
    y_pred = df_bench['rsi_pred'].to_numpy()
    y_test = df_bench['rsi_14_norm']
    rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
    print("rmse:",rmse)
    y_rest = y_pred - y_test
    print("y_rest:",sum(y_rest))
    

if __name__ == "__main__":
    benchmark_rsi()