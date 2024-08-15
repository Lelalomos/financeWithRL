import pandas as pd
import os
from functions import predict_lstm_single
import numpy as np
import torch
from utils import normalization_data

df = pd.read_parquet(os.path.join(os.getcwd(),'dataset','original_dataset.parquet'))
normalize = normalization_data()

# rmse: 0.14743329345924763
def benchmark_single(column_input, column_label):
    # split data
    df_bench = df.copy()
    df_bench = df_bench[[column_input,column_label]]
    df_bench[f'{column_input}_norm'] = normalize.normalize_minmax_1d_data(df_bench[column_input].to_numpy())
    df_bench[f'{column_label}_norm'] = normalize.normalize_minmax_1d_data(df_bench[column_label].to_numpy())
    df_bench = df_bench.iloc[-200:,:]
    rsi_model_path = os.path.join(os.getcwd(),'saved_model',f'{column_label}_model.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device:",device)
    df_bench[f'{column_label}_pred'] = df_bench.apply(lambda x :predict_lstm_single(x[f'{column_input}_norm'], rsi_model_path, device),axis=1)
    # df_bench['rsi_distance']
    y_pred = df_bench[f'{column_label}_pred'].to_numpy()
    y_test = df_bench[f'{column_label}_norm']
    rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
    print(f"rmse {column_label}:",rmse)
    y_rest = y_pred - y_test
    print(f"y_rest {column_label}:",sum(y_rest))
    
    

if __name__ == "__main__":
    # benchmark rsi_14
    benchmark_single('Close','rsi_14')
    # benchmark stochrsi_14
    benchmark_single('Close','stochrsi_14')
    # benchmark tema_200
    benchmark_single('Close','tema_200')
    # 