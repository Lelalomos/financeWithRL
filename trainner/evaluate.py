import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("/app")
from model.model import LSTMModel, LSTMModelwithAttention
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class evaluate_model:
    def __init__(self, batch_size = 64, shuffle_train = False, model_name = None):
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.model_name = model_name

    def evaluate(self, test_X, test_Y, stock_id, group_id, day_id, month_id, feature_dim, num_stocks, num_group, num_day, num_month):
        test_X = torch.tensor(test_X.to_numpy(), dtype=torch.float32).to(device)
        test_X = test_X.unsqueeze(1)
        test_Y = torch.tensor(np.array(test_Y), dtype=torch.float32).to(device)

        stock_tensor = torch.tensor(stock_id, dtype=torch.long)
        group_tensor = torch.tensor(group_id, dtype=torch.long)
        month_tensor = torch.tensor(month_id, dtype=torch.long)
        day_tensor = torch.tensor(day_id, dtype=torch.long)

        stock_tensor = stock_tensor.unsqueeze(1)
        group_tensor = group_tensor.unsqueeze(1)
        month_tensor = month_tensor.unsqueeze(1)
        day_tensor = day_tensor.unsqueeze(1)

        print('shape:',test_X.shape, stock_tensor.shape)
        test_dataset = TensorDataset(test_X, stock_tensor, group_tensor, month_tensor, day_tensor, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train)
        
        if config.MODEL == "lstm":
            self.lstm_model = LSTMModel(
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config
            ).to(device)
        elif config.MODEL == "lstm_with_attention":
            self.lstm_model = LSTMModelwithAttention(
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config
            ).to(device)

        self.lstm_model.load_state_dict(torch.load(self.model_name))

        # Test the model
        self.lstm_model.eval()
        test_loss = 0
        predictions, actuals = [], []
        criterion = nn.HuberLoss(delta=config.LSTM_PARAMS['delta'])
        with torch.no_grad():
            for inputs, stock_tensor, group_tensor, month_tensor, day_tensor, labels in test_loader:
                inputs, stock_tensor, group_tensor, month_tensor, day_tensor, labels = inputs.to(device), stock_tensor.to(device), group_tensor.to(device), month_tensor.to(device), day_tensor.to(device), labels.to(device)
                outputs = self.lstm_model(stock_tensor, group_tensor, day_tensor, month_tensor, inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                predictions.append(outputs.cpu().numpy())
                actuals.append(labels.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        print(f"predictions: {predictions[0]}")
        print(f"actuals: {actuals[0]}")

        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        print(f"Evaluation Results:")
        print(f"  MSE  : {mse:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        print(f"  MAE  : {mae:.6f}")
        print(f"  R²   : {r2:.6f}")


def evaluate_lstm(model_name = None,df_test = pd.read_parquet(os.path.join(os.getcwd(),"data","test_dataset.parquet"))):
    num_stocks = len(df_test['tic_id'].unique())
    num_group = len(df_test['group_id'].unique())
    num_month = len(df_test['month'].unique())+1
    num_day = len(df_test['day'].unique())+1
    list_except_group = [columns for columns in list(df_test.columns) if columns not in ['tic_id','group_id','month','day']]
    feature = df_test[list_except_group]
    y_val = feature[['pre_7']]
    list_except_group = [columns for columns in list_except_group if columns not in ['pre_7']]
    X_val = feature[list_except_group]
    feature_dim = len(X_val.columns)

    stock_tensor = df_test['tic_id'].astype(int).to_list()
    group_tensor = df_test['group_id'].astype(int).to_list()
    month_tensor = df_test['month'].astype(int).to_list()
    day_tensor = df_test['day'].astype(int).to_list()

    if model_name is not None:
        eval_model = evaluate_model(model_name=model_name)
        eval_model.evaluate(X_val, y_val, stock_tensor, group_tensor, day_tensor, month_tensor, feature_dim, num_stocks, num_group, num_day, num_month)
            

if __name__ == "__main__":
    evaluate_lstm("saved_model/20250313_lstm_model.pth")