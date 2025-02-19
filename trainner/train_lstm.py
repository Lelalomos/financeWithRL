import torch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sys.path.append("/app")

from model.model import LSTMModel
import config
import os

class train_lstm:
    def __init__(self,
                 epochs = 200, 
                 debug_loss = True,
                 threshold_loss = 0.02, 
                 batch_size = 64, 
                 shuffle_train = False, 
                 shuffle_test= False,
                 path_save_loss = "loss.jpg",
                 path_save_model = "model.pth"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:',self.device)
        print("epochs:",epochs)
        self.epochs = epochs
        self.threshold_loss = threshold_loss
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test
        self.debug_loss = debug_loss
        self.list_loss = []
        self.path_save_loss = path_save_loss
        self.path_save_model = path_save_model
        
    
    def train(self, train_X, train_Y, stock_id, group_id, day_id, month_id, feature_dim, num_stocks, num_group, num_day, num_month):
        train_X = torch.tensor(train_X.to_numpy(), dtype=torch.float32).to(self.device)
        train_X = train_X.unsqueeze(1)
        train_Y = torch.tensor(np.array(train_Y), dtype=torch.float32).to(self.device)

        stock_tensor = torch.tensor(stock_id, dtype=torch.long)
        group_tensor = torch.tensor(group_id, dtype=torch.long)
        month_tensor = torch.tensor(month_id, dtype=torch.long)
        day_tensor = torch.tensor(day_id, dtype=torch.long)

        stock_tensor = stock_tensor.unsqueeze(1)
        group_tensor = group_tensor.unsqueeze(1)
        month_tensor = month_tensor.unsqueeze(1)
        day_tensor = day_tensor.unsqueeze(1)

        print('shape:',train_X.shape, stock_tensor.shape)
        train_dataset = TensorDataset(train_X, stock_tensor, group_tensor, month_tensor, day_tensor, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train)

        lstm_model = LSTMModel(
            feature_dim,
            num_stocks,
            num_group,
            num_day,
            num_month,
            config
        ).to(self.device)
        optimizer = torch.optim.Adam(lstm_model.parameters())
        criterion = nn.HuberLoss(delta=config.LSTM_PARAMS['delta'])
        
        print("-"*20,"Model","-"*20)
        for name, param in lstm_model.state_dict().items():
            print(f"{name}: {param.shape}")
        print("-"*50)
        
        # Train the model
        for epoch in range(self.epochs):
            val_loss = 0
            for inputs, stock_tensor, group_tensor, month_tensor, day_tensor, labels in train_loader:
                optimizer.zero_grad()
                inputs, stock_tensor, group_tensor, month_tensor, day_tensor, labels = inputs.to(self.device), stock_tensor.to(self.device), group_tensor.to(self.device), month_tensor.to(self.device), day_tensor.to(self.device), labels.to(self.device)
                outputs = lstm_model(stock_tensor, group_tensor, day_tensor, month_tensor, inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                self.list_loss.append(loss.item())
                val_loss += loss.item()
                    
                if self.debug_loss:
                    if loss.item() < self.threshold_loss:
                        print(f'Loss is below threshold ({self.threshold_loss}), saving the model...')
                        torch.save(lstm_model.state_dict(), 'model.pth')
                        break

            avg_loss = val_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

        
    def eval(self, test_X, test_Y, stock_id, group_id, day_id, month_id, feature_dim, num_stocks, num_group, num_day, num_month):
        test_X = torch.tensor(test_X.to_numpy(), dtype=torch.float32).to(self.device)
        test_X = test_X.unsqueeze(1)
        test_Y = torch.tensor(np.array(test_Y), dtype=torch.float32).to(self.device)
        
        stock_tensor = torch.tensor(stock_id, dtype=torch.long)
        group_tensor = torch.tensor(group_id, dtype=torch.long)
        month_tensor = torch.tensor(month_id, dtype=torch.long)
        day_tensor = torch.tensor(day_id, dtype=torch.long)

        stock_tensor = stock_tensor.unsqueeze(1)
        group_tensor = group_tensor.unsqueeze(1)
        month_tensor = month_tensor.unsqueeze(1)
        day_tensor = day_tensor.unsqueeze(1)
        
        lstm_model = LSTMModel(feature_dim,
            num_stocks,
            num_group,
            num_day,
            num_month,
            config).to(self.device)
        
        test_dataset = TensorDataset(test_X, stock_tensor, group_tensor, month_tensor, day_tensor, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle_test)
        
        # Test the model
        lstm_model.eval()
        test_loss = 0
        predictions, actuals = [], []
        criterion = nn.HuberLoss(delta=config.LSTM_PARAMS['delta'])
        with torch.no_grad():
            for inputs, stock_tensor, group_tensor, month_tensor, day_tensor, labels in test_loader:
                inputs, stock_tensor, group_tensor, month_tensor, day_tensor, labels = inputs.to(self.device), stock_tensor.to(self.device), group_tensor.to(self.device), month_tensor.to(self.device), day_tensor.to(self.device), labels.to(self.device)
                outputs = lstm_model(stock_tensor, group_tensor, day_tensor, month_tensor, inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                predictions.append(outputs.cpu().numpy())
                actuals.append(labels.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        print(f"Evaluation Results:")
        print(f"  MSE  : {mse:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        print(f"  MAE  : {mae:.6f}")
        print(f"  R²   : {r2:.6f}")
            
        
    def plot_image(self, save_image):
        plt.plot(self.list_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(save_image)  # Save the plot as an image file
        plt.close()
    
    def export_model(self, save_model):
        torch.save(self.model.state_dict(), save_model)
        

if __name__ == "__main__":
    # print("config f",config.LSTM_PARAMS)
    # prepare data
    df_train = pd.read_parquet(os.path.join(os.getcwd(),"data","train_dataset.parquet"))
    num_stocks = len(df_train['tic_id'].unique())
    num_group = len(df_train['group_id'].unique())
    num_month = len(df_train['month'].unique())+1
    num_day = len(df_train['day'].unique())+1
    list_except_group = [columns for columns in list(df_train.columns) if columns not in ['tic_id','group_id','month','day']]
    feature = df_train[list_except_group]
    y_val = feature[['pre_7']]
    list_except_group = [columns for columns in list_except_group if columns not in ['pre_7']]
    X_val = feature[list_except_group]
    feature_dim = len(X_val.columns)

    stock_tensor = df_train['tic_id'].astype(int).to_list()
    group_tensor = df_train['group_id'].astype(int).to_list()
    month_tensor = df_train['month'].astype(int).to_list()
    day_tensor = df_train['day'].astype(int).to_list()

    lstm = train_lstm(epochs=10)
    lstm.train(X_val, y_val, stock_tensor, group_tensor, day_tensor, month_tensor, feature_dim, num_stocks, num_group, num_day, num_month)

    