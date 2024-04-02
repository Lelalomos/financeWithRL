import pandas as pd
import yfinance as yf
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from stockstats import StockDataFrame as Sdf
import numpy as np
from scipy.stats import zscore
import torch
from model import LSTMModel
import torch.nn as nn
import matplotlib.pyplot as plt

def search_null_value(df):
    columns_with_null = df.columns[df.isnull().any()]
    print('columns_with_null:',columns_with_null)
    for column in columns_with_null:
        print(f"Null values in column '{column}':")
        print(df[df[column].isnull()])
        
def fill_nan_value(df):
    pass
    
def pre_data(data):
    if 'Close' in data.columns:
        data =data.drop(['Close'],axis = 1)
    if 'Adj Close' in data.columns:
        data = data.rename(columns={"Adj Close":"Close"})
        
    return data
     
def prepare_data_forlstm(data):
    data = data.dropna(how = "any", ignore_index = True)
    data = data.sort_values(["Date"]).reset_index(drop=True)
    return np.array(data['Close']), np.array(data['rsi_14'])

def normalize_minmax_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rsi_scaled = scaler.fit_transform(data.reshape(-1, 1))
    return rsi_scaled

def normalize_robustscaler(data):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(data.reshape(-1, 1))
    return X_scaled

def cal_zscore(list_data):
    z_scores = zscore(list_data)
    outlier_indices = np.where(np.abs(z_scores) >= 3)[0]
    filtered_data = list_data[np.abs(z_scores) < 3]
    return filtered_data, outlier_indices

def detect_outliers_iqr(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    return len(np.where((data < lower_bound) | (data > upper_bound))[0])
    
def download_data(ticker_list=[], start_date=None, end_date=None, interval = "1d", proxy = None, engine_download = "yahoo", logging = None):
    df_data = pd.DataFrame()
    if len(ticker_list) > 0:
        for tic in ticker_list:
            if engine_download == "yahoo":
                logging.info(f"download {tic} data from {engine_download}")
                temp_df = yf.download(
                        tic, start=start_date, end=end_date, proxy=proxy, interval = interval
                    )
                logging.info(f"download {tic} data from {engine_download}")
                
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                df_data = pd.concat([df_data, temp_df])
        df_data = df_data.reset_index()
    return df_data


def add_technical_indicator(data, tech_indicator_list):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["Date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["Date"] = df[df.tic == unique_ticker[i]][
                        "Date"
                    ].to_list()
                    # indicator_df = indicator_df.append(
                    #     temp_indicator, ignore_index=True
                    # )
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "Date", indicator]], on=["tic", "Date"], how="left"
            )
        df = df.sort_values(by=["Date", "tic"])
        return df
    
def train_lstm(train_X, train_Y, test_X, test_Y, epochs, threshold_loss):
    train_X = torch.from_numpy(train_X).unsqueeze(2).float()
    train_Y = torch.from_numpy(train_Y).float()
    test_X = torch.from_numpy(test_X).unsqueeze(2).float()
    test_Y = torch.from_numpy(test_Y).float()
    
    train_dataset = TensorDataset(train_X, train_Y)
    test_dataset = TensorDataset(test_X, test_Y)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=True)
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train the model
    list_loss = []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            list_loss.append(loss.item())
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                
            # if loss.item() < threshold_loss:
            #     print(f'Loss is below threshold ({threshold_loss}), saving the model...')
            #     torch.save(model.state_dict(), 'model.pth')
            #     break

    # Test the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Test MSE: {test_loss}')
    
    torch.save(model.state_dict(), 'model.pth')
    
    plt.plot(list_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss_plot.png')  # Save the plot as an image file
    plt.close()