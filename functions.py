import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from stockstats import StockDataFrame as Sdf
import numpy as np
from scipy import stats
import torch
import torch.nn as nn

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
    rsi_scaled = scaler.fit_transform(data)
    return rsi_scaled

def normalize_robustscaler(data):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(data.reshape(-1, 1))
    return X_scaled

def cal_zscore(list_data):
    return len(np.where(np.abs(stats.zscore(list_data))>3))

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
        df = df.sort_values(by=["tic", "Date"])
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
    
def train_lstm(train_X, train_Y):
    train_X = torch.from_numpy(train_X).unsqueeze(2).float()
    train_Y = torch.from_numpy(train_Y).float()
    test_X = torch.from_numpy(test_X).unsqueeze(2).float()
    test_Y = torch.from_numpy(test_Y).float()