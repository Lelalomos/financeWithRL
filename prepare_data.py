from stockstats import StockDataFrame as Sdf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from utility import return_logs
import os
import yfinance as yf
import traceback
import config
from functions import predict_nanvalue_lstm, return_candle_pattern, predict_nanvalue_lstm_vwma, predict_nanvalue_lstm_ichimoku

class prepare_data:
    def __init__(self):
        os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
        self.logging = return_logs(os.path.join(os.getcwd(),'logs','prepare_data.log'))
        
    def download_data(self, ticker_list=[], start_date=None, end_date=None, interval = "1d", proxy = None, engine_download = "yahoo"):
        df_data = pd.DataFrame()
        if len(ticker_list) > 0:
            for tic in ticker_list:
                if engine_download == "yahoo":
                    temp_df = yf.download(
                            tic, start=start_date, end=end_date, proxy=proxy, interval = interval
                        )
                    self.logging.info(f"download {tic} data from {engine_download}")
                    
                temp_df["tic"] = tic
                if len(temp_df) > 0:
                    df_data = pd.concat([df_data, temp_df])
            df_data = df_data.reset_index()
            self.logging.info(f"{len(df_data.index)} rows downloaded")
        return df_data
    
    def pre_clean_data(self, dataframe):
        # drop close column and rename adj column to close column on dataframe
        self.logging.info(f"pre clean process")
        df = dataframe.copy()
        if 'Close' in df.columns:
            df = df.drop(['Close'],axis = 1)
        if 'Adj Close' in df.columns:
            df = df.rename(columns={"Adj Close":"Close"})
        return df
    
    def cal_zscore(self, list_data, threshold = 3):
        self.logging.info(f"filter base on cal zscore process")
        self.logging.info(f"len data: {len(list_data)}, threshold:{threshold}")
        z_scores = zscore(list_data)
        outlier_indices = np.where(np.abs(z_scores) >= threshold)[0]
        filtered_data = list_data[np.abs(z_scores) < threshold]
        return filtered_data, outlier_indices
    
    def cal_zscore_df(self, dataframe, threshold = 3):
        standardized_df = dataframe.apply(zscore)
        filtered_df = dataframe[(standardized_df < threshold) & (standardized_df > -threshold)].dropna()
        return filtered_df
    
    def add_technical_indicator(self, dataframe, tech_indicator_list):
        self.logging.info(f"add technical indicator into dataframe process, tech_indicator_list:{tech_indicator_list}")
        df = dataframe.copy()
        df = df.sort_values(by=["Date","tic"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["Date"] = df[df.tic == unique_ticker[i]]["Date"].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    self.logging.error(f"error add indicator: {traceback.format_exc()}")
                    self.logging.error(f"error add indicator: {e}")
                    
            df = df.merge(
                indicator_df[["tic", "Date", indicator]], on=["tic", "Date"], how="left"
            )
        df = df.sort_values(by=["Date", "tic"])
        return df
    
    def start(self, save_file = False):
        data = self.download_data(config.TICKET_LIST)
        data_preparing = self.pre_clean_data(data)
        data_preparing = self.add_technical_indicator(data_preparing, config.INDICATOR_LIST)
        
        data_preparing.drop(["Date"],axis=1,inplace=True)
        
        rsi_count = pd.isna(data_preparing['rsi_14']).sum()
        strsi_count = pd.isna(data_preparing['stochrsi_14']).sum()
        vwma_count = pd.isna(data_preparing['vwma_14']).sum()
        tema_count = pd.isna(data_preparing['tema_200']).sum()
        ichimoku_count = pd.isna(data_preparing['ichimoku']).sum()
        print('rsi count:',rsi_count)
        print('strsi_count:',strsi_count)
        print('vwma_count:',vwma_count)
        print('tema_count:',tema_count)
        print('ichimoku_count:',ichimoku_count)
        
        # rsi pred
        print("fill missing value in rsi")
        rsi_model_path = os.path.join(os.getcwd(),'saved_model','rsi_14_model.pth')
        data_preparing['rsi_pred'] = data_preparing.apply(lambda x :predict_nanvalue_lstm(x[['rsi_14','Close']], 'rsi_14', rsi_model_path),axis=1)
        
        # # stochrsi pred
        print("fill missing value in stochrsi")
        stochrsi_model_path = os.path.join(os.getcwd(),'saved_model','stochrsi_14_model.pth')
        data_preparing['stochrsi_pred'] = data_preparing.apply(lambda x :predict_nanvalue_lstm(x[['stochrsi_14','Close']], "stochrsi_14", stochrsi_model_path),axis=1)
        
        # # tema pred
        print("fill missing value in tema")
        tema_model_path = os.path.join(os.getcwd(),'saved_model','tema_200_model.pth')
        data_preparing['tema_pred'] = data_preparing.apply(lambda x :predict_nanvalue_lstm(x[['tema_200','Close']], "tema_200", tema_model_path),axis=1)
        
        # vwma 
        print("fill missing value in vwma")
        vwma_model_path = os.path.join(os.getcwd(),'saved_model','vwma_14_model.pth')
        data_preparing['vwma_pred'] = data_preparing.apply(lambda x :predict_nanvalue_lstm_vwma(x[['vwma_14',"Close","Volume"]], "vwma_14", vwma_model_path),axis=1)
        
        # ichimoku
        print("fill missing value in ichimoku")
        ichimoku_model_path = os.path.join(os.getcwd(),'saved_model','ichimoku_model.pth')
        data_preparing['ichimoku_pred'] = data_preparing.apply(lambda x :predict_nanvalue_lstm_ichimoku(x[['ichimoku',"High","Low","Close"]], "ichimoku", ichimoku_model_path),axis=1)
        
        print('data_preparing:')
        print(data_preparing.columns)
        # drop column 
        data_preparing.drop(['rsi_14','stochrsi_14', 'vwma_14', 'tema_200', 'ichimoku'],axis = 1, inplace=True)
        
        data_preparing = return_candle_pattern(data_preparing)
        
        # convert tic to integer
        list_tic_unique = list(data_preparing['tic'].unique())
        map_tic = {}
        for i,key in enumerate(list_tic_unique):
            map_tic[key] = i
            
        data_preparing['tic'] = data_preparing['tic'].replace(map_tic)
        
        
        if save_file:
            print(f"path data: {os.path.join(os.getcwd(),'dataset')}")
            os.makedirs(os.path.join(os.getcwd(),'dataset'),exist_ok=True)
            data_preparing.to_csv(os.path.join(os.getcwd(),'dataset','train_test.csv'))
        
        return data_preparing
    
if __name__ == "__main__":
    data_pipeline=prepare_data()
    data = data_pipeline.start(save_file=True)