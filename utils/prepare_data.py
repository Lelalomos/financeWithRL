import pandas as pd
import numpy as np
from scipy.stats import zscore
import config
from utils.logger import return_logs
import os
import yfinance as yf
import traceback
from functions import cal_rsi, cal_storsi, cal_ema
import sys
sys.path.append("/app")
from stockstats.stockstats import StockDataFrame as Sdf

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

        # rename column to lower char
        df = df.rename(columns=config.MAP_COLUMNS_NAME)
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
    
    def add_indicator(self, dataframe, list_indicator):
        self.logging.info(f"add technical indicator into dataframe process, tech_indicator_list:{list_indicator}")
        print(f"add technical indicator into dataframe process, tech_indicator_list:{list_indicator}")
        indicator_func = Sdf(dataframe)
        list_tic = list(dataframe['tic'].unique())
        df_all = pd.DataFrame(dtype=str)
        for tic in list_tic:
            split_tic_data = dataframe[dataframe['tic'] == tic]
            for indicator in config.INDICATOR_LIST:
                split_tic_data[indicator] = indicator_func[indicator]

            split_tic_data[indicator] = split_tic_data[indicator]

            df_all = pd.concat([df_all, split_tic_data],axis=0, ignore_index=True)

        return df_all
    
    def sep_datetime(self, dataframe):
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe['month'] = dataframe['Date'].dt.month
        dataframe['day'] = dataframe['Date'].dt.day

        return dataframe
    
    def interpret_indicator(self, dataframe):
        # rsi
        dataframe['rsi_14'] = dataframe['rsi_14']/100
        dataframe['rsi_interpret'] = dataframe['rsi_14'].apply(cal_rsi)

        dataframe['stochrsi_14'] = dataframe['stochrsi_14']/100
        dataframe['stochrsi_14_interpret'] = dataframe['stochrsi_14'].apply(cal_storsi)

        dataframe['ema_50100'] = dataframe.apply(cal_ema,args=(50,100),axis=1)
        dataframe['ema_100200'] = dataframe.apply(cal_ema,args=(100,200),axis=1)
        dataframe['ema_50200'] = dataframe.apply(cal_ema,args=(50,200),axis=1)

        # normalize volumns
        df_all = pd.DataFrame(dtype=str)
        list_tic = list(dataframe['tic'].unique())
        for tic in list_tic:
            split_tic_data = dataframe[dataframe['tic'] == tic]
            split_tic_data = (split_tic_data['volume'] - split_tic_data['volume'].min()) / (split_tic_data['volume'].max() - split_tic_data['volume'].min())
            df_all = pd.concat([df_all, split_tic_data],axis=0, ignore_index=True)

        return df_all

    
    
if __name__ == "__main__":
    data_pipeline = prepare_data()
    path_data = data_pipeline.download_data(config.TICKET_LIST)
    
    # print(data)
    # data = data_pipeline.download_data(ticker_list = config.TICKET_LIST)
    # data = data_pipeline.start(save_file=False,data=data)