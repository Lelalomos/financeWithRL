from stockstats import StockDataFrame as Sdf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from utility import return_logs
import os
import yfinance as yf
import traceback

class prepare_data:
    def __init__(self):
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
    
    def add_technical_indicator(self, dataframe, tech_indicator_list):
        self.logging.info(f"add technical indicator into dataframe process, tech_indicator_list:{tech_indicator_list}")
        df = dataframe.copy()
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