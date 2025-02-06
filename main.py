from utils import return_logs, prepare_data, normalization_data
from functions import return_candle_pattern, groupping_stock, cal_rsi,cal_storsi, cal_ichimoku, cal_ema
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import config

pdata_func = prepare_data()
norm_func = normalization_data()

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'logs_images'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'saved_model'),exist_ok=True)
    
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))

    end_date = datetime.now()  # วันที่ปัจจุบัน
    start_date = end_date - timedelta(days=config.YEAR_END*365)  # ย้อนหลัง 5 ปี
    
    logging.info("pull data from yahoo")
    # pull data
    if os.path.isfile(os.path.join(os.getcwd(),'data','dataset.parquet')):
        data = pd.read_parquet(os.path.join(os.getcwd(),'data','dataset.parquet'))
    else:
        # download data
        data = pdata_func.download_data(config.TICKET_LIST, interval="1d", start_date= start_date, end_date=end_date)
        data.to_parquet(os.path.join(os.getcwd(), 'data','dataset.parquet'))


    logging.info("prepare data")
    # clean data
    data = pdata_func.pre_clean_data(data)
    data = pdata_func.add_indicator(data, config.INDICATOR_LIST)
    data = return_candle_pattern(data)
    data = data.fillna(0)
    data = data.sort_values(by=["Date", "tic"])
    data.drop(['Date'], inplace=True, axis=1)
    # grouping sector in stock
    group_sector = groupping_stock(data, config)

    # interpreter data
    group_sector['stochrsi_14'] = group_sector['stochrsi_14']/100
    group_sector['stochrsi_14_decision'] = group_sector['stochrsi_14'].apply(cal_storsi)

    group_sector['rsi_14'] = group_sector['rsi_14']/100
    group_sector['rsi_14_decision'] = group_sector['rsi_14'].apply(cal_rsi)

    group_sector['ichimoku_decision'] = group_sector['ichimoku'].apply(cal_ichimoku)

    print(group_sector.columns)

    group_sector['ema_50100'] = group_sector.apply(cal_ema,args=(50,100),axis=1)
    group_sector['ema_50200'] = group_sector.apply(cal_ema,args=(50,200),axis=1)
    group_sector['ema_50200'] = group_sector.apply(cal_ema,args=(100,200),axis=1)

    # column Outliers
    outliers_column = ['close','high','low','open','volume','vwma_20','ema_200','ema_50','ema_100','macd','ichimoku']
    # df_outlier = group_sector[outliers_column]
    group_sector = norm_func.norm_each_row_bylogtransform(group_sector, outliers_column)
    group_sector.to_csv("test_dataset.csv")

    # take log transformation with outliers column
    # for out_column in outliers_column:
    #     df_outlier[f'log_{out_column}'] = np.where(df_outlier[out_column] > 0, np.log(df_outlier[out_column]), np.log(df_outlier[out_column] + 1))

    # standardization 





    

    
    # data.to_csv("data_final.csv")

    # data.drop(["Date"],axis=1,inplace=True)

    print("data:")
    print(data.columns)


    del data



if __name__ == "__main__":
    main()