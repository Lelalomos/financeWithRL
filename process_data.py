from utils import return_logs, prepare_data, normalization_data
from functions import return_candle_pattern, groupping_stock, cal_rsi,cal_storsi, cal_ichimoku, cal_ema, convert_string2int, split_dataset
import os
import pandas as pd
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
    
    logging.info("pull data from yahoo")
    # pull data
    if os.path.isfile(os.path.join(os.getcwd(),'data','dataset.parquet')):
        data = pd.read_parquet(os.path.join(os.getcwd(),'data','dataset.parquet'))
    else:
        # download data
        data = pdata_func.download_data(config.TICKET_LIST, interval="1d")
        data.to_parquet(os.path.join(os.getcwd(), 'data','dataset.parquet'))

    logging.info("prepare data")
    # clean data
    # data = pdata_func.pre_clean_data(data)
    data = data.rename(columns=config.MAP_COLUMNS_NAME)
    print("columns",data.columns)
    data = pdata_func.add_indicator(data, config.INDICATOR_LIST)

    data = return_candle_pattern(data)
    data = data.fillna(0)
    # separate date
    data['Date'] = pd.to_datetime(data['Date'])
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data = data.sort_values(by=["Date", "tic"])
    data.drop(['Date'], inplace=True, axis=1)

    # make label
    data["pre_7"] = data["close"].pct_change(periods=7).shift(-7) * 100  # เปลี่ยนเป็น %
    data["pre_7"] = np.tanh(data["pre_7"] / 100) * 100
    data["pre_7"] = data["pre_7"].fillna(method="bfill", limit=7)

    # grouping sector in stock
    group_sector = groupping_stock(data, config)
    group_sector = convert_string2int(group_sector)

    # interpreter data
    group_sector['stochrsi_14'] = group_sector['stochrsi_14']/100
    group_sector['stochrsi_14_decision'] = group_sector['stochrsi_14'].apply(cal_storsi)

    group_sector['rsi_14'] = group_sector['rsi_14']/100
    group_sector['rsi_14_decision'] = group_sector['rsi_14'].apply(cal_rsi)

    group_sector['ichimoku_decision'] = group_sector['ichimoku'].apply(cal_ichimoku)

    group_sector['ema_50100'] = group_sector.apply(cal_ema,args=(50,100),axis=1)
    group_sector['ema_50200'] = group_sector.apply(cal_ema,args=(50,200),axis=1)
    group_sector['ema_50200'] = group_sector.apply(cal_ema,args=(100,200),axis=1)

    # column Outliers
    outliers_column = ['close','high','low','open','volume','vwma_20','ema_200','ema_50','ema_100','macd','ichimoku']

    # df_outlier = group_sector[outliers_column]
    group_sector = norm_func.norm_each_row_bylogtransform(group_sector, outliers_column)
    group_sector['ichimoku'] = group_sector['ichimoku'].fillna(-1)
    group_sector['macd'] = group_sector['macd'].fillna(-1)

    # ต้องเพิ่ม label ว่าต้องการแบบไหน
    # split train, validate, test
    train_set, validate_set, test_set = split_dataset(group_sector)

    validate_set = validate_set.drop(["year"],axis=1)
    test_set = test_set.drop(["year"],axis=1)
    train_set = train_set.drop(["year"],axis=1)

    validate_set.to_parquet(os.path.join(os.getcwd(),"data","validate_dataset.parquet"))
    test_set.to_parquet(os.path.join(os.getcwd(),"data","test_dataset.parquet"))
    train_set.to_parquet(os.path.join(os.getcwd(),"data","train_dataset.parquet"))


if __name__ == "__main__":
    main()