from utils import return_logs, prepare_data
from functions import return_candle_pattern
import os
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import config

pdata_func = prepare_data()

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
    data = pdata_func.pre_clean_data(data)
    data = pdata_func.add_indicator(data, config.INDICATOR_LIST)
    data = return_candle_pattern(data)
    data = data.fillna(0)
    data.to_csv("data_final.csv")

    # data.drop(["Date"],axis=1,inplace=True)

    print("data:")
    print(data.columns)


    del data


if __name__ == "__main__":
    main()