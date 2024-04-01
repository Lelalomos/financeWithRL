from functions import download_data, normalize_robustscaler, add_technical_indicator, prepare_data_forlstm, pre_data, cal_zscore, detect_outliers_iqr, train_lstm
import config
from utility import return_logs
import os


def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))
    
    logging.info("download data")
    data = download_data(config.TICKET_LIST, logging=logging)
    data = pre_data(data)
    data = data.dropna(how = "any", ignore_index = True)
    
    logging.info("add indicator")
    print('indicator:', config.INDICATOR_LIST)
    df_with_indicator = add_technical_indicator(data, config.INDICATOR_LIST)
    print(df_with_indicator.columns)
    # prepare data for lstm
    logging.info("prepare data for lstm")
    data, label = prepare_data_forlstm(df_with_indicator)
    
    zscore_data = cal_zscore(data)
    print('zscore_data:',zscore_data)
    zscore_label = cal_zscore(label)
    print('zscore_label:',zscore_label)
    iqr_data =  detect_outliers_iqr(data)
    print('iqr_data:',iqr_data)
    iqr_label =  detect_outliers_iqr(label)
    print('iqr_label:',iqr_label)
    
    norm_data = normalize_robustscaler(data)
    print("norm_data")
    print(norm_data)
    norm_label = normalize_robustscaler(label)
    print("norm_label")
    print(norm_label)
    
    train_lstm(data)
    
    
if __name__ == "__main__":
    main()