from functions import download_data, normalize_robustscaler, add_technical_indicator, prepare_data_forlstm, pre_data, cal_zscore, detect_outliers_iqr, train_lstm, normalize_minmax_data
import config
from sklearn.model_selection import train_test_split
from utility import return_logs
import os
import numpy as np

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs1'),exist_ok=True)
    logging = return_logs(os.path.join(os.getcwd(),'logs1','process.log'))
    
    epoch = 200
    thes_train = 0.02
    
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
    
    print('before filter',len(label))
    print('before filter',len(data))
    data, outlier = cal_zscore(data)
    print('zscore_data:',data)
    
    label_filtered = np.delete(label, outlier)
    print('lable:',len(label_filtered))
    print('zscore_data',len(data))
    
    iqr_data =  detect_outliers_iqr(data)
    print('iqr_data:',iqr_data)
    iqr_label =  detect_outliers_iqr(label_filtered)
    print('iqr_label:',iqr_label)
    
    norm_data = normalize_minmax_data(data)
    print("norm_data")
    print(norm_data)
    norm_label = normalize_minmax_data(label_filtered)
    print("norm_label")
    print(norm_label)
    
    x_train, x_test, y_train, y_test = train_test_split(norm_data, norm_label,random_state = 42, test_size=3)

    train_lstm(x_train, y_train, x_test, y_test, epoch, thes_train)
    
    
if __name__ == "__main__":
    main()