from functions import prepare_data_forlstm, detect_outliers_iqr, predict_nanvalue_lstm, return_candle_pattern
from train_lstm import train_lstm
import config
from sklearn.model_selection import train_test_split
from utility import return_logs
import os
import pandas as pd
from prepare_data import prepare_data

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'logs_images'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'saved_model'),exist_ok=True)
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))
    
    # init prepare class
    prepare = prepare_data()
    
    logging.info("download data")
    data = prepare.download_data(config.TICKET_LIST)
    data_preparing = prepare.pre_clean_data(data)
    data_preparing = prepare.add_technical_indicator(data_preparing, config.INDICATOR_LIST)
    print('data_preparing:')
    print(data_preparing)
    
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
    rsi_model_path = os.path.join(os.getcwd(),'saved_model','rsi_14_model.pth')
    data_preparing['rsi_pred'] = data_preparing.apply(lambda x :predict_nanvalue_lstm(x[['rsi_14','Close']], 'rsi_14', rsi_model_path),axis=1)
    
    # stochrsi pred
    stochrsi_model_path = os.path.join(os.getcwd(),'saved_model','stochrsi_14_model.pth')
    data_preparing['stochrsi_pred'] = data_preparing.apply(lambda x :predict_nanvalue_lstm(x[['stochrsi_14','Close']], "stochrsi_14", stochrsi_model_path),axis=1)
    print('data_preparing:')
    print(data_preparing.columns)
    # drop column 
    data_preparing.drop(['rsi_14','stochrsi_14'],axis = 1, inplace=True)
    
    data_preparing = return_candle_pattern(data_preparing)
    
    # convert tic to integer
    list_tic_unique = list(data_preparing['tic'].unique())
    map_tic = {list_tic_unique[i-1]:i for i in range(1,len(list_tic_unique))}
    data_preparing['tic'] = data_preparing['tic'].replace(map_tic)
    
    data_preparing.to_csv(os.path.join(os.getcwd(),'tests','train.csv'))
    
    
    
def train_lstm4pred(indicator_name):
    prepare = prepare_data()
    data = prepare.download_data(config.TICKET_LIST)
    train_data = train_lstm(data, threshold_loss = 0.001, batch_size = 1024, path_save_loss= os.path.join(os.getcwd(),'logs_images',f'loss_{indicator_name}.jpg'), path_save_model= os.path.join(os.getcwd(),'saved_model',f'{indicator_name}_model.pth'), epochs=100, splitdata_test_size=0.2)
    train_data.for_rsi(config.INDICATOR_LIST, indicator_name)
    
if __name__ == "__main__":
    # train_lstm4pred('rsi_14')
    # train_lstm4pred('stochrsi_14')
    
    main()