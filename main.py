from functions import prepare_data_forlstm, detect_outliers_iqr
from train_lstm import train_lstm
import config
from sklearn.model_selection import train_test_split
from utility import return_logs
import os
import numpy as np
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
    train_data = train_lstm(data, threshold_loss = 0.001, batch_size = 1024, path_save_loss= os.path.join(os.getcwd(),'logs_images','loss.jpg'), path_save_model= os.path.join(os.getcwd(),'saved_model','best_model.pth'), epochs=100, splitdata_test_size=0.2)
    train_data.for_rsi(config.INDICATOR_LIST, 'rsi_14')

    
    
    
if __name__ == "__main__":
    main()