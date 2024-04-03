from functions import prepare_data_forlstm, detect_outliers_iqr, train_lstm
import config
from sklearn.model_selection import train_test_split
from utility import return_logs
import os
import numpy as np
from prepare_data import prepare_data

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs1'),exist_ok=True)
    logging = return_logs(os.path.join(os.getcwd(),'logs1','process.log'))
    
    # init prepare class
    prepare = prepare_data()
    
    epoch = 200
    thes_train = 0.02
    
    logging.info("download data")
    data = prepare.download_data(config.TICKET_LIST)
    
    
    
if __name__ == "__main__":
    main()