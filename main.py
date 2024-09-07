from utils import return_logs, prepare_data
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
    
    # pull data
    if os.path.isfile(os.path.join(os.getcwd(),'data','dataset.parquet')):
        data = pd.read_parquet(os.path.join(os.getcwd(),'data','dataset.parquet'))
    else:
        # data = pdata_func.download_data(config.TICKET_LIST)
        data = pdata_func.collect_data(True,True)
        data.to_parquet("data/dataset.parquet")
          
    # handle data
    data = pdata_func.filling_missing_value(data, "default_value")
    print(data.head())
    


    
    # train reinforcement learning
    # logging.info("train reinforcement leaning")
    # data = pd.read_csv(os.path.join(os.getcwd(),'dataset','train_test.csv'),index_col=0)
    
    
    # model_rl = train_rl(dataset=data)
    # model_rl.start()
    
    # 
    
    
if __name__ == "__main__":
    main()