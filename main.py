from utils import return_logs
import os
import pandas as pd
# from train_rl import follow_tendline


def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'logs_images'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'saved_model'),exist_ok=True)
    
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))
    
    # train reinforcement learning
    logging.info("train reinforcement leaning")
    # data = pd.read_csv(os.path.join(os.getcwd(),'dataset','train_test.csv'),index_col=0)
    
    
    # model_rl = train_rl(dataset=data)
    # model_rl.start()
    
    # 
    
    
if __name__ == "__main__":
    main()