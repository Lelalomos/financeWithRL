from utility import return_logs
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from model import trading_env
import pandas as pd

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'logs_images'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'saved_model'),exist_ok=True)
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))
    
    data = pd.read_csv(os.path.join(os.getcwd(),'dataset','train_test.csv'),index_col=0)
    
    env = DummyVecEnv([lambda: trading_env(df_train=data)])
    model = DQN('MlpPolicy', env, verbose=1, batch_size=100000)
    model.learn(total_timesteps=1000000)
    
    model.save(os.path.join("mlp_rl_mem.zip"))
    
    
    
    
    
    
    
        
if __name__ == "__main__":
    
    main()