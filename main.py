from utility import return_logs
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from model import trading_env
import pandas as pd

from normalization import normalization_data

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'logs_images'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'saved_model'),exist_ok=True)
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))
    normalize_func = normalization_data()
    
    data = pd.read_csv(os.path.join(os.getcwd(),'dataset','train_test.csv'),index_col=0)
    
    data['Volume'] = normalize_func.normalize_minmax_1d_data(data['Volume'].to_numpy())
    
    env = DummyVecEnv([lambda: trading_env(df_train=data, window_size=120)])
    model = PPO('MlpPolicy', env, verbose=1, batch_size=100000, n_epochs= 1000)
    model.learn(total_timesteps=1000000)
    
    model.save(os.path.join("mlp_rl_mem.zip"))
    
    
if __name__ == "__main__":
    main()