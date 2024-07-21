from utility import return_logs
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from model import trading_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import pandas as pd

from normalization import normalization_data

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'logs_images'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'saved_model'),exist_ok=True)
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))
    normalize_func = normalization_data()
    
    data = pd.read_csv(os.path.join(os.getcwd(),'dataset','train_test.csv'),index_col=0)
    
    column_norm = ['Open', 'High', 'Low', 'Close', 'Volume', 'tema_pred','vwma_pred','ichimoku_pred']
    for norm_col in column_norm:
        data[norm_col] = normalize_func.normalize_minmax_1d_data(data[norm_col].to_numpy())
    
    # for c in data.columns:
    #     print(f"column {c}:",np.isnan(data[c].values).any())
    
    env = DummyVecEnv([lambda: trading_env(df_train=data, window_size=120)])
    eval_callback = EvalCallback(env, best_model_save_path='./saved_model/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False,
                             callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=20, verbose=1))
    
    model = PPO('MlpPolicy', env, verbose=1, batch_size=100000, n_epochs= 1000)
    model.learn(total_timesteps=1000000, callback= eval_callback)
    
    model.save(os.path.join("mlp_rl_mem.zip"))
    
    
if __name__ == "__main__":
    main()