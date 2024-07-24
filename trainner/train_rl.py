import pandas as pd
from utils.normalization import normalization_data
from stable_baselines3.common.vec_env import DummyVecEnv
from model.model import trading_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO
import os
from gymnasium.envs.registration import register
import gymnasium as gym


class train_rl:
    def __init__(self, dataset:pd.DataFrame = None, 
                 window_size:int = 120, 
                 reward_threshold:int = 1, 
                 verbose:int = 1,
                 batch_size:int = 100000,
                 n_epochs:int = 1000,
                 total_timesteps:int = 200000) -> None:
        self.data = dataset
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.total_timesteps = total_timesteps
        
        assert dataset is not None, "data is not found"
    
    def prepare_data(self):
        normalize_func = normalization_data()
        column_norm = ['Open', 'High', 'Low', 'Close', 'Volume', 'tema_pred','vwma_pred','ichimoku_pred']
        for norm_col in column_norm:
            self.data[norm_col] = normalize_func.normalize_minmax_1d_data(self.data[norm_col].to_numpy())
    
    def model(self):
        env = DummyVecEnv([lambda: trading_env(df_train=self.data, window_size=self.window_size)])
        eval_callback = EvalCallback(env, best_model_save_path='./saved_model/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False,
                             callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=self.reward_threshold, verbose=self.verbose))
        model = PPO('MlpPolicy', env, verbose=self.verbose, batch_size=self.batch_size, n_epochs= self.n_epochs)
        return model, eval_callback
        
    def start(self, path_model = os.path.join(os.getcwd(),'saved_model','ppo_mlp_stock.zip')):
        self.prepare_data()
        model, callback = self.model()
        model.learn(total_timesteps=self.total_timesteps, callback= callback)
        model.save(path_model)
        
        
if __name__ == "__main__":
    # data = pd.read_csv(os.path.join(os.getcwd(),'dataset','train_test.csv'),index_col=0)
    register(
            id='trading_env-v0',
            entry_point='model:trading_env'
    )
    
    # env = gym.make('trading_env-v0', df_train=data)
    
    # model_rl = train_rl(dataset=data)
    # model_rl.start()
    
    # env = gym.make('train_rl-v0', dataset=data)
