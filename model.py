from gymnasium import spaces
import gymnasium as gym
import numpy as np
import torch.nn as nn

class trading_env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                 df_train,
                 window_size = 31
                 ):
        super(trading_env, self).__init__()
        self.action_space = spaces.Discrete(3,seed=42, start=0)
        self.observation_space = spaces.Box(low=0, high=10, shape=(31,12), dtype=np.float16)
        self.buy = 0
        self.data = df_train
        self.reward = 0
        self.window_size = window_size
        self.fee = 15
        self.noise = 0.0001
        self.quest = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.quest = False
        self.buy = 0
        return self._next_observation(), {}
    
    def _next_observation(self):
        state = self.data.iloc[self.current_step:self.current_step + self.window_size]
        return state.values
    
    def step(self, actions):
        self._take_action(actions)
        obs = self._next_observation()
        if self.quest:
            print('reward:',self.reward)
            
        return obs, self.reward, self.quest, False, {}
    
    def _take_action(self, action):
        if action == 0:
            if self.buy == 0:
                self.buy = float(self.data.loc[self.current_step, "Close"])+self.fee
                print('buy:',float(self.data.loc[self.current_step, "Close"])+self.fee)
            else:
                self.reward -= (float(self.data.loc[self.current_step, "Close"])+self.fee)*self.noise
                
        elif action == 1:
            sell = self.data.loc[self.current_step, "Close"]
            rest = float(sell) - float(self.buy)
            print('sell step, buy:',self.buy)
            print('sell:',sell)
            if rest >0:
                print('rest:',rest)
                self.reward += rest*self.noise
                self.quest = True
            else:
                self.reward -= rest*self.noise
                
            self.buy = 0
        else:
            self.reward -= self.noise
            
    def render(self, mode = 'human', close=False):
        print(f'reward: {self.reward}')
        
        
# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 32, 3, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.lstm21 = nn.LSTM(32, 64, 3, batch_first=True)
        self.lstm22 = nn.LSTM(64, 64, 3, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(64, 32, 3, batch_first=True)
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        lstm_out21, _ = self.lstm21(out)
        lstm_out22, _ = self.lstm22(lstm_out21)
        out1 = self.dropout(lstm_out22)
        lstm_out3, _ = self.lstm3(out1)
        fc_out = self.fc(lstm_out3)
        return fc_out