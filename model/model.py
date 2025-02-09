from gymnasium import spaces
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch

class trading_env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                 df_train,
                 window_size = 31
                 ):
        super(trading_env, self).__init__()
        self.action_space = spaces.Discrete(2,seed=42, start=0)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(window_size,len(df_train.columns)), dtype=np.float16)
        self.buy = 0
        self.data = df_train
        self.window_size = window_size
        self.fee = 0.07
        self.noise = 0.0001
        self.quest = False
        self.max_step = len(df_train.index)-(window_size)
        # self._random_state = np.random.RandomState(seed=42)
        
    def reset(self, seed=None, options=None):
        self.current_step = np.random.randint(0, self.max_step)
        self.quest = False
        self.buy = 0
        return self._next_observation(), {}
    
    def _next_observation(self):
        # 329430 is error
        # 329312 error when plus
        if self.current_step >= self.max_step:
            self.quest = True
        else:
            self.quest = False
            
        state = self.data.iloc[self.current_step:self.current_step + self.window_size]
        self.current_step+=1
        return state.values
    
    def step(self, actions):
        if self.quest:
            self.reset()
            
        reward = self._take_action(actions)
        obs = self._next_observation()
            
        return obs, reward, self.quest, False, {}
    
    def _take_action(self, action):
        reward = 0
        if action == 0:
            if self.buy == 0:
                self.buy = float(self.data.loc[self.current_step, "Close"])+(float(self.data.loc[self.current_step, "Close"])*self.fee)
                reward += self.noise
                print('buy:',float(self.data.loc[self.current_step, "Close"])+(float(self.data.loc[self.current_step, "Close"])*self.fee))
            else:
                # self.reward -= (float(self.data.loc[self.current_step, "Close"])+self.fee)*self.noise
                reward -= self.noise
                print("not sell")
                
        elif action == 1:
            if float(self.buy) > 0:
                reward += self.noise
                sell = self.data.loc[self.current_step, "Close"]
                rest = float(sell) - float(self.buy)
                print('sell:',self.buy, sell)
                print('rest:',rest)
                if rest >0 and self.buy == 0:
                    print("rewad +:","{:.20f}".format(rest))
                    reward += rest*self.noise
                elif rest < 0:
                    print("rewad -:",rest*self.noise)
                    reward -= rest*self.noise
                    
                self.buy = 0
            else:
                reward -= self.noise
        else:
            reward -= self.noise
            
        return reward
        
        
    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        
        
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
    

# Define the LSTM model
class LSTMModel_HYPER(nn.Module):
    def __init__(self, output_size, 
                 num_stocks, 
                 num_group, 
                 num_day, 
                 num_month, 
                 embedding_dim_stock, 
                 embedding_dim_group, 
                 embedding_dim_day, 
                 embedding_dim_month,
                 feature_dim,
                 hidden_size_norm,
                 first_layer_hidden_size,
                 first_layer_size,
                 second_layer_hidden_size,
                 second_layer_size,
                 third_layer_hidden_size,
                 third_layer_size,
                 dropout_value):
        super(LSTMModel, self).__init__()

        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim_stock)
        self.group_embedding = nn.Embedding(num_group, embedding_dim_group)
        self.day_embedding = nn.Embedding(num_day, embedding_dim_day)
        self.month_embedding = nn.Embedding(num_month, embedding_dim_month)

        input_dim = embedding_dim_stock + embedding_dim_group + embedding_dim_day + embedding_dim_month + feature_dim

        self.lstm1 = nn.LSTM(input_dim, first_layer_hidden_size, first_layer_size, batch_first=True)
        self.lstm2 = nn.LSTM(first_layer_hidden_size, second_layer_hidden_size, second_layer_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size_norm)
        self.lstm3 = nn.LSTM(second_layer_hidden_size, third_layer_hidden_size, third_layer_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_value)
        self.fc = nn.Linear(third_layer_hidden_size, output_size)

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        day_emb = self.day_embedding(day_name)
        month_emb = self.month_embedding(month_name)

        combind_input = torch.cat([stock_emb, group_emb, day_emb, month_emb, feature], dim=1)

        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out22, _ = self.layer_norm(lstm_out21)
        lstm_out3, _ = self.lstm3(lstm_out22)
        out1 = self.dropout(lstm_out3[:, -1, :])
        fc_out = self.fc(out1)
        return fc_out