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
    def __init__(self, 
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config):
        super(LSTMModel, self).__init__()

        config = config.LSTM_PARAMS
        self.stock_embedding = nn.Embedding(num_stocks, config["embedding_dim_stock"])
        self.group_embedding = nn.Embedding(num_group, config['embedding_dim_group'])
        self.day_embedding = nn.Embedding(num_day, config['embedding_dim_day'])
        self.month_embedding = nn.Embedding(num_month, config['embedding_dim_month'])

        input_dim = config['embedding_dim_stock'] + config['embedding_dim_group'] + config['embedding_dim_day'] + config['embedding_dim_month'] + feature_dim
        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        self.lstm1 = nn.LSTM(input_dim, config['first_layer_hidden_size'], config['first_layer_size'], batch_first=True)
        self.lstm2 = nn.LSTM(config["first_layer_hidden_size"], config['second_layer_hidden_size'], config['second_layer_size'], batch_first=True)
        self.lstm3 = nn.LSTM(config['second_layer_hidden_size'], config['third_layer_hidden_size'], config['third_layer_size'], batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(config["third_layer_hidden_size"], config['output_size'])

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)

        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        
        # normalize
        batch_size, seq_len, input_size = combind_input.shape
        combind_input = combind_input.view(-1, input_size)
        combind_input = self.batch_norm_input(combind_input)
        combind_input = combind_input.view(batch_size, seq_len, input_size)

        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)
        out1 = self.dropout(lstm_out3)
        fc_out = self.fc(out1)
        return torch.tanh(fc_out)
    

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
                 first_layer_hidden_size,
                 first_layer_size,
                 second_layer_hidden_size,
                 second_layer_size,
                 third_layer_hidden_size,
                 third_layer_size,
                 dropout_value,
                 hidden_bilstm,
                 num_bilstm):
        super().__init__()

        # print("num_stocks:",num_stocks)
        # print("num_group:",num_group)
        # print("num_day:",num_day,embedding_dim_day)
        # print("num_month:",num_month)

        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim_stock)
        self.group_embedding = nn.Embedding(num_group, embedding_dim_group)
        self.day_embedding = nn.Embedding(num_day, embedding_dim_day)
        self.month_embedding = nn.Embedding(num_month, embedding_dim_month)

        
        input_dim = embedding_dim_stock + embedding_dim_group + embedding_dim_day+ embedding_dim_month+ feature_dim
        self.bilstm = nn.LSTM(input_dim, hidden_bilstm, num_bilstm, batch_first=True, bidirectional=True)
        self.batch_norm_input = nn.BatchNorm1d(hidden_bilstm*2)

        self.lstm1 = nn.LSTM(hidden_bilstm*2, first_layer_hidden_size, first_layer_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(first_layer_hidden_size*2, second_layer_hidden_size, second_layer_size, batch_first=True, bidirectional=True)
        print("input_dim:",input_dim)
        self.lstm3 = nn.LSTM(second_layer_hidden_size*2, third_layer_hidden_size, third_layer_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_value)
        

        self.fc = nn.Linear(third_layer_hidden_size*2, output_size)
        # print('finish init')

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)
        
        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        # print("Embedding output shape:", combind_input.shape)

        # print("out bi:",out)
        combind_input, _ = self.bilstm(combind_input)
        # print("bilstm shape:", combind_input.shape)
        combind_input = combind_input[:, -1, :]
        # print("bilstm re-shape:", combind_input.shape)

        # combind_input = combind_input.permute(0, 2, 1)
        combind_input = self.batch_norm_input(combind_input)
        # print("batch_norm_input:", combind_input.shape)
        combind_input = combind_input.unsqueeze(1)
        # print("add dimension batch_norm_input:", combind_input.shape)
        # combind_input = combind_input.permute(0, 2, 1)

        # normalize
        # batch_size, seq_len, input_size = combind_input.shape
        # combind_input = combind_input.view(-1, input_size)
        # combind_input = self.batch_norm_input(combind_input)
        # combind_input = combind_input.view(batch_size, seq_len, input_size)
        out, _ = self.lstm1(combind_input)
        # print("lstm1:", out.shape)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)

        out1 = self.dropout(lstm_out3)

        # out1, _ = self.bilstm(out1)

        fc_out = self.fc(out1)
        # print("fc_out:",fc_out)
        return torch.tanh(fc_out)
    
class AttentionLayer(nn.Module):
    def __init__(self, input_size, attention_size):
        super(AttentionLayer, self).__init__()
        self.query_linear = nn.Linear(input_size, attention_size)
        self.tanh = nn.Tanh()
        self.key_linear = nn.Linear(input_size, attention_size)
        self.value_linear = nn.Linear(input_size, attention_size)

    def forward(self, x):
        Q = self.query_linear(x)  # Query
        K = self.key_linear(x)    # Key
        V = self.value_linear(x)  # Value

        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # Q * K^T
        attention_scores = attention_scores / (K.size(-1) ** 0.5)  # Scaling
        attention_scores = self.tanh(attention_scores)
        attention_weights = nn.softmax(attention_scores, dim=-1)  # Softmax
        
        # Weighted sum of values to produce context vector
        context_vector = torch.bmm(attention_weights, V)
        
        return context_vector, attention_weights