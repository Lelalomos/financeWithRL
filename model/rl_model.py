from gymnasium import spaces
import gymnasium as gym
import numpy as np

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