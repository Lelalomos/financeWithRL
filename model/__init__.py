from gymnasium.envs.registration import register
from model.model import trading_env, LSTMModel, LSTMModel_HYPER

# regis rl model
register(
    id='trading_env-v0',
    entry_point='model:trading_env'
)