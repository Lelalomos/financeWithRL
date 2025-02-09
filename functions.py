from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from model import  LSTMModel
import pandas as pd
import talib
import config
from pipe import dedup

def search_null_value(df):
    columns_with_null = df.columns[df.isnull().any()]
    print('columns_with_null:',columns_with_null)
    for column in columns_with_null:
        print(f"Null values in column '{column}':")
        print(df[df[column].isnull()])
    
def prepare_data_forlstm(data):
    data = data.dropna(how = "any", ignore_index = True)
    data = data.sort_values(["Date"]).reset_index(drop=True)
    return np.array(data['close']), np.array(data['rsi_14'])

def detect_outliers_iqr(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    return len(np.where((data < lower_bound) | (data > upper_bound))[0])


def predict_nanvalue_lstm(data, column_name, model_path, device, default_value = 0):
    if pd.isna(data[column_name]) or data[column_name] in [None,np.nan,""]:
        close = float(data['close'])
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        x = np.array([[[close]]])
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = x.to(torch.float32)
        
        with torch.no_grad():
            prediction = model(x).item()
            
        if np.isnan(prediction):
            prediction = default_value
        return prediction
    else:
        return data[column_name]
    
def predict_lstm_single(close_value, model_path, device, default_value = 0):
    # print("device",device)
    close = float(close_value)
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    x = np.array([[[close]]])
    x = torch.tensor(x, dtype=torch.float32).to(device)
    x = x.to(torch.float32)
    
    with torch.no_grad():
        prediction = model(x).item()
        
    if np.isnan(prediction):
        prediction = default_value
        
    return prediction

def predict_lstm_multiple(data:pd.Series, model_path:str, device, default_value = 0):
    model = LSTMModel(input_size=len(data.keys()))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    x = np.array([[data]])
    x = torch.tensor(x,dtype=torch.float32).to(device)
    x = x.to(torch.float32)
    with torch.no_grad():
        prediction = model(x).item()
        
    if np.isnan(prediction):
        prediction = default_value
        
    return prediction
    
# must fix parameter model and parameter data
def predict_nanvalue_lstm_vwma(data, column_name, model_path, device, default_value = 0):
    if pd.isna(data[column_name]) or data[column_name] in [None,np.nan,""]:
        close = float(data['close'])
        volumn = float(data['Volume'])
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        x = np.array([[[close,volumn]]])
        x = torch.tensor(x,dtype=torch.float32).to(device)
        x = x.to(torch.float32)
        with torch.no_grad():
            prediction = model(x).item()
            
        if np.isnan(prediction):
            prediction = default_value
        return prediction
    else:
        return data[column_name]
    
def predict_nanvalue_lstm_ichimoku(data, column_name, model_path, device, default_value = 0):
    if pd.isna(data[column_name]) or data[column_name] in [None,np.nan,""]:
        close = float(data['close'])
        high = float(data['high'])
        low = float(data['low'])
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        x = np.array([[[close,high,low]]])
        x = torch.tensor(x,dtype=torch.float32).to(device)
        x = x.to(torch.float32)
        with torch.no_grad():
            prediction = model(x).item()
        if np.isnan(prediction):
            prediction = default_value
        return prediction
    else:
        return data[column_name]
    
def refill_missingvalue(data, column_name, default_value = 0):
    if pd.isna(data[column_name]) or data[column_name] in [None,np.nan,""]:
        return default_value
    else:
        return data[column_name]
    
def return_candle_pattern(data):
    # 'open', 'high', 'low', 'close', 'Volume'
    data['CDL2CROWS'] = talib.CDL2CROWS(data['open'], data['high'], data['low'], data['close'])
    data['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(data['open'], data['high'], data['low'], data['close'])
    data['CDL3INSIDE'] = talib.CDL3INSIDE(data['open'], data['high'], data['low'], data['close'])
    data['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(data['open'], data['high'], data['low'], data['close'])
    data['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(data['open'], data['high'], data['low'], data['close'])
    data['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(data['open'], data['high'], data['low'], data['close'])
    data['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(data['open'], data['high'], data['low'], data['close'])
    data['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(data['open'], data['high'], data['low'], data['close'])
    data['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(data['open'], data['high'], data['low'], data['close'])
    data['CDLBELTHOLD'] = talib.CDLBELTHOLD(data['open'], data['high'], data['low'], data['close'])
    data['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(data['open'], data['high'], data['low'], data['close'])
    data['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(data['open'], data['high'], data['low'], data['close'])
    data['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(data['open'], data['high'], data['low'], data['close'])
    data['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(data['open'], data['high'], data['low'], data['close'])
    data['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(data['open'], data['high'], data['low'], data['close'])
    data['CDLDOJI'] = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close'])
    data['CDLDOJISTAR'] = talib.CDLDOJISTAR(data['open'], data['high'], data['low'], data['close'])
    data['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(data['open'], data['high'], data['low'], data['close'])
    data['CDLENGULFING'] = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])
    data['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(data['open'], data['high'], data['low'], data['close'])
    data['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(data['open'], data['high'], data['low'], data['close'])
    data['CDLHAMMER'] = talib.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])
    data['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(data['open'], data['high'], data['low'], data['close'])
    data['CDLHARAMI'] = talib.CDLHARAMI(data['open'], data['high'], data['low'], data['close'])
    data['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(data['open'], data['high'], data['low'], data['close'])
    data['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(data['open'], data['high'], data['low'], data['close'])
    data['CDLHIKKAKE'] = talib.CDLHIKKAKE(data['open'], data['high'], data['low'], data['close'])
    data['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(data['open'], data['high'], data['low'], data['close'])
    data['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(data['open'], data['high'], data['low'], data['close'])
    data['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(data['open'], data['high'], data['low'], data['close'])
    data['CDLINNECK'] = talib.CDLINNECK(data['open'], data['high'], data['low'], data['close'])
    data['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(data['open'], data['high'], data['low'], data['close'])
    data['CDLKICKING'] = talib.CDLKICKING(data['open'], data['high'], data['low'], data['close'])
    data['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(data['open'], data['high'], data['low'], data['close'])
    data['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(data['open'], data['high'], data['low'], data['close'])
    data['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(data['open'], data['high'], data['low'], data['close'])
    data['CDLLONGLINE'] = talib.CDLLONGLINE(data['open'], data['high'], data['low'], data['close'])
    data['CDLMARUBOZU'] = talib.CDLMARUBOZU(data['open'], data['high'], data['low'], data['close'])
    data['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(data['open'], data['high'], data['low'], data['close'])
    data['CDLMATHOLD'] = talib.CDLMATHOLD(data['open'], data['high'], data['low'], data['close'])
    data['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(data['open'], data['high'], data['low'], data['close'])
    data['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close'])
    data['CDLONNECK'] = talib.CDLONNECK(data['open'], data['high'], data['low'], data['close'])
    data['CDLPIERCING'] = talib.CDLPIERCING(data['open'], data['high'], data['low'], data['close'])
    data['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(data['open'], data['high'], data['low'], data['close'])
    data['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(data['open'], data['high'], data['low'], data['close'])
    data['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(data['open'], data['high'], data['low'], data['close'])
    data['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(data['open'], data['high'], data['low'], data['close'])
    data['CDLSHORTLINE'] = talib.CDLSHORTLINE(data['open'], data['high'], data['low'], data['close'])
    data['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(data['open'], data['high'], data['low'], data['close'])
    data['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(data['open'], data['high'], data['low'], data['close'])
    data['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(data['open'], data['high'], data['low'], data['close'])
    data['CDLTAKURI'] = talib.CDLTAKURI(data['open'], data['high'], data['low'], data['close'])
    data['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(data['open'], data['high'], data['low'], data['close'])
    data['CDLTHRUSTING'] = talib.CDLTHRUSTING(data['open'], data['high'], data['low'], data['close'])
    data['CDLTRISTAR'] = talib.CDLTRISTAR(data['open'], data['high'], data['low'], data['close'])
    data['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(data['open'], data['high'], data['low'], data['close'])
    data['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(data['open'], data['high'], data['low'], data['close'])
    data['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(data['open'], data['high'], data['low'], data['close'])
    
    return data

def groupping_stock(data, config):
    u_tic = list(data['tic'].unique())
    df_all = pd.DataFrame(dtype = str)
    for tic in u_tic:
        for group in config.SECTOR_GROUP:
            if tic in config.SECTOR_GROUP[group]:
                df_temp = data[data['tic'] == tic]
                df_temp['group'] = group
                df_all = pd.concat([df_all, df_temp])

    return df_all

def convert_string2int(data, list_column=['group','tic']):
    stock_encoder = LabelEncoder()
    for column in list_column:
        data[f"{column}_id"] = stock_encoder.fit_transform(data[column])
    data = data.drop(columns=list_column,axis =1)
    return data
        
# calculate indicator
def cal_rsi(value):
    value = float(value)
    if value >= config.RSI_UP:
        return 1
    elif value <= config.RSI_DOWN:
        return -1
    else:
        return 0
    
def cal_storsi(value):
    value = float(value)
    if value >= config.STORSI_UP:
        return 1
    elif value <= config.STORSI_DOWN:
        return -1
    else:
        return 0
    
def cal_ichimoku(value):
    if value > config.ICHIMOKU_UP:
        return 1
    elif value < config.ICHIMOKU_DOWN:
        return -1
    else:
        return 0
    
def cal_ema(value, min_tema, max_tema):
    tema_min = value[f'ema_{min_tema}']
    tema_max = value[f'ema_{max_tema}']
    tema_min = float(tema_min)
    tema_max = float(tema_max)
    if tema_min > tema_max:
        return 1
    elif tema_min < tema_max:
        return -1
    else:
        return 0
    

def split_dataset(df):
    list_year = df['year'].to_list()
    list_year = list(list_year|dedup)

    # split train_set, validate_set, test_set
    train_set = list_year[:-5]
    validate_set = list_year[-5:-3]
    test_set = list_year[-3:-1]

    df['year'] = df['year'].astype(int)
    df_train_set = df[(df['year']>=train_set[0]) & (df['year']<=train_set[-1])]
    df_validate_set = df[(df['year']>=validate_set[0]) & (df['year']<=validate_set[-1])]
    df_test_set = df[(df['year']>=test_set[0]) & (df['year']<=test_set[-1])]

    return df_train_set, df_validate_set, df_test_set