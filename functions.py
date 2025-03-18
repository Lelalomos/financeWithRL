from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from model import  LSTMModel
import pandas as pd
import talib
import config
from pipe import dedup
from datetime import datetime
import math

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
    dict_data = {}
    for column in list_column:
        stock_encoder = LabelEncoder()
        stock_encoder.fit(data[column])
        data[f"{column}_id"] = stock_encoder.transform(data[column])
        decoded_values = stock_encoder.inverse_transform(data[f"{column}_id"])
        dict_data[f'{column}_label'] = data[f"{column}_id"].to_list()
        dict_data[f'{column}_str'] = decoded_values
    
    df = pd.DataFrame(dict_data)
    df.to_excel("interpret.xlsx")

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
    

def split_dataset(df, test_ratio=0.06, validate_ratio = 0.06):
    today = datetime.today()
    df_test = pd.DataFrame(dtype=str)
    df_train = pd.DataFrame(dtype=str)
    df_validate = pd.DataFrame(dtype=str)

    # df['Date'] = pd.to_datetime(df['Date'])
    for tic in list(df['tic_id'].unique()):
        print(tic)
        temp = df[df['tic_id'] == tic]
        temp['year'] = temp['year'].astype('int')
        list_year = list(temp['year'].unique())
        list_year = [y for y in list_year if str(y) != str(today.year)]
        len_year = len(list_year)
        print(f"list_date: {list_year}")
        if len_year > 2:
            num_test = math.ceil(len_year*test_ratio)
            num_validate = math.ceil(len_year*validate_ratio)
            print("int(num_test)*-1:",int(num_test), int(num_validate))
            list_test = list_year[int(num_test)*-1:]
            list_validate = list_year[(int(num_test)+int(num_validate))*-1:int(num_test)*-1]
            list_train = list_year[:(int(num_test)+int(num_validate))*-1]

            print(f"list_test:{list_test}")
            print(f"list_validate:{list_validate}")
            print(f"list_train: {list_train}")

            min_test = min(list_test)
            max_test = max(list_test)
            filter_test_temp = temp[(temp['year'] >= min_test) & (temp['year'] <= max_test)]
            df_test = pd.concat([df_test, filter_test_temp])

            min_validate = min(list_validate)
            max_validate = max(list_validate)
            filter_validate_temp = temp[(temp['year'] >= min_validate) & (temp['year'] <= max_validate)]
            df_validate = pd.concat([df_validate, filter_validate_temp])

            min_train = min(list_train)
            max_train = max(list_train)
            filter_train_temp = temp[(temp['year'] >= min_train) & (temp['year'] <= max_train)]
            df_train = pd.concat([df_train, filter_train_temp])
        else:
            test_df = temp.copy()
            len_df = len(test_df.index)
            num_test = math.ceil(len_df*0.2)
            num_validate = math.ceil(len_df*0.2)
            filter_test_temp = test_df[num_test*-1:]
            test_df = test_df[:len(test_df)-num_test]
            filter_validate_temp = test_df[num_validate*-1:]
            test_df = test_df[:len(test_df)-num_validate]
            filter_train_temp = test_df

            df_train = pd.concat([df_train, filter_train_temp])
            df_validate = pd.concat([df_validate, filter_validate_temp])
            df_test = pd.concat([df_test, filter_test_temp])
            
    return df_train, df_validate, df_test

def split_realdata(df, test_ratio=0.06):
    today = datetime.today()
    df_test = pd.DataFrame(dtype=str)
    df_train = pd.DataFrame(dtype=str)

    # df['Date'] = pd.to_datetime(df['Date'])
    for tic in list(df['tic_id'].unique()):
        print(tic)
        temp = df[df['tic_id'] == tic]
        temp['year'] = temp['year'].astype('int')
        list_year = list(temp['year'].unique())
        list_year = [y for y in list_year if str(y) != str(today.year)]
        len_year = len(list_year)
        print(f"list_date: {list_year}")
        if len_year > 2:
            num_test = math.ceil(len_year*test_ratio)
            list_test = list_year[int(num_test)*-1:]
            list_train = list_year[:int(num_test)*-1]

            print(f"list_test:{list_test}")
            print(f"list_train: {list_train}")

            min_test = min(list_test)
            max_test = max(list_test)
            filter_test_temp = temp[(temp['year'] >= min_test) & (temp['year'] <= max_test)]
            df_test = pd.concat([df_test, filter_test_temp])

            min_train = min(list_train)
            max_train = max(list_train)
            filter_train_temp = temp[(temp['year'] >= min_train) & (temp['year'] <= max_train)]
            df_train = pd.concat([df_train, filter_train_temp])
        else:
            test_df = temp.copy()
            len_df = len(test_df.index)
            num_test = math.ceil(len_df*0.2)
            filter_test_temp = test_df[num_test*-1:]
            test_df = test_df[:len(test_df)-num_test]
            filter_train_temp = test_df

            df_train = pd.concat([df_train, filter_train_temp])
            df_test = pd.concat([df_test, filter_test_temp])
            
    return df_train, df_test
       