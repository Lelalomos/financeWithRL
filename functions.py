from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from model.model import LSTMModel
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
import talib

def search_null_value(df):
    columns_with_null = df.columns[df.isnull().any()]
    print('columns_with_null:',columns_with_null)
    for column in columns_with_null:
        print(f"Null values in column '{column}':")
        print(df[df[column].isnull()])
    
def prepare_data_forlstm(data):
    data = data.dropna(how = "any", ignore_index = True)
    data = data.sort_values(["Date"]).reset_index(drop=True)
    return np.array(data['Close']), np.array(data['rsi_14'])

def detect_outliers_iqr(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    return len(np.where((data < lower_bound) | (data > upper_bound))[0])


def predict_nanvalue_lstm(data, column_name, model_path, default_value = 0):
    if pd.isna(data[column_name]) or data[column_name] in [None,np.nan,""]:
        close = float(data['Close'])
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        x = np.array([[[close]]])
        x = torch.tensor(x)
        x = x.to(torch.float32)
        prediction = model(x).item()
        if np.isnan(prediction):
            prediction = default_value
        return prediction
    else:
        return data[column_name]
    
def predict_nanvalue_lstm_vwma(data, column_name, model_path, default_value = 0):
    if pd.isna(data[column_name]) or data[column_name] in [None,np.nan,""]:
        close = float(data['Close'])
        volumn = float(data['Volume'])
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        x = np.array([[[close,volumn]]])
        x = torch.tensor(x)
        x = x.to(torch.float32)
        prediction = model(x).item()
        if np.isnan(prediction):
            prediction = default_value
        return prediction
    else:
        return data[column_name]
    
def predict_nanvalue_lstm_ichimoku(data, column_name, model_path, default_value = 0):
    if pd.isna(data[column_name]) or data[column_name] in [None,np.nan,""]:
        close = float(data['Close'])
        high = float(data['High'])
        low = float(data['Low'])
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        x = np.array([[[close,high,low]]])
        x = torch.tensor(x)
        x = x.to(torch.float32)
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
    # 'Open', 'High', 'Low', 'Close', 'Volume'
    data['CDL2CROWS'] = talib.CDL2CROWS(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDL3INSIDE'] = talib.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLBELTHOLD'] = talib.CDLBELTHOLD(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLDOJI'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLDOJISTAR'] = talib.CDLDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLENGULFING'] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHAMMER'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHARAMI'] = talib.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHIKKAKE'] = talib.CDLHIKKAKE(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLINNECK'] = talib.CDLINNECK(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLKICKING'] = talib.CDLKICKING(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLLONGLINE'] = talib.CDLLONGLINE(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLMARUBOZU'] = talib.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLMATHOLD'] = talib.CDLMATHOLD(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLONNECK'] = talib.CDLONNECK(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLPIERCING'] = talib.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLSHORTLINE'] = talib.CDLSHORTLINE(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLTAKURI'] = talib.CDLTAKURI(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLTHRUSTING'] = talib.CDLTHRUSTING(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLTRISTAR'] = talib.CDLTRISTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(data['Open'], data['High'], data['Low'], data['Close'])
    data['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(data['Open'], data['High'], data['Low'], data['Close'])
    
    return data
    
    