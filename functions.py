from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import talib
import config
from datetime import datetime
import math
from scipy.signal import argrelextrema
from model.prophet_model import pipeline_prophet
import os


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

# wave
def get_zigzag(df, percent=5):
    price = df['close'].values
    highs = argrelextrema(price, np.greater, order=5)[0]
    lows = argrelextrema(price, np.less, order=5)[0]
    pivots = sorted(np.concatenate((highs, lows)))
    pivots_filtered = [pivots[0]]
    for i in range(1, len(pivots)):
        change = abs((price[pivots[i]] - price[pivots_filtered[-1]]) / price[pivots_filtered[-1]]) * 100
        if change >= percent:
            pivots_filtered.append(pivots[i])
    return pivots_filtered

def classify_wave_structure(p1, p2, p3, p4, p5, prices):
    v1, v2, v3, v4, v5 = prices[p1], prices[p2], prices[p3], prices[p4], prices[p5]
    len1 = abs(v2 - v1)
    len3 = abs(v3 - v2)
    len5 = abs(v5 - v4)

    # Rule 1: Wave 2 does not go below the start of Wave 1
    if v2 < v1:
        return None

    # Rule 2: Wave 3 must not be the shortest
    if not (len3 > len1 and len3 > len5):
        return None

    # Rule 3: Wave 4 does not overlap with Wave 1
    if v4 < v1:
        return None

    # Rule 4: Wave 5 goes beyond Wave 3
    if v5 <= v3:
        return None

    # Rule 5: Fibonacci guideline check for Wave 3 ~ 1.618 * Wave 1
    if not np.isclose(len3 / len1, 1.618, atol=0.6):
        return None

    return 'impulse'

def is_diagonal_wave(p1, p2, p3, p4, p5, prices, pivots):
    v1, v2, v3, v4, v5 = prices[p1], prices[p2], prices[p3], prices[p4], prices[p5]
    return (
        v2 > v1 * 0.5 and
        v4 < v1 and  # allow overlap
        v5 > v3 and
        all(abs(prices[pivots[i+1]] - prices[pivots[i]]) < 0.1 * prices[pivots[i]] for i in range(len(pivots)-1))
    )

def label_elliott_patterns(pivots, prices):
    waves = {}
    if len(pivots) < 6:
        return waves

    for i in range(len(pivots) - 5):
        p1, p2, p3, p4, p5 = pivots[i:i+5]
        wave_type = classify_wave_structure(p1, p2, p3, p4, p5, prices)

        if wave_type:
            waves[p1] = '1'
            waves[p2] = '2'
            waves[p3] = '3'
            waves[p4] = '4'
            waves[p5] = '5'
        elif is_diagonal_wave(p1, p2, p3, p4, p5, prices, pivots):
            waves[p1] = '1d'
            waves[p2] = '2d'
            waves[p3] = '3d'
            waves[p4] = '4d'
            waves[p5] = '5d'

    # Try to detect simple ABC correction after the impulse
    if len(pivots) >= 8:
        for i in range(5, len(pivots) - 2):
            a, b, c = pivots[i:i+3]
            va, vb, vc = prices[a], prices[b], prices[c]
            if va > vb < vc and vc < va:
                waves[a] = 'A'
                waves[b] = 'B'
                waves[c] = 'C'

    return waves

def predict_macro_value(df):
    df_copy = df.copy()
    tic_name = list(df_copy['tic'].unique())
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    for tic in tic_name:
        temp = df_copy[df_copy['tic']==tic]
        temp['Date'] = pd.to_datetime(temp['Date'])
        temp = temp.sort_values(by='Date', ascending=True).reset_index(drop=True)
        
        for macro_field in config.MACRO_DATA:
            print(f"forecast: {macro_field}")
            filter_notna = temp[~temp[macro_field].isna()]
            filter_na = temp[temp[macro_field].isna()]
            # print(cal_date)
            if not filter_na.empty:
                date = filter_na['Date'].to_list()
                # print(date)
                cal_date = date[-1]-date[0]
                print(f"date cal: {date[-1]} - {date[0]}")

                dataset = filter_notna[[macro_field,'Date']]
                dataset = dataset.rename(columns={macro_field:"y","Date":"ds"})
                print(f"cal_date: {cal_date.days}")
                forecast = pipeline_prophet(dataset, save_model=False, period=cal_date.days+5)
                forecast = forecast[(forecast['ds'] >= date[0]) & (forecast['ds'] <= date[-1])]
                print("pho forecast:",len(forecast))

                # filter columns
                forecast = forecast[config.PCA_MACRO_DATA_COLUMN]
                
                # generate datetime
                forecast['Date'] = pd.date_range(start=date[0], periods=len(forecast.index), freq='D')
                # merge
                forecast = forecast.rename(columns={config.PCA_MACRO_DATA_COLUMN[0]:f"{macro_field}"})

                df_filter = df_copy[df_copy['tic']==tic]
                df_filter = df_filter[df_filter[macro_field].isna()]
                df_filter = df_filter.drop([macro_field],axis=1)
                print(f"df_filter: {df_filter.columns}")
                merge = pd.merge(df_filter, forecast, how="left", on="Date")

                df_copy = df_copy.dropna(subset=[macro_field])
                df_copy = pd.concat([df_copy,merge])

    df = df.drop(['Year','Month'],axis=1)
    print("columns finish predict",df_copy.columns)
    df_copy = df_copy.reset_index(drop=True)
    return df_copy


def download_and_prepare_data(logging, start_date=None, end_date=None, interval = "1d"):
    from utils import prepare_data, normalization_data
    
    pdata_func = prepare_data()
    norm_func = normalization_data()

    data = pdata_func.download_data(ticker_list=config.TICKET_LIST,start_date=start_date, end_date=end_date, interval=interval)
    data.to_parquet(os.path.join(os.getcwd(), 'data','dataset.parquet'))

    print("columns:",data.columns)

    # data = data[data["tic"].isin(config.TICKET_LIST)]
    data = data[data['Date'].dt.year >=2001]
    data = data.rename(columns=config.MAP_COLUMNS_NAME)

    # data = pdata_func.add_elliott_wave(data)
    data = pdata_func.add_macro_data(data)

    # predict macro value
    # print("before predict",data.columns)
    data = predict_macro_value(data)

    # data.to_csv("test.csv")

    logging.info("prepare data")
    # clean data
    # data = pdata_func.pre_clean_data(data)
    data = data.rename(columns=config.MAP_COLUMNS_NAME)
    print("columns",data.columns)
    data = pdata_func.add_indicator(data, config.INDICATOR_LIST)

    # add resource data
    # data.to_csv("before_add.csv")
    data = pdata_func.add_commodity_data(data)
    data = pdata_func.add_vix_data(data)
    data = pdata_func.add_bond_yields(data)

    data = return_candle_pattern(data)
    data = data.fillna(0)
    # separate date
    data['Date'] = pd.to_datetime(data['Date'])
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data = data.sort_values(by=["Date", "tic"])

    data["pre_7"] = data["close"].pct_change(periods=7).shift(-7) * 100  # เปลี่ยนเป็น %
    data["pre_7"] = np.tanh(data["pre_7"] / 100) * 100
    data["pre_7"] = data["pre_7"].fillna(method="bfill", limit=7)

    # grouping sector in stock
    group_sector = groupping_stock(data, config)
    group_sector = group_sector.sort_values(by=["Date", "tic"])
    group_sector = convert_string2int(group_sector)

    # interpreter data
    group_sector['stochrsi_14'] = group_sector['stochrsi_14']/100
    group_sector['stochrsi_14_decision'] = group_sector['stochrsi_14'].apply(cal_storsi)

    group_sector['rsi_14'] = group_sector['rsi_14']/100
    group_sector['rsi_14_decision'] = group_sector['rsi_14'].apply(cal_rsi)

    group_sector['ichimoku_decision'] = group_sector['ichimoku'].apply(cal_ichimoku)

    group_sector['ema_50100'] = group_sector.apply(cal_ema,args=(50,100),axis=1)
    group_sector['ema_50200'] = group_sector.apply(cal_ema,args=(50,200),axis=1)
    group_sector['ema_50200'] = group_sector.apply(cal_ema,args=(100,200),axis=1)

    # column Outliers
    outliers_column = ['close','high','low','open','volume','vwma_20','ema_200','ema_50','ema_100','macd','ichimoku',"vix","bondyield"]+list(config.COMMODITY.values())+list(config.MACRO_DATA)

    # df_outlier = group_sector[outliers_column]
    group_sector = norm_func.norm_each_row_bylogtransform(group_sector, outliers_column)
    group_sector['ichimoku'] = group_sector['ichimoku'].fillna(-1)
    group_sector['macd'] = group_sector['macd'].fillna(-1)

    # add log transformation with pre_7
    group_sector = norm_func.norm_each_row_bylogtransform(group_sector, ["pre_7"])

    # group_sector = group_sector.round(4)
    group_sector.drop(['Date'], inplace=True, axis=1)

    return group_sector
