import sys
sys.path.append("/app")

from utils import prepare_data, normalization_data
import pandas as pd
import numpy as np
from functions import return_candle_pattern, groupping_stock, cal_rsi,cal_storsi, cal_ichimoku, cal_ema, convert_string2int, predict_macro_value
import config
from model.model import LSTMModel, LSTMModelwithAttention, LSTMModelxTNCwithAttention
import torch
import os
from datetime import datetime

class backtest:
    def __init__(self):
        self.predata_func = prepare_data()
        self.norm_func = normalization_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, start_date, end_date):
        data = self.predata_func.download_data(config.TICKET_LIST, start_date=start_date, end_date=end_date, interval="1d")
        data = data.rename(columns=config.MAP_COLUMNS_NAME)
        data = self.predata_func.add_elliott_wave(data)
        print(data)
        print(data.columns)
        if config.MODEL == 'LSTMxTNCwithAttention':
            data = self.predata_func.add_macro_data(data)
            # predict macro value
            print("before predict",data.columns)
            data = predict_macro_value(data)

        data = self.predata_func.add_indicator(data, config.INDICATOR_LIST)
        data = self.predata_func.add_commodity_data(data)
        data = self.predata_func.add_vix_data(data)
        data = self.predata_func.add_bond_yields(data)
        data = return_candle_pattern(data)
        data = data.fillna(0)
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
        if config.MODEL == 'LSTMxTNCwithAttention':
            outliers_column = ['close','high','low','open','volume','vwma_20','ema_200','ema_50','ema_100','macd','ichimoku',"vix","bondyield"]+list(config.COMMODITY.values())+['Elliott_Wave_Label']+list(config.MACRO_DATA)
        else:
            outliers_column = ['close','high','low','open','volume','vwma_20','ema_200','ema_50','ema_100','macd','ichimoku',"vix","bondyield"]+list(config.COMMODITY.values())+['Elliott_Wave_Label']

        # df_outlier = group_sector[outliers_column]
        group_sector = self.norm_func.norm_each_row_bylogtransform(group_sector, outliers_column)
        group_sector['ichimoku'] = group_sector['ichimoku'].fillna(-1)
        group_sector['macd'] = group_sector['macd'].fillna(-1)

        # add log transformation with pre_7
        group_sector = self.norm_func.norm_each_row_bylogtransform(group_sector, ["pre_7"])

        # group_sector = group_sector.round(4)
        group_sector.drop(['Date','year'], inplace=True, axis=1)
        return group_sector
    
    def test(self, data, model_path):
        num_stocks = len(data['tic_id'].unique())
        num_group = len(data['group_id'].unique())
        num_month = 13
        num_day = 32
        list_except_group = [columns for columns in list(data.columns) if columns not in ['tic_id','group_id','month','day']]
        feature = data[list_except_group]
        y_val = feature[['pre_7']]
        list_except_group = [columns for columns in list_except_group if columns not in ['pre_7']]
        X_val = feature[list_except_group]
        feature_dim = len(X_val.columns)
        X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32).to(self.device)
        X_val = X_val.unsqueeze(1)

        stock_tensor = data['tic_id'].astype(int).to_list()
        group_tensor = data['group_id'].astype(int).to_list()
        month_tensor = data['month'].astype(int).to_list()
        day_tensor = data['day'].astype(int).to_list()

        stock_tensor = torch.tensor(stock_tensor, dtype=torch.long)
        group_tensor = torch.tensor(group_tensor, dtype=torch.long)
        month_tensor = torch.tensor(month_tensor, dtype=torch.long)
        day_tensor = torch.tensor(day_tensor, dtype=torch.long)

        stock_tensor = stock_tensor.unsqueeze(1)
        group_tensor = group_tensor.unsqueeze(1)
        month_tensor = month_tensor.unsqueeze(1)
        day_tensor = day_tensor.unsqueeze(1)

        print(f"model: {config.MODEL}")
        if config.MODEL == "lstm_with_attention":
            lstm_model = LSTMModelwithAttention(
                    feature_dim,
                    num_stocks,
                    num_group,
                    num_day,
                    num_month,
                    config
            ).to(self.device)
        elif config.MODEL == 'LSTMxTNCwithAttention':
            lstm_model = LSTMModelxTNCwithAttention(
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config
            ).to(self.device)

        lstm_model.load_state_dict(torch.load(model_path))
        lstm_model.eval()

        with torch.no_grad():
            stock_tensor = stock_tensor.to(self.device)
            group_tensor = group_tensor.to(self.device)
            day_tensor = day_tensor.to(self.device)
            month_tensor = month_tensor.to(self.device)
            X_val = X_val.to(self.device)
            outputs = lstm_model(stock_tensor, group_tensor, day_tensor, month_tensor, X_val)
            # print(outputs)

        print(y_val)
        df = pd.DataFrame(np.hstack((outputs.cpu().numpy(), y_val, month_tensor.cpu().numpy(), day_tensor.cpu().numpy(), group_tensor.cpu().numpy(), stock_tensor.cpu().numpy())), columns=['predictions', 'actuals', 'month', 'day', 'group','stock'])
        df.to_excel("backtest.xlsx")
        # print("Predicted Value:", outputs.item())

        return df


if __name__ == "__main__":
    bk = backtest()
    # if os.path.isfile("data/test_real.parquet"):
    #     data = pd.read_parquet("data/test_real.parquet")
    # else:
    data = bk.prepare_data('2025-03-01','2025-04-01')
    data.to_parquet("data/test_real.parquet")

    os.makedirs(os.path.join(os.getcwd(),'output'),exist_ok=True)
    output = bk.test(data, "saved_model/20250428_lstm_model.pth")
    today = datetime.today()
    ymd = today.strftime("%Y%m%d-%H%M%S")
    path_output = os.path.join(os.getcwd(),'output',f'backtest-{ymd}.xlsx')
    output.to_excel(path_output)

    # read map data
    mapping_df = pd.read_excel("interpret.xlsx",dtype=str,index_col=0)
    tic_mapping = mapping_df.groupby("tic_label")["tic_str"].apply(lambda x: list(set(x))[0]).to_dict()
    group_mapping = mapping_df.groupby("group_label")["group_str"].apply(lambda x: list(set(x))[0]).to_dict()
    # print(f"tic_mapping: {tic_mapping}")

    output = pd.read_excel(path_output, index_col=0, dtype=str)
    output = output.reset_index()
    output["stock_name"] = output["stock"].map(tic_mapping)
    output["group_name"] = output["group"].map(group_mapping)
    output = output.drop(['stock','group'], axis=1)

    output.to_excel(os.path.join(os.getcwd(),'output',f"result-{ymd}.xlsx"))



        
        