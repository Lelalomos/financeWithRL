from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

class normalization_data:
    def __init__(self):
        pass
        
    def normalize_minmax_1d_data(self, dataframe):
        data = dataframe.copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        rsi_scaled = scaler.fit_transform(data.reshape(-1, 1))
        return rsi_scaled
    
    def normalize_robustscaler(self, dataframe):
        data = dataframe.copy()
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(data.reshape(-1, 1))
        return X_scaled
    
    def norm_each_row_minmax(self, data):
        # Create a MinMaxScaler object
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(data)
        normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

        return normalized_df
    
    def norm_log_transform(self, data, column_name, type_log = "support_0"):
        if type_log == "support_0":
            data[f'Log_{column_name}'] = np.log1p(data[column_name])
        else:
            data[f'Log_{column_name}'] = np.log(data[column_name])
        return data
    
    def norm_each_row_bylogtransform(self, data, list_column):
        data_temp = data.copy()
        for column in list_column:
            # use log transform
            data_temp[column] = np.where(data_temp[column] > 0, np.log(data_temp[column]), np.log(data_temp[column] + 1))

        return data_temp

    def norm_power_transform(data, list_column):
        data_temp = data.copy()
        # this method is support negative value
        pt = PowerTransformer(method='yeo-johnson')
        for column in list_column:
            transformed_data = pt.fit_transform(data_temp[column].values.reshape(-1, 1))
            data_temp[column] = pd.DataFrame(transformed_data)

        return data_temp
    
    def norm_quantile_transform(data, list_column):
        data_temp = data.copy()
        qt = QuantileTransformer(output_distribution='normal')  # ปรับให้เป็น Gaussian
        for column in list_column:
            data_transformed = qt.fit_transform(data_temp[column].values.reshape(-1, 1))
            data_temp[column] = pd.DataFrame(data_transformed)

        return data_temp
            

    def norm_winsor(self, data, column_name, type = "normal"):
        """
        type: 
            normal --> normal data
            hight_outlier --> hight outliers in data
            low_outlier --> lower outliers in data
        """
        if type == "normal":
            data[f'{column_name}_winsorized'] = winsorize(data[column_name], limits=(0.02, 0.98))
        elif type == "hight_outlier":
            data[f'{column_name}_winsorized'] = winsorize(data[column_name], limits=(0.01, 0.99))
        elif type == "low_outlier":
            data[f'{column_name}_winsorized'] = winsorize(data[column_name], limits=(0.05, 0.95))

        return data
    
    


        