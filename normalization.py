from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd

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