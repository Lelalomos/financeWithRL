from sklearn.preprocessing import MinMaxScaler, RobustScaler

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