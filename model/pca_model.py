from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas_datareader.data as web
from datetime import datetime, timedelta
from model.prophet_model import pipeline_prophet

class pca_model:
    def __init__(self, componant = 2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components= componant)

    def norm(self, dataframe):
        features_scaled = self.scaler.fit_transform(dataframe)
        return features_scaled
    
    def start(self, norm_data, dataframe):
        print(type(dataframe))
        pca_components = self.pca.fit_transform(norm_data)
        dataframe['PCA1'] = pca_components[:, 0]
        return dataframe


if __name__ == "__main__":
    start = datetime(2000, 1, 1)
    end = datetime.today()-timedelta(days=30)
    data = web.DataReader('FEDFUNDS', 'fred', start, end)
    data = data.reset_index()
    data = data.rename(columns={"DATE":"ds","FEDFUNDS":"y"})
    forecast = pipeline_prophet(data)
    list_column = ['yhat','yhat_lower','yhat_upper','trend_lower','trend_upper',"yearly","yearly_lower","yearly_upper"]

    _pca = pca_model(componant=1)
    forecast_data = forecast[list_column]
    norm_data = _pca.norm(forecast_data)
    forecasted = _pca.start(norm_data, forecast_data)

    print(forecasted.tail())

