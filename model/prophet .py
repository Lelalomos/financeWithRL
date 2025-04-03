from prophet import Prophet
import joblib
from datetime import datetime
import os
import pandas as pd

class prophet_model:
    def __init__(self,changepoint_prior_scale =0.3, holidays_prior_scale = 7, seasonality_prior_scale = 7):
        self.prophet = Prophet(
            growth='linear', 
            seasonality_prior_scale = seasonality_prior_scale, 
            holidays_prior_scale = holidays_prior_scale,
            changepoint_prior_scale = changepoint_prior_scale)
        
    def train(self, df_data, save_model = True):
        self.prophet.fit(df_data)
        today = datetime.today()
        ymd = today.strftime("%Y%m%d")

        if save_model:
            joblib.dump(self.prophet, os.path.join()f"prophet-model-{ymd}.pkl")

        return self.prophet

    def predict(self, model, period = 7):
        future = model.make_future_dataframe(period)
        forecast = model.predict(future)
        return forecast
    
def pipeline_prophet(df_data, save_model = True, changepoint_prior_scale = 0.3, holidays_prior_scale = 7, seasonality_prior_scale = 7):
    model = prophet_model(changepoint_prior_scale, holidays_prior_scale, seasonality_prior_scale)
    ppm = model.train(df_data, save_model)
    forecast = model.predict(ppm)
    return forecast

if __name__ == "__main__":
    data = pd.read_parquet(os.path.join(os.getcwd(),'data','real_train_dataset.parquet'))
    forecast = pipeline_prophet(data)


    