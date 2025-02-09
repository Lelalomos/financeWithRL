import optuna
import torch
import torch.nn as nn
from model.model import LSTMModel_HYPER

# กำหนดฟังก์ชัน objective (เป้าหมายของการหาค่าที่ดีที่สุด)
def objective(trial, df_validate_set):

    num_stocks = len(df_validate_set['tic'].unique())
    num_group = len(df_validate_set['group_id'].unique())
    num_month = len(df_validate_set['month'].unique())
    num_day = len(df_validate_set['day'].unique())
    feature_dim = len(df_validate_set.columns)
    list_except_group = [columns for columns in list(df_validate_set.columns) if columns not in ['tic','group_id','month','day']]
    feature = df_validate_set[list_except_group]

    output_size = trial.suggest_int("output_size", 64, 256)
    embedding_dim_stock = trial.suggest_int("embedding_dim_stock", 5, 20)
    embedding_dim_group = trial.suggest_int("embedding_dim_group", 5, 20)
    embedding_dim_day = trial.suggest_int("embedding_dim_day", 5, 20)
    embedding_dim_month = trial.suggest_int("embedding_dim_month", 5, 20)
    hidden_size_norm = trial.suggest_int("hidden_size_norm", 32, 256)
    first_layer_hidden_size = trial.suggest_int("first_layer_hidden_size", 64, 256)
    first_layer_size = trial.suggest_int("first_layer_size", 1, 5)
    second_layer_hidden_size = trial.suggest_int("second_layer_hidden_size", 256, 512)
    second_layer_size = trial.suggest_int("second_layer_size", 1, 5)
    third_layer_hidden_size = trial.suggest_int("third_layer_hidden_size", 128, 256)
    third_layer_size = trial.suggest_int("third_layer_size", 1, 5)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

    # สร้างโมเดล LSTM
    lstm_model = LSTMModel_HYPER(output_size,
                            num_stocks,
                            num_group,
                            num_day,
                            num_month,
                            embedding_dim_stock,
                            embedding_dim_group,
                            embedding_dim_day,
                            embedding_dim_month,
                            feature_dim,
                            hidden_size_norm,
                            first_layer_hidden_size,
                            first_layer_size,
                            second_layer_hidden_size,
                            second_layer_size,
                            third_layer_hidden_size,
                            third_layer_size,
                            dropout)
    
    # convert to tensor
    stock_tensor = torch.tensor(df_validate_set['tic'].to_list())
    group_tensor = torch.tensor(df_validate_set['group_id'].to_list())
    month_tensor = torch.tensor(df_validate_set['month'].to_list())
    day_tensor = torch.tensor(df_validate_set['day'].to_list())
    feature_tensor = torch.tensor(feature, dtype=torch.float32)

    
    loss = 0.2  # ค่านี้ควรได้จากการเทรนโมเดลจริงๆ
    
    return loss  # Optuna จะพยายามลด loss ให้น้อยที่สุด


if __name__ == "__main__":
    

    # เริ่มต้น Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # ทดลอง 20 รอบ

    # แสดงค่า Hyperparameter ที่ดีที่สุด
    print("Best Hyperparameters:", study.best_params)
