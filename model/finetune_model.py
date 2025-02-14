import optuna
import torch
import torch.nn as nn
from model import LSTMModel_HYPER
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import numpy as np

# กำหนดฟังก์ชัน objective (เป้าหมายของการหาค่าที่ดีที่สุด)
def objective(trial):
    df_validate_set = pd.read_parquet(os.path.join(os.getcwd(),'data','validate_dataset.parquet'))

    num_stocks = len(df_validate_set['tic_id'].unique())
    num_group = len(df_validate_set['group_id'].unique())
    num_month = len(df_validate_set['month'].unique())+1
    num_day = len(df_validate_set['day'].unique())+1
    list_except_group = [columns for columns in list(df_validate_set.columns) if columns not in ['tic_id','group_id','month','day']]
    feature = df_validate_set[list_except_group]
    y_val = feature[['pre_7']]
    list_except_group = [columns for columns in list_except_group if columns not in ['pre_7']]
    X_val = feature[list_except_group]
    feature_dim = len(X_val.columns)

    # convert to tensor
    stock_tensor = torch.tensor(df_validate_set['tic_id'].astype(int).to_list(), dtype=torch.long)
    group_tensor = torch.tensor(df_validate_set['group_id'].astype(int).to_list(), dtype=torch.long)
    month_tensor = torch.tensor(df_validate_set['month'].astype(int).to_list(), dtype=torch.long)
    day_tensor = torch.tensor(df_validate_set['day'].astype(int).to_list(), dtype=torch.long)
    feature_data = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    feature_label = torch.tensor(np.array(y_val.values), dtype=torch.float32)

    batch_size = 64
    epochs = 10
    
    dataset = TensorDataset(feature_data,stock_tensor, group_tensor, month_tensor, day_tensor, feature_label)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    output_size = trial.suggest_int("output_size", 64, 256)
    embedding_dim_stock = trial.suggest_int("embedding_dim_stock", num_stocks+1, 50)
    embedding_dim_group = trial.suggest_int("embedding_dim_group", num_group+1, 50)
    embedding_dim_day = trial.suggest_int("embedding_dim_day", num_day+1, 50)
    embedding_dim_month = trial.suggest_int("embedding_dim_month", num_month+1, 50)
    # hidden_size_norm = trial.suggest_int("hidden_size_norm", feature_dim, 256)
    first_layer_hidden_size = trial.suggest_int("first_layer_hidden_size", 64, 256)
    first_layer_size = trial.suggest_int("first_layer_size", 1, 5)
    second_layer_hidden_size = trial.suggest_int("second_layer_hidden_size", 256, 512)
    second_layer_size = trial.suggest_int("second_layer_size", 1, 5)
    third_layer_hidden_size = trial.suggest_int("third_layer_hidden_size", 128, 256)
    third_layer_size = trial.suggest_int("third_layer_size", 1, 5)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    delta_params = trial.suggest_float("delta", 0.1, 10.0) 

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
                            first_layer_hidden_size,
                            first_layer_size,
                            second_layer_hidden_size,
                            second_layer_size,
                            third_layer_hidden_size,
                            third_layer_size,
                            dropout)
    

    criterion = nn.HuberLoss(delta=delta_params)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    for epoch in range(epochs):
        val_loss = 0
        for batch_X, stock_tensor, group_tensor, month_tensor, day_tensor, batch_y in val_loader:
            optimizer.zero_grad()
            output = lstm_model(stock_tensor, group_tensor, day_tensor, month_tensor, batch_X)  # ส่งข้อมูลเข้า LSTM
            # print(output)
            loss = criterion(output, batch_y)  # คำนวณ loss

            loss.backward()
            optimizer.step()

            val_loss += loss.item()

        avg_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return loss.item()


if __name__ == "__main__":

    # เริ่มต้น Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # ทดลอง 20 รอบ

    # แสดงค่า Hyperparameter ที่ดีที่สุด
    print("Best Hyperparameters:", study.best_params)
