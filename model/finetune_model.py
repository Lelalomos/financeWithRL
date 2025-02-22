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

    num_stocks = df_validate_set['tic_id'].nunique()+2
    num_group = df_validate_set['group_id'].nunique()+2
    num_month = df_validate_set['month'].nunique()+2
    num_day = df_validate_set['day'].nunique()+2

    print(num_stocks, num_group, num_month, num_day)
    list_except_group = [columns for columns in list(df_validate_set.columns) if columns not in ['tic_id','group_id','month','day']]
    feature = df_validate_set[list_except_group]
    y_val = feature[['pre_7']]
    list_except_group = [columns for columns in list_except_group if columns not in ['pre_7']]
    X_val = feature[list_except_group]
    feature_dim = len(X_val.columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("device:",device)

    # convert to tensor
    stock_tensor = torch.tensor(df_validate_set['tic_id'].astype(int).to_list(), dtype=torch.long)
    group_tensor = torch.tensor(df_validate_set['group_id'].astype(int).to_list(), dtype=torch.long)
    month_tensor = torch.tensor(df_validate_set['month'].astype(int).to_list(), dtype=torch.long)
    day_tensor = torch.tensor(df_validate_set['day'].astype(int).to_list(), dtype=torch.long)
    feature_data = torch.tensor(X_val.to_numpy(), dtype=torch.float32).to(device)
    feature_label = torch.tensor(np.array(y_val.values), dtype=torch.float32).to(device)

    stock_tensor = stock_tensor.unsqueeze(1)
    group_tensor = group_tensor.unsqueeze(1)
    month_tensor = month_tensor.unsqueeze(1)
    feature_data = feature_data.unsqueeze(1)
    day_tensor = day_tensor.unsqueeze(1)

    batch_size = 64
    epochs = 10
    
    dataset = TensorDataset(feature_data,stock_tensor, group_tensor, month_tensor, day_tensor, feature_label)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    output_size = trial.suggest_int("output_size", 64, 256)
    embedding_dim_stock = trial.suggest_int("embedding_dim_stock", num_stocks,num_stocks*2)
    embedding_dim_group = trial.suggest_int("embedding_dim_group", num_group, num_group*2)
    embedding_dim_day = trial.suggest_int("embedding_dim_day", num_day, num_day*2)
    embedding_dim_month = trial.suggest_int("embedding_dim_month", num_month, num_month*2)
    # hidden_size_norm = trial.suggest_int("hidden_size_norm", feature_dim, 256)
    hidden_bilstm = trial.suggest_int("hidden_bilstm", 64, 256)
    num_bilstm = trial.suggest_int("num_bilstm", 1, 5)

    first_layer_hidden_size = trial.suggest_int("first_layer_hidden_size", 64, 256)
    first_layer_size = trial.suggest_int("first_layer_size", 1, 5)
    second_layer_hidden_size = trial.suggest_int("second_layer_hidden_size", 256, 512)
    second_layer_size = trial.suggest_int("second_layer_size", 1, 5)
    third_layer_hidden_size = trial.suggest_int("third_layer_hidden_size", 128, 256)
    third_layer_size = trial.suggest_int("third_layer_size", 1, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
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
                            dropout,
                            hidden_bilstm,
                            num_bilstm).to(device)
    

    criterion = nn.HuberLoss(delta=delta_params)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    for epoch in range(epochs):
        val_loss = 0
        for batch_X, stock_tensor, group_tensor, month_tensor, day_tensor, batch_y in val_loader:
            optimizer.zero_grad()
            batch_X, stock_tensor, group_tensor, month_tensor, day_tensor, batch_y = batch_X.to(device), stock_tensor.to(device), group_tensor.to(device), month_tensor.to(device), day_tensor.to(device), batch_y.to(device)
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
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=500)  # ทดลอง 20 รอบ

    history_df = study.trials_dataframe()

    # สามารถบันทึก DataFrame ลงไฟล์ CSV ได้
    history_df.to_csv('optimization_history.csv', index=False)

    # แสดงค่า Hyperparameter ที่ดีที่สุด
    print("Best Hyperparameters:", study.best_params)
    print("Best Trial:", study.best_trial.number)
    print("Best Parameters:", study.best_trial.params)
    print("Best Loss:", study.best_trial.value)  # ค่านี้ต้องต่ำที่สุด

    '''
    0.7
    {'output_size': 220, 'embedding_dim_stock': 12, 'embedding_dim_group': 8, 'embedding_dim_day': 46, 'embedding_dim_month': 31, 'first_layer_hidden_size': 107, 'first_layer_size': 4, 'second_layer_hidden_size': 304, 'second_layer_size': 1, 'third_layer_hidden_size': 165, 'third_layer_size': 4, 'dropout': 0.312948360649174, 'delta': 0.10003407111863402}
    
    Best Hyperparameters: {'output_size': 139, 'embedding_dim_stock': 49, 'embedding_dim_group': 15, 'embedding_dim_day': 38, 'embedding_dim_month': 35, 'first_layer_hidden_size': 186, 'first_layer_size': 2, 'second_layer_hidden_size': 405, 'second_layer_size': 5, 'third_layer_hidden_size': 202, 'third_layer_size': 1, 'dropout': 0.16918304089580144, 'delta': 0.10013449995372502}
    Best Trial: 183
    Best Parameters: {'output_size': 139, 'embedding_dim_stock': 49, 'embedding_dim_group': 15, 'embedding_dim_day': 38, 'embedding_dim_month': 35, 'first_layer_hidden_size': 186, 'first_layer_size': 2, 'second_layer_hidden_size': 405, 'second_layer_size': 5, 'third_layer_hidden_size': 202, 'third_layer_size': 1, 'dropout': 0.16918304089580144, 'delta': 0.10013449995372502}
    Best Loss: 0.7113943099975586
    '''
