import optuna
import torch
import torch.nn as nn
from model.model import LSTMModel_HYPER
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

# กำหนดฟังก์ชัน objective (เป้าหมายของการหาค่าที่ดีที่สุด)
def objective(trial):
    df_validate_set = pd.read_parquet(os.path.join(os.getcwd(),'data','validate_dataset.parquet'))

    num_stocks = len(df_validate_set['tic'].unique())
    num_group = len(df_validate_set['group_id'].unique())
    num_month = len(df_validate_set['month'].unique())
    num_day = len(df_validate_set['day'].unique())
    feature_dim = len(df_validate_set.columns)
    list_except_group = [columns for columns in list(df_validate_set.columns) if columns not in ['tic','group_id','month','day']]
    feature = df_validate_set[list_except_group]
    y_val = feature[['pre_7']]
    X_val = feature[list_except_group]

    # convert to tensor
    stock_tensor = torch.tensor(df_validate_set['tic'].to_list())
    group_tensor = torch.tensor(df_validate_set['group_id'].to_list())
    month_tensor = torch.tensor(df_validate_set['month'].to_list())
    day_tensor = torch.tensor(df_validate_set['day'].to_list())
    feature_data = torch.tensor(X_val, dtype=torch.float32)
    feature_label = torch.tensor(y_val, dtype=torch.float32)

    batch_size = 64
    epochs = 10
    
    dataset = TensorDataset(feature_data,stock_tensor, group_tensor, month_tensor, day_tensor, feature_label)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
                            hidden_size_norm,
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

            output = lstm_model(batch_X, stock_tensor, group_tensor, month_tensor, day_tensor)  # ส่งข้อมูลเข้า LSTM
            loss = criterion(output, batch_y)  # คำนวณ loss

            loss.backward()
            optimizer.step()

            val_loss += loss.item()

        avg_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Validation loop
        lstm_model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # No gradient calculation during validation
            for batch_X, stock_tensor, group_tensor, month_tensor, day_tensor, batch_y in val_loader:
                output = lstm_model(batch_X, stock_tensor, group_tensor, month_tensor, day_tensor)  # Forward pass
                val_loss = criterion(output, batch_y)  # Calculate validation loss
                running_val_loss += val_loss.item()  # Accumulate validation loss
                predicted = (output > 0).float()  # Predict 0 or 1
                correct_val += (predicted == batch_y).sum().item()  # Count correct predictions
                total_val += batch_y.size(0)  # Count total samples

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        return avg_loss


if __name__ == "__main__":

    # เริ่มต้น Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # ทดลอง 20 รอบ

    # แสดงค่า Hyperparameter ที่ดีที่สุด
    print("Best Hyperparameters:", study.best_params)
