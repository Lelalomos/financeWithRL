from utils import return_logs
from functions import  split_dataset, split_realdata, download_and_prepare_data
import os
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def main():
    os.makedirs(os.path.join(os.getcwd(),'logs'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'logs_images'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'saved_model'),exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),'data'),exist_ok=True)
    
    logging = return_logs(os.path.join(os.getcwd(),'logs','process.log'))

    group_sector = download_and_prepare_data(logging)
    # save dataset
    temp_data = group_sector.copy()
    temp_data.to_parquet("saved_model/clean_model.parquet")

    # ต้องเพิ่ม label ว่าต้องการแบบไหน
    # split train, validate, test
    train_set, validate_set, test_set = split_dataset(group_sector)
    real_train_dataset, real_test_dataset = split_realdata(group_sector)

    real_train_dataset = real_train_dataset.drop(["year"],axis=1)
    real_test_dataset = real_test_dataset.drop(["year"],axis=1)

    real_test_dataset.to_parquet(os.path.join(os.getcwd(),"data","real_test_dataset.parquet"))
    real_train_dataset.to_parquet(os.path.join(os.getcwd(),"data","real_train_dataset.parquet"))

    validate_set = validate_set.drop(["year"],axis=1)
    test_set = test_set.drop(["year"],axis=1)
    train_set = train_set.drop(["year"],axis=1)


    validate_set.to_parquet(os.path.join(os.getcwd(),"data","validate_dataset.parquet"))
    test_set.to_parquet(os.path.join(os.getcwd(),"data","test_dataset.parquet"))
    train_set.to_parquet(os.path.join(os.getcwd(),"data","train_dataset.parquet"))


if __name__ == "__main__":
    main()