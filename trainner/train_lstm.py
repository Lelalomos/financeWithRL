from utils import prepare_data
from utils import normalization_data
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import LSTMModel
import torch.nn as nn
import matplotlib.pyplot as plt
import config
import os

class train_lstm:
    def __init__(self, 
                 data = None, 
                 epochs = 200, 
                 debug_loss = True,
                 threshold_loss = 0.02, 
                 batch_size = 64, 
                 shuffle_train = None, 
                 shuffle_test= None,
                 splitdata_rd_stage4test = None,
                 splitdata_test_size = 0.3,
                 splitdata_shuffle = False,
                 path_save_loss = "loss.jpg",
                 path_save_model = "model.pth"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:',self.device)
        self.data = data
        self.epochs = epochs
        self.threshold_loss = threshold_loss
        self.pre_data = prepare_data()
        self.normalize = normalization_data()
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test
        self.splitdata_rd_stage4test = splitdata_rd_stage4test
        self.splitdata_test_size = splitdata_test_size
        self.splitdata_shuffle = splitdata_shuffle
        self.debug_loss = debug_loss
        self.model = LSTMModel().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.list_loss = []
        self.path_save_loss = path_save_loss
        self.path_save_model = path_save_model
        
        if data.empty:
            raise "data is empty"
        
    def prepare_data(self, list_indicator):
        data_preparing = self.data.copy()
        data_preparing = self.pre_data.pre_clean_data(data_preparing)
        data_preparing = self.pre_data.add_technical_indicator(data_preparing, list_indicator)
        # sort data with date
        data_preparing = data_preparing.sort_values(["Date"]).reset_index(drop=True)
        # drop nan value each rows
        data_preparing = data_preparing.dropna(how = "any", ignore_index = True)
        return data_preparing
    
    def train(self, train_X, train_Y):
        train_X = torch.tensor(train_X, dtype=torch.float32).to(self.device)
        train_Y = torch.tensor(train_Y, dtype=torch.float32).to(self.device)
        print('shape:',train_X.shape[1])
        train_dataset = TensorDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train)
        self.model = LSTMModel(input_size=train_X.shape[1]).to(self.device)
        
        print("-"*20,"Model","-"*20)
        for name, param in self.model.state_dict().items():
            print(f"{name}: {param.shape}")
        print("-"*50)
        
        # Train the model
        for epoch in range(self.epochs):
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                self.list_loss.append(loss.item())
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
                    
                if self.debug_loss:
                    if loss.item() < self.threshold_loss:
                        print(f'Loss is below threshold ({self.threshold_loss}), saving the model...')
                        torch.save(self.model.state_dict(), 'model.pth')
                        break

        
    def eval(self, test_X, test_Y):
        test_X = torch.tensor(test_X, dtype=torch.float32).to(self.device)
        test_Y = torch.tensor(test_Y, dtype=torch.float32).to(self.device)
        self.model = LSTMModel(input_size=test_X.shape[1]).to(self.device)
        test_dataset = TensorDataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle_test)
        
        # Test the model
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                test_loss += loss.item()
        test_loss /= len(test_loader)
        
        print(f'Test MSE: {test_loss}')
        
        
    def for_single_feature(self, list_indicator, indicator_name, column_select):
        data2train_rsi = self.prepare_data(list_indicator)
        # convert data to array
        close_value = np.array(data2train_rsi[column_select])
        rsi_value = np.array(data2train_rsi[indicator_name])
        # filter outlier with z-score
        data, outlier = self.pre_data.cal_zscore(close_value)
        label_filtered = np.delete(rsi_value, outlier)
        # normalize data and label
        norm_data = self.normalize.normalize_minmax_1d_data(data)
        norm_label = self.normalize.normalize_minmax_1d_data(label_filtered)
        # split data to train and test
        
        print('percent of test:',self.splitdata_test_size)
        x_train, x_test, y_train, y_test = train_test_split(norm_data, 
                                                            norm_label, 
                                                            random_state = self.splitdata_rd_stage4test, 
                                                            test_size = self.splitdata_test_size,
                                                            shuffle = self.splitdata_shuffle)
        print('len train:',len(x_train))
        print('len test:',len(x_test))
        self.train(x_train, y_train)
        self.eval(x_test, y_test)
        self.plot_image(self.path_save_loss)
        self.export_model(self.path_save_model)
        
    def for_multiple_feature(self, list_indicator, column_select_data, column_select_label):
        data2train_rsi = self.prepare_data(list_indicator)
        
        data = data2train_rsi[column_select_data+column_select_label]
        print('len data:',len(data))
        filter_out = self.pre_data.cal_zscore_df(data)
        print('filter_out:',len(filter_out))
        print(filter_out)
        
        label = filter_out[column_select_label]
        data = filter_out[column_select_data]
        # df_label = label.to_numpy()
        
        print('len train:',len(data))
        print('len test:',len(label))
        
        # data['Close'] = self.normalize.normalize_minmax_1d_data(data['Close'])
        norm_data = self.normalize.norm_each_row_minmax(data)
        norm_label = self.normalize.norm_each_row_minmax(label)
        print("norm_data:",norm_data)
        
        
        x_train, x_test, y_train, y_test = train_test_split(norm_data, 
                                                            norm_label, 
                                                            random_state = self.splitdata_rd_stage4test, 
                                                            test_size = self.splitdata_test_size,
                                                            shuffle = self.splitdata_shuffle)
        # extract from dataframe
        x_train = np.array(x_train)
        y_train = np.array(y_train.values)
        x_test = np.array(x_test)
        y_test = np.array(y_test.values)
        print('data')
        print(x_train.shape)
        print('label')
        print(y_train.shape)
        # print('len train:',len(x_train))
        # print('len test:',len(x_test))
        self.train(x_train, y_train)
        self.eval(x_test, y_test)
        self.plot_image(self.path_save_loss)
        self.export_model(self.path_save_model)
        
    def plot_image(self, save_image):
        plt.plot(self.list_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(save_image)  # Save the plot as an image file
        plt.close()
    
    def export_model(self, save_model):
        torch.save(self.model.state_dict(), save_model)
        
def train_lstm4pred_singlefeature(indicator_name, save_name, batch_size = 500, epochs=100):
    prepare = prepare_data()
    data = prepare.download_data(config.TICKET_LIST)
    train_data = train_lstm(data, threshold_loss = 0.001, batch_size = batch_size, path_save_loss= os.path.join(os.getcwd(),'logs_images',f'loss_{save_name}.jpg'), path_save_model= os.path.join(os.getcwd(),'saved_model',f'{save_name}_model.pth'), epochs = epochs, splitdata_test_size=0.2)
    train_data.for_single_feature(config.INDICATOR_LIST, indicator_name, 'Close')
    
def train_lstm4pred_multifeature(indicator_name, list_column_data, list_column_label, batch_size = 500, epochs=100):
    prepare = prepare_data()
    data = prepare.download_data(config.TICKET_LIST)
    train_data = train_lstm(data, threshold_loss = 0.001, batch_size = batch_size, path_save_loss= os.path.join(os.getcwd(),'logs_images',f'loss_{indicator_name}.jpg'), path_save_model= os.path.join(os.getcwd(),'saved_model',f'{indicator_name}_model.pth'), epochs=epochs, splitdata_test_size=0.2)
    train_data.for_multiple_feature(config.INDICATOR_LIST, list_column_data, list_column_label)
    
if __name__ == "__main__":
    # train_lstm4pred_singlefeature('rsi_14')
    # train_lstm4pred_singlefeature('stochrsi_14')
    # train_lstm4pred_singlefeature('tema_200')
    train_lstm4pred_multifeature('vwma_14',["Close","Volume"], ['vwma_14'])
    # train_lstm4pred_multifeature('ichimoku',["High","Low","Close"], ['ichimoku'])
    