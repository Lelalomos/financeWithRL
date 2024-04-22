from prepare_data import prepare_data
from normalization import normalization_data
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import LSTMModel
import torch.nn as nn
import matplotlib.pyplot as plt

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
        print('shape:',train_X.shape)
        train_dataset = TensorDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train)
        model = LSTMModel(input_size=train_X.shape[1]).to(self.device)
        
        # Train the model
        for epoch in range(self.epochs):
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                self.list_loss.append(loss.item())
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
                    
                if self.debug_loss:
                    if loss.item() < self.threshold_loss:
                        print(f'Loss is below threshold ({self.threshold_loss}), saving the model...')
                        torch.save(model.state_dict(), 'model.pth')
                        break

        
    def eval(self, test_X, test_Y):
        test_X = torch.tensor(test_X, dtype=torch.float32).to(self.device)
        test_Y = torch.tensor(test_Y, dtype=torch.float32).to(self.device)
        model = LSTMModel(input_size=test_X.shape[1]).to(self.device)
        test_dataset = TensorDataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle_test)
        
        # Test the model
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                test_loss += loss.item()
        test_loss /= len(test_loader)
        
        print(f'Test MSE: {test_loss}')
        
        
    def for_rsi(self, list_indicator, indicator_name):
        data2train_rsi = self.prepare_data(list_indicator)
        # convert data to array
        close_value = np.array(data2train_rsi['Close'])
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
        
    def plot_image(self, save_image):
        plt.plot(self.list_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(save_image)  # Save the plot as an image file
        plt.close()
    
    def export_model(self, save_model):
        torch.save(self.model.state_dict(), save_model)
        
        
    
    
        
        
        