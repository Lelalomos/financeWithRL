import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTMModel_HYPER(nn.Module):
    def __init__(self,
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
                 second_layer_hidden_size,
                 third_layer_hidden_size,
                 dropout_value,
                 hidden_bilstm):
        super().__init__()

        # print("num_stocks:",num_stocks)
        # print("num_group:",num_group)
        # print("num_day:",num_day,embedding_dim_day)
        # print("num_month:",num_month)

        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim_stock)
        self.group_embedding = nn.Embedding(num_group, embedding_dim_group)
        self.day_embedding = nn.Embedding(num_day, embedding_dim_day)
        self.month_embedding = nn.Embedding(num_month, embedding_dim_month)

        
        input_dim = embedding_dim_stock + embedding_dim_group + embedding_dim_day+ embedding_dim_month+ feature_dim
        self.bilstm = nn.LSTM(input_dim, hidden_bilstm, 1, batch_first=True, bidirectional=True)
        self.batch_norm_input = nn.BatchNorm1d(hidden_bilstm*2)

        self.lstm1 = nn.LSTM(hidden_bilstm*2, first_layer_hidden_size, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(first_layer_hidden_size*2, second_layer_hidden_size, 1, batch_first=True, bidirectional=True)
        print("input_dim:",input_dim)
        self.lstm3 = nn.LSTM(second_layer_hidden_size*2, third_layer_hidden_size, 1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_value)
        

        self.fc = nn.Linear(third_layer_hidden_size*2, 1)
        # print('finish init')

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)
        
        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        # print("Embedding output shape:", combind_input.shape)

        # print("out bi:",out)
        combind_input, _ = self.bilstm(combind_input)
        combind_input = combind_input[:, -1, :]
        combind_input = self.batch_norm_input(combind_input)
        combind_input = combind_input.unsqueeze(1)
        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)

        out1 = self.dropout(lstm_out3)

        fc_out = self.fc(out1)
        # print("fc_out:",fc_out)
        return fc_out
    
class LSTMModelwithAttention_HYPER(nn.Module):
    def __init__(self,
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
                 second_layer_hidden_size,
                 third_layer_hidden_size,
                 dropout_value,
                 hidden_bilstm,
                 attent_hidden_size):
        super().__init__()

        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim_stock)
        self.group_embedding = nn.Embedding(num_group, embedding_dim_group)

        self.day_embedding = nn.Embedding(num_day, embedding_dim_day)
        self.month_embedding = nn.Embedding(num_month, embedding_dim_month)

        
        input_dim = embedding_dim_stock + embedding_dim_group + embedding_dim_day+ embedding_dim_month+ feature_dim
        self.bilstm = nn.LSTM(input_dim, hidden_bilstm, 1, batch_first=True, bidirectional=True)
        self.batch_norm_input = nn.BatchNorm1d(hidden_bilstm*2)

        self.lstm1 = nn.LSTM(hidden_bilstm*2, first_layer_hidden_size, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(first_layer_hidden_size*2, second_layer_hidden_size, 1, batch_first=True, bidirectional=True)
        print("input_dim:",input_dim)
        self.lstm3 = nn.LSTM(second_layer_hidden_size*2, third_layer_hidden_size, 1, batch_first=True, bidirectional=True)
        print("before attention")
        self.attention = AttentionLayer(third_layer_hidden_size * 2, attent_hidden_size)
        print("after attention")

        self.dropout = nn.Dropout(dropout_value)
        

        self.fc = nn.Linear(attent_hidden_size, 1)
        self.softsign = nn.Softsign()
        # print('finish init')

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)
        
        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        # print("Embedding output shape:", combind_input.shape)

        # print("out bi:",out)
        combind_input, _ = self.bilstm(combind_input)
        # combind_input = combind_input[:, -1, :]
        # combind_input = self.batch_norm_input(combind_input)
        # combind_input = combind_input.unsqueeze(1)
        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)

        context_vector, _ = self.attention(lstm_out3)

        # Flatten context_vector and pass through fully connected layer
        context_vector = context_vector.squeeze(1)
        out1 = self.dropout(context_vector)

        fc_out = self.fc(out1)
        # print("fc_out:",fc_out)
        # return torch.tanh(fc_out)
        # return self.softsign(fc_out)
        return fc_out
    
class AttentionLayer(nn.Module):
    def __init__(self, input_size, attention_size):
        super(AttentionLayer, self).__init__()
        self.query_linear = nn.Linear(input_size, attention_size)
        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.key_linear = nn.Linear(input_size, attention_size)
        self.value_linear = nn.Linear(input_size, attention_size)

    def forward(self, x):
        Q = self.query_linear(x)  # Query
        K = self.key_linear(x)    # Key
        V = self.value_linear(x)  # Value

        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # Q * K^T
        attention_scores = attention_scores / (K.size(-1) ** 0.5)  # Scaling
        # attention_scores = self.tanh(attention_scores)
        attention_weights =  F.softmax(attention_scores, dim=-1)  # Softmax
        
        # Weighted sum of values to produce context vector
        context_vector = torch.bmm(attention_weights, V)

        return context_vector, attention_weights