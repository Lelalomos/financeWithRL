import torch.nn as nn
import torch
import torch.nn.functional as F
        
# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, 
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config):
        super(LSTMModel, self).__init__()

        config = config.LSTM_PARAMS
        self.stock_embedding = nn.Embedding(num_stocks, config["embedding_dim_stock"])
        self.group_embedding = nn.Embedding(num_group, config['embedding_dim_group'])
        self.day_embedding = nn.Embedding(num_day, config['embedding_dim_day'])
        self.month_embedding = nn.Embedding(num_month, config['embedding_dim_month'])

        input_dim = config['embedding_dim_stock'] + config['embedding_dim_group'] + config['embedding_dim_day'] + config['embedding_dim_month'] + feature_dim
        self.lstm1 = nn.LSTM(input_dim, config['first_layer_hidden_size'], 1, batch_first=True)
        self.lstm2 = nn.LSTM(config["first_layer_hidden_size"], config['second_layer_hidden_size'], 1, batch_first=True)
        self.lstm3 = nn.LSTM(config['second_layer_hidden_size'], config['third_layer_hidden_size'], 1, batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(config["third_layer_hidden_size"], 1)

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)

        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)

        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)
        out1 = self.dropout(lstm_out3)
        fc_out = self.fc(out1)
        return fc_out
    

class LSTMModelwithAttention(nn.Module):
    def __init__(self, 
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config):
        super(LSTMModelwithAttention, self).__init__()

        config = config.LSTM_ATTENTION_PARAMS
        self.stock_embedding = nn.Embedding(num_stocks, config["embedding_dim_stock"])
        self.group_embedding = nn.Embedding(num_group, config['embedding_dim_group'])
        self.day_embedding = nn.Embedding(num_day, config['embedding_dim_day'])
        self.month_embedding = nn.Embedding(num_month, config['embedding_dim_month'])

        input_dim = config["embedding_dim_stock"] + config['embedding_dim_group'] + config['embedding_dim_day']+ config['embedding_dim_month'] + feature_dim
        
        self.bilstm = nn.LSTM(input_dim, config['hidden_bilstm'], 1, batch_first=True, bidirectional=True)
        self.batch_norm_input = nn.BatchNorm1d(config['hidden_bilstm']*2)

        self.lstm1 = nn.LSTM(config['hidden_bilstm']*2, config['first_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(config['first_layer_hidden_size']*2, config['second_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        print("input_dim:",input_dim)
        self.lstm3 = nn.LSTM(config['second_layer_hidden_size']*2, config['third_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        print("before attention")
        self.attention = AttentionLayer(config['third_layer_hidden_size'] * 2, config['attent_hidden_size'])
        print("after attention")

        self.dropout = nn.Dropout(config['dropout'])
        
        self.fc = nn.Linear(config['attent_hidden_size'], 1)
        self.softsign = nn.Softsign()

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)

        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        
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
    
class LSTMModelwithMultiheadAttention(nn.Module):
    def __init__(self, 
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config):
        super(LSTMModelwithMultiheadAttention, self).__init__()

        config = config.LSTMModelWithMultiAttention_PARAMS
        self.stock_embedding = nn.Embedding(num_stocks, config["embedding_dim_stock"])
        self.group_embedding = nn.Embedding(num_group, config['embedding_dim_group'])
        self.day_embedding = nn.Embedding(num_day, config['embedding_dim_day'])
        self.month_embedding = nn.Embedding(num_month, config['embedding_dim_month'])

        input_dim = config["embedding_dim_stock"] + config['embedding_dim_group'] + config['embedding_dim_day']+ config['embedding_dim_month'] + feature_dim
        
        self.bilstm = nn.LSTM(input_dim, config['hidden_bilstm'], 1, batch_first=True, bidirectional=True)
        self.batch_norm_input = nn.BatchNorm1d(config['hidden_bilstm']*2)

        self.lstm1 = nn.LSTM(config['hidden_bilstm']*2, config['first_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(config['first_layer_hidden_size']*2, config['second_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        print("input_dim:",input_dim)
        self.lstm3 = nn.LSTM(config['second_layer_hidden_size']*2, config['third_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=config['third_layer_hidden_size'] * 2, num_heads=config['num_head_attention'], batch_first=True)
        print("after attention")

        self.dropout = nn.Dropout(config['dropout'])
        
        self.fc = nn.Linear(config['third_layer_hidden_size'] * 2, 1)
        self.softsign = nn.Softsign()

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)

        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        
        combind_input, _ = self.bilstm(combind_input)
        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)

        attn_out, _ = self.attention(lstm_out3,lstm_out3,lstm_out3)
        context_vector = attn_out.mean(dim=1)
        out1 = self.dropout(context_vector)

        fc_out = self.fc(out1)
        return fc_out
    
class LSTMModelxCNNwithAttention(nn.Module):
    def __init__(self, 
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config):
        super(LSTMModelxCNNwithAttention, self).__init__()

        config = config.LSTMxCNN_ATTENTION_PARAMS
        self.stock_embedding = nn.Embedding(num_stocks, config["embedding_dim_stock"])
        self.group_embedding = nn.Embedding(num_group, config['embedding_dim_group'])
        self.day_embedding = nn.Embedding(num_day, config['embedding_dim_day'])
        self.month_embedding = nn.Embedding(num_month, config['embedding_dim_month'])

        input_dim = config["embedding_dim_stock"] + config['embedding_dim_group'] + config['embedding_dim_day']+ config['embedding_dim_month'] + feature_dim
        
        # self.tcn = TCN(input_size=input_dim, num_channels=config['tcn_chanel'], kernel_size=config['tcn_kernel'], dropout=0.2)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=config['cnn_chanel1'], kernel_size=1,padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=config['cnn_chanel1'], out_channels=config['cnn_chanel2'], kernel_size=1,padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=1,stride=2)
        )

        self.bilstm = nn.LSTM(config['cnn_chanel2'], config['hidden_bilstm'], 1, batch_first=True, bidirectional=True)
        self.batch_norm_input = nn.BatchNorm1d(config['hidden_bilstm']*2)

        self.lstm1 = nn.LSTM(config['hidden_bilstm']*2, config['first_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(config['first_layer_hidden_size']*2, config['second_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(config['second_layer_hidden_size']*2, config['third_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        print("before attention")
        self.attention = AttentionLayer(config['third_layer_hidden_size'] * 2, config['attent_hidden_size'])
        print("after attention")

        self.dropout = nn.Dropout(config['dropout'])
        
        self.fc = nn.Linear(config['attent_hidden_size'], 1)
        self.softsign = nn.Softsign()

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)

        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        combind_input = combind_input.transpose(1, 2)
        # print(f"tcn: {combind_input.shape}")
        combind_input = self.cnn(combind_input)        # Output: (batch, tcn_channels[-1], seq_len)
        # print("tcn final")
        combind_input = combind_input.transpose(1, 2)
        
        # combind_input = combind_input.transpose(1, 2)
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
        return fc_out
    
class LSTMModelxCNNxNORMWithAttention(nn.Module):
    def __init__(self, 
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config):
        super(LSTMModelxCNNxNORMWithAttention, self).__init__()

        config = config.LSTMxCNNxNORM_ATTENTION_PARAMS
        self.stock_embedding = nn.Embedding(num_stocks, config["embedding_dim_stock"])
        self.group_embedding = nn.Embedding(num_group, config['embedding_dim_group'])
        self.day_embedding = nn.Embedding(num_day, config['embedding_dim_day'])
        self.month_embedding = nn.Embedding(num_month, config['embedding_dim_month'])

        input_dim = config["embedding_dim_stock"] + config['embedding_dim_group'] + config['embedding_dim_day']+ config['embedding_dim_month'] + feature_dim
        
        # self.tcn = TCN(input_size=input_dim, num_channels=config['tcn_chanel'], kernel_size=config['tcn_kernel'], dropout=0.2)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=config['cnn_chanel1'], kernel_size=1,padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(config['cnn_chanel1']),
            nn.Conv1d(in_channels=config['cnn_chanel1'], out_channels=config['cnn_chanel2'], kernel_size=1,padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(config['cnn_chanel2']),
            nn.MaxPool1d(kernel_size=1,stride=2)
        )

        self.bilstm = nn.LSTM(config['cnn_chanel2'], config['hidden_bilstm'], 1, batch_first=True, bidirectional=True)
        self.layernorm1 = nn.LayerNorm(config['hidden_bilstm']*2)

        self.lstm1 = nn.LSTM(config['hidden_bilstm']*2, config['first_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(config['first_layer_hidden_size']*2, config['second_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(config['second_layer_hidden_size']*2, config['third_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.layernorm2 = nn.LayerNorm(config['third_layer_hidden_size']*2)

        print("before attention")
        self.attention = AttentionLayer(config['third_layer_hidden_size'] * 2, config['attent_hidden_size'])
        print("after attention")

        self.dropout = nn.Dropout(config['dropout'])
        
        self.fc = nn.Linear(config['attent_hidden_size'], 1)
        self.softsign = nn.Softsign()

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)

        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        combind_input = combind_input.transpose(1, 2)
        # print(f"tcn: {combind_input.shape}")
        combind_input = self.cnn(combind_input)        # Output: (batch, tcn_channels[-1], seq_len)
        # print("tcn final")
        combind_input = combind_input.transpose(1, 2)
        
        # combind_input = combind_input.transpose(1, 2)
        combind_input, _ = self.bilstm(combind_input)
        combind_input = self.layernorm1(combind_input)
        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)
        lstm_out3 = self.layernorm2(lstm_out3)

        context_vector, _ = self.attention(lstm_out3)

        # Flatten context_vector and pass through fully connected layer
        context_vector = context_vector.squeeze(1)
        out1 = self.dropout(context_vector)

        fc_out = self.fc(out1)
        return fc_out
    
class LSTMModelxCNNxNORMWithMultiAttention(nn.Module):
    def __init__(self, 
                feature_dim,
                num_stocks,
                num_group,
                num_day,
                num_month,
                config):
        super(LSTMModelxCNNxNORMWithMultiAttention, self).__init__()

        config = config.LSTMxCNNxNORM_MULRIATTENTION_PARAMS
        self.stock_embedding = nn.Embedding(num_stocks, config["embedding_dim_stock"])
        self.group_embedding = nn.Embedding(num_group, config['embedding_dim_group'])
        self.day_embedding = nn.Embedding(num_day, config['embedding_dim_day'])
        self.month_embedding = nn.Embedding(num_month, config['embedding_dim_month'])

        input_dim = config["embedding_dim_stock"] + config['embedding_dim_group'] + config['embedding_dim_day']+ config['embedding_dim_month'] + feature_dim
        
        # self.tcn = TCN(input_size=input_dim, num_channels=config['tcn_chanel'], kernel_size=config['tcn_kernel'], dropout=0.2)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=config['cnn_chanel1'], kernel_size=1,padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(config['cnn_chanel1']),
            nn.Conv1d(in_channels=config['cnn_chanel1'], out_channels=config['cnn_chanel2'], kernel_size=1,padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(config['cnn_chanel2']),
            nn.MaxPool1d(kernel_size=1,stride=2)
        )

        self.bilstm = nn.LSTM(config['cnn_chanel2'], config['hidden_bilstm'], 1, batch_first=True, bidirectional=True)
        self.layernorm1 = nn.LayerNorm(config['hidden_bilstm']*2)

        self.lstm1 = nn.LSTM(config['hidden_bilstm']*2, config['first_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(config['first_layer_hidden_size']*2, config['second_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(config['second_layer_hidden_size']*2, config['third_layer_hidden_size'], 1, batch_first=True, bidirectional=True)
        self.layernorm2 = nn.LayerNorm(config['third_layer_hidden_size']*2)

        print("before attention")
        # self.attention = AttentionLayer(config['third_layer_hidden_size'] * 2, config['attent_hidden_size'])
        self.attention = nn.MultiheadAttention(embed_dim=config['third_layer_hidden_size'] * 2, num_heads=config['num_head_attention'], batch_first=True)
        print("after attention")

        self.dropout = nn.Dropout(config['dropout'])
        
        self.fc = nn.Linear(config['third_layer_hidden_size']*2, 1)
        self.softsign = nn.Softsign()

    def forward(self, stock_name, group_name, day_name, month_name, feature):
        stock_emb = self.stock_embedding(stock_name)
        group_emb = self.group_embedding(group_name)
        month_emb = self.month_embedding(month_name)
        day_emb = self.day_embedding(day_name)

        combind_input = torch.cat([stock_emb, group_emb,day_emb,month_emb, feature], dim=2)
        combind_input = combind_input.transpose(1, 2)
        # print(f"tcn: {combind_input.shape}")
        combind_input = self.cnn(combind_input)        # Output: (batch, tcn_channels[-1], seq_len)
        # print("tcn final")
        combind_input = combind_input.transpose(1, 2)
        
        # combind_input = combind_input.transpose(1, 2)
        combind_input, _ = self.bilstm(combind_input)
        combind_input = self.layernorm1(combind_input)
        out, _ = self.lstm1(combind_input)
        lstm_out21, _ = self.lstm2(out)
        lstm_out3, _ = self.lstm3(lstm_out21)
        lstm_out3 = self.layernorm2(lstm_out3)

        attn_out, _ = self.attention(lstm_out3,lstm_out3,lstm_out3)
        context_vector = attn_out.mean(dim=1)

        out1 = self.dropout(context_vector)

        fc_out = self.fc(out1)
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

    
def time_embeddings(time_steps, embedding_dim):
    # positions = torch.arange(1,time_steps).unsqueeze(1)
    len_timestep = len(time_steps)
    positions = torch.tensor(time_steps, dtype=torch.float32).unsqueeze(1)
    # print(positions)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
    pos_enc = torch.zeros(len_timestep, embedding_dim)
    pos_enc[:, 0::2] = torch.sin(positions * div_term)
    pos_enc[:, 1::2] = torch.cos(positions * div_term)
    return pos_enc

class CombinedTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(CombinedTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        # Inverse frequency approach
        self.inv_freq = 1 / (10000 ** (torch.arange(0, embedding_dim // 2, 2).float() / embedding_dim))

        # Exponential scaling approach
        self.div_term = torch.exp(torch.arange(0, embedding_dim // 2, 2) * (-torch.log(torch.tensor(10000.0)) / embedding_dim))

    def forward(self, time_steps):
        """
        time_steps: tensor of shape (batch_size, sequence_length)
        """
        time_steps = time_steps.unsqueeze(-1).float()  # Shape: (batch_size, sequence_length, 1)
        
        # Compute sinusoidal embeddings for both approaches
        sinusoids_inv = torch.sin(time_steps * self.inv_freq)  # Shape: (batch_size, seq_len, embedding_dim/2)
        cosines_inv = torch.cos(time_steps * self.inv_freq)

        sinusoids_exp = torch.sin(time_steps * self.div_term)
        cosines_exp = torch.cos(time_steps * self.div_term)

        # Concatenate both embeddings
        combined_embeddings = torch.cat([sinusoids_inv, cosines_inv, sinusoids_exp, cosines_exp], dim=-1)

        return combined_embeddings