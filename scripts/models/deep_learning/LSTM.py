import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """
    Sequential Neural Network using LSTM layers

    Arguments
    ---------
        batch_size: int
            Size of the batch
        hidden_dim: int
            Dimension of the hidden dimensions of the model 
        lstm_layers : 
            Number of LSTM layer
        max_padding : int
            Input size

    """

    def __init__(self, batch_size, hidden_dim, lstm_layers, max_padding):
        super(LSTMClassifier, self).__init__()
        
        # Hyperparameters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = max_padding
        
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim*2)
        self.fc2 = nn.Linear(self.hidden_dim*2, 18)

    def forward(self, x):
        
        # Hidden and cell state definion
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        
        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        # Each sequence "x" is passed through an embedding layer
        out = self.embedding(x)
        # Feed LSTMs
        out, (hidden, cell) = self.lstm(out, (h,c))
        out = self.dropout(out)
        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))

        return out
