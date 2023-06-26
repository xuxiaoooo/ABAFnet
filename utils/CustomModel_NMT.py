import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from MyAttention import MyAttention

class CustomModel(nn.Module):
    def __init__(self, num_heads, input_size):
        self.input_size = input_size
        super(CustomModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.my_attention_1 = MyAttention(embed_dim=64, num_heads=num_heads)
        self.dense_1 = nn.Linear(64, 32)
        self.batch_normalization_1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(32, 16)
        self.batch_normalization_2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.5)
        self.dense_3 = nn.Linear(16, 1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.input_size)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.transpose(0, 1)
        x = self.my_attention_1(x)
        x = x.transpose(0, 1)
        x = x[:, -1, :]
        x = self.dense_1(x)
        x = self.batch_normalization_1(x)
        x = self.dropout1(x)
        x = self.dense_2(x)
        x = self.batch_normalization_2(x)
        x = self.dropout2(x)
        x = self.dense_3(x)
        return x
