import torch
import torch.nn as nn

class NumModel(nn.Module):
    def __init__(self, input_size_num):
        super().__init__()
        self.fc_num = nn.Linear(input_size_num, 128)

    def forward(self, num_features):
        num_features = self.fc_num(num_features)
        return num_features