import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

class CustomDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = torch.tensor(self.data_list[index], dtype=torch.float32)
        label = self.label_list[index]
        return data, label
    
    def feature_size(self):
        return len(self.data_list[0])
