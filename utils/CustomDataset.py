import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

class CustomDataset(Dataset):
    def __init__(self, data_list_img1, data_list_img2, data_list_img3, data_list_num, label_list, transform=None):
        self.data_list_img1 = data_list_img1
        self.data_list_img2 = data_list_img2
        self.data_list_img3 = data_list_img3
        self.data_list_num = data_list_num
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list_img1)

    def __getitem__(self, index):
        data_img1 = Image.open(self.data_list_img1[index]).convert('RGB')
        data_img2 = Image.open(self.data_list_img2[index]).convert('RGB')
        data_img3 = Image.open(self.data_list_img3[index]).convert('RGB')

        if self.transform:
            data_img1 = self.transform(data_img1)
            data_img2 = self.transform(data_img2)
            data_img3 = self.transform(data_img3)

        data_num = torch.tensor(self.data_list_num[index], dtype=torch.float32)
        label = self.label_list[index]
        return data_img1, data_img2, data_img3, data_num, label

    def feature_size(self):
        return len(self.data_list_img1[0]), len(self.data_list_img2[0]), len(self.data_list_img3[0]), len(self.data_list_num[0])
