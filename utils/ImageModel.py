import torch
import torch.nn as nn

class ImageModel(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(56 * 56 * 32, 128)

    def forward(self, img):
        img = self.pool(self.activation(self.conv1(img)))
        img = self.pool(self.activation(self.conv2(img)))
        img = torch.flatten(img, start_dim=1)
        img = self.fc(img)
        return img
