import torch
import torch.nn as nn
from PIL import Image
from MyAttention import MyAttention
from ImageModel import ImageModel
from NumModel import NumModel

class FusionModel(nn.Module):
    def __init__(self, img_model, num_model, lstm_input_size=256, hidden_size=256, num_heads=4, weight_img1=0, weight_img2=0.5, weight_img3=0.5, weight_num=0):
        super().__init__()
        self.img_model = img_model
        self.num_model = num_model

        self.embedding_adjustment = nn.Linear(128, lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.num_features_adjustment = nn.Linear(128, lstm_input_size)

        self.my_attention = MyAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.dense_1 = nn.Linear(hidden_size, 128)
        self.batch_normalization_1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(128, 64)
        self.batch_normalization_2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        self.dense_3 = nn.Linear(64, 1)

        self.weight_img1 = weight_img1
        self.weight_img2 = weight_img2
        self.weight_img3 = weight_img3
        self.weight_num = weight_num

    def forward(self, img1, img2, img3, num_features):
        img1 = self.embedding_adjustment(self.img_model(img1)) * self.weight_img1
        img2 = self.embedding_adjustment(self.img_model(img2)) * self.weight_img2
        img3 = self.embedding_adjustment(self.img_model(img3)) * self.weight_img3
        num_features = self.num_features_adjustment(self.num_model(num_features)) * self.weight_num

        # Calculate weighted sum of image embeddings and numeric features
        combined = img1 + img2 + img3 + num_features

        # Normalize the combined features
        combined = combined / (self.weight_img1 + self.weight_img2 + self.weight_img3 + self.weight_num)

        batch_size = combined.size(0)
        combined = combined.view(batch_size, 1, -1)
        x, _ = self.lstm(combined)

        x = x.transpose(0, 1)
        x = self.my_attention(x)
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
