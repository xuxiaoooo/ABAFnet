import sys, os
sys.path.append('../utils/')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from CustomDataset import CustomDataset
from ImageModel import ImageModel
from NumModel import NumModel
from FusionModel import FusionModel

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据
folder_labels = {
    "NC": 0,
    "Mild": 1,
    "Moderate": 1,
    "Severe": 1,
}
base_path = '/home/user/xuxiao/ABAFnet/audio_data/CNRAC'
feature_path = '/home/user/xuxiao/ABAFnet/features/CNRAC_features'
df = pd.read_csv('/home/user/xuxiao/ABAFnet/features/CNRAC_features/emo_large_res.csv')
data_list_img1, data_list_img2, data_list_img3, data_list_num = [], [], [], []
label_list = []
for folder, label in folder_labels.items():
    for root, dirs, files in os.walk(os.path.join(base_path, folder)):
        for file in files:
            data_list_img1.append(os.path.join(feature_path, file[:-6], 'envelope.png'))
            data_list_img2.append(os.path.join(feature_path, file[:-6], 'spectrogram.png'))
            data_list_img3.append(os.path.join(feature_path, file[:-6], 'mel_spectrogram.png'))
            data_list_num.append(df[df['name'] == file[:-6]].drop(columns=['name','class']).values.tolist()[0])
            label_list.append(label)

dataset = CustomDataset(data_list_img1=data_list_img1,data_list_img2=data_list_img2,data_list_img3=data_list_img3,data_list_num=data_list_num, label_list=label_list, transform=data_transform)

# 加载最佳模型
input_size_img1, input_size_img2, input_size_img3, input_size_num = dataset.feature_size()
num_heads = 8
img_model = ImageModel(nn.ReLU())  
num_model = NumModel(input_size_num)
model = FusionModel(img_model, num_model, num_heads=num_heads)
model.load_state_dict(torch.load('/home/user/xuxiao/ABAFnet/model/best_model.pth'), strict=False)
model.eval()

device = torch.device('cuda')
model.to(device)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

ncdf = pd.read_excel('/home/user/xuxiao/ABAFnet/audio_data/CNRAC/CNRAC_list.xlsx', sheet_name='健康组')
dedf = pd.read_excel('/home/user/xuxiao/ABAFnet/audio_data/CNRAC/CNRAC_list.xlsx', sheet_name='病人组')

# 进行预测并打印结果
with torch.no_grad():
    for i, (img1, img2, img3, num_features, labels) in enumerate(dataloader):
        img1, img2, img3, num_features, labels = img1.to(device), img2.to(device), img3.to(device), num_features.to(device), labels.to(device)
        outputs = model(img1, img2, img3, num_features)
        preds = (torch.sigmoid(outputs.view(-1)) > 0.3).cpu().numpy()
        
        file_name = df.iloc[i]['name']
        correct = (preds[0] == labels.cpu().numpy()[0])
        if correct == False:
            if file_name in ncdf['standard_id'].values:
                print(f"File: {file_name}, NC, {ncdf[ncdf['standard_id'] == file_name]['HAMD17_total_score'].values[0]}")
            elif file_name in dedf['standard_id'].values:
                print(f"File: {file_name}, DE, {dedf[dedf['standard_id'] == file_name]['HAMD17_total_score'].values[0]}")
