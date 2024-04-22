# Validation Task
import sys, os
from random import sample
sys.path.append('../utils/')
import numpy as np
import time
import pandas as pd
import random, copy
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from CustomDataset import CustomDataset
from MyAttention import MyAttention
from ImageModel import ImageModel
from NumModel import NumModel
from FusionModel import FusionModel


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, learning_rate, device, num_epochs, patience):
    start_time = time.time()  # 开始时间
    model.to(device)

    # Create the optimizer after moving the model to the target device
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        train_start_time = time.time()  # 训练开始时间
        model.train()
        for (img1, img2, img3, num_features, labels) in train_loader:
            img1, img2, img3, num_features, labels = img1.to(device), img2.to(device), img3.to(device), num_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2, img3, num_features)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()
        train_end_time = time.time()  # 训练结束时间

        # Validation
        val_start_time = time.time()  # 验证开始时间
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for (img1, img2, img3, num_features, labels) in val_loader:
                img1, img2, img3, num_features, labels = img1.to(device), img2.to(device), img3.to(device), num_features.to(device), labels.to(device)
                outputs = model(img1, img2, img3, num_features)
                preds = (torch.sigmoid(outputs.view(-1)) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        val_end_time = time.time()  # 验证结束时间
        val_acc = accuracy_score(val_true, val_preds)

        print(f"Validation data size: {len(val_true)}")  # 添加这行代码来检查验证数据的数量

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
        else:
            epochs_no_improve += 1
            # if epochs_no_improve == patience:
            #     print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy")
            #     break

    train_time = train_end_time - train_start_time  # 训练总时间
    val_time = val_end_time - val_start_time  # 验证总时间
    print(f"Total training time: {train_time} seconds.")
    print(f"Total validation time: {val_time} seconds.")

    return best_val_acc, best_model, train_time, val_time

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load data
score = 10
random_state = 0
random.seed(random_state)
print(score)
df = pd.read_csv('/home/user/xuxiao/ABAFnet/audio_data/CS-NRAC/list.csv')
group1 = df[df['PHQ-9_score'] < 1]['cust_id'].values.tolist()
group2 = df[df['PHQ-9_score'] >= score]['cust_id'].values.tolist()
group1 = sample(group1, len(group2))

feature_path = '/home/user/xuxiao/ABAFnet/features/CS-NRAC_features'
feature = pd.read_csv('/home/user/xuxiao/ABAFnet/features/CS-NRAC_features/emo_large.csv')

data_list_img1, data_list_img2, data_list_img3, data_list_num = [], [], [], []
label_list = []

for item in group1:
    if os.path.exists(os.path.join(feature_path, str(item))):
        data_list_img1.append(os.path.join(feature_path, str(item), 'envelope.png'))
        data_list_img2.append(os.path.join(feature_path, str(item), 'spectrogram.png'))
        data_list_img3.append(os.path.join(feature_path, str(item), 'mel_spectrogram.png'))
        data_list_num.append(feature[feature['name'] == item].drop(columns=['name','class']).values.tolist()[0])
        label_list.append(0)
print(len(data_list_img1))
for item in group2:
    if os.path.exists(os.path.join(feature_path, str(item))):
        data_list_img1.append(os.path.join(feature_path, str(item), 'envelope.png'))
        data_list_img2.append(os.path.join(feature_path, str(item), 'spectrogram.png'))
        data_list_img3.append(os.path.join(feature_path, str(item), 'mel_spectrogram.png'))
        data_list_num.append(feature[feature['name'] == item].drop(columns=['name','class']).values.tolist()[0])
        label_list.append(1)
print(len(data_list_img1))
dataset = CustomDataset(data_list_img1=data_list_img1,data_list_img2=data_list_img2,data_list_img3=data_list_img3,data_list_num=data_list_num, label_list=label_list, transform=data_transform)

# Configurations
num_epochs = 40
device = torch.device('cuda')

params = {
    'criterion': nn.BCEWithLogitsLoss(),
    'activation': nn.ReLU(),
    'optimizer': optim.SGD,
    'learning_rate': 0.001,
}

combined_data_list = list(zip(data_list_img1, data_list_img2, data_list_img3, data_list_num))
train_idx, val_idx = train_test_split(range(len(combined_data_list)), test_size=0.3, stratify=label_list, random_state=random_state)

input_size_img1, input_size_img2, input_size_img3, input_size_num = dataset.feature_size()
num_heads = 8
img_model = ImageModel(params['activation'])
num_model = NumModel(input_size_num)
model = FusionModel(img_model, num_model, num_heads=num_heads)

train_set = Subset(dataset, train_idx)
val_set= Subset(dataset, val_idx)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

criterion = params['criterion']
optimizer_class = params['optimizer']

val_acc, best_model, train_time, val_time = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, params['learning_rate'], device, num_epochs, patience=7)

val_true = np.concatenate([labels.numpy() for (_, _, _, _, labels) in val_loader], axis=0)
val_preds = []
val_probs = []
with torch.no_grad():
    for (img1, img2, img3, num_features, _) in val_loader:
        img1, img2, img3, num_features = img1.to(device), img2.to(device), img3.to(device), num_features.to(device)
        outputs = best_model(img1, img2, img3, num_features)  # 使用best_model
        preds = (torch.sigmoid(outputs.view(-1)) > 0.5).cpu().numpy()
        probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
        val_preds.extend(preds)
        val_probs.extend(probs)
val_preds = np.array(val_preds)
val_probs = np.array(val_probs)

precision = precision_score(val_true, val_preds)
recall = recall_score(val_true, val_preds)
f1 = f1_score(val_true, val_preds)
roc = roc_auc_score(val_true, val_probs)

# Calculate confusion matrix
total_confusion_matrix = confusion_matrix(val_true, val_preds)

print(f"Training time: {train_time} seconds.")
print(f"Testing time: {val_time} seconds.")
print(f"Accuracy: {val_acc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
print(f"ROC AUC: {roc:.3f}")
print("Total confusion matrix:\n", total_confusion_matrix)
