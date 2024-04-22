import sys, os
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
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
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

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy")
                break

    train_time = train_end_time - train_start_time  # 训练总时间
    val_time = val_end_time - val_start_time  # 验证总时间
    print(f"Total training time: {train_time} seconds.")
    print(f"Total validation time: {val_time} seconds.")

    return best_val_acc, best_model, train_time, val_time


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
files1 = pd.read_excel('/home/user/xuxiao/ABAFnet/audio_data/CNRAC/CNRAC_list.xlsx', sheet_name='病人组')
files1 = files1[(files1['age'] >= 13) & (files1['age'] <= 25)]['standard_id']
files2 = pd.read_excel('/home/user/xuxiao/ABAFnet/audio_data/CNRAC/CNRAC_list.xlsx', sheet_name='健康组')
files2 = files2[(files2['age'] >= 13) & (files2['age'] <= 25)]['standard_id']
file_filter = pd.concat([files1, files2])
label_list = []
for folder, label in folder_labels.items():
    for root, dirs, files in os.walk(os.path.join(base_path, folder)):
        for file in files:
            if file[:-6] in file_filter.values:
                data_list_img1.append(os.path.join(feature_path, file[:-6], 'envelope.png'))
                data_list_img2.append(os.path.join(feature_path, file[:-6], 'spectrogram.png'))
                data_list_img3.append(os.path.join(feature_path, file[:-6], 'mel_spectrogram.png'))
                data_list_num.append(df[df['name'] == file[:-6]].drop(columns=['name','class']).values.tolist()[0])
                label_list.append(label)
print(len(data_list_img1), len(data_list_img2), len(data_list_img3), len(data_list_num), len(label_list), label_list.count(0), label_list.count(1))
dataset = CustomDataset(data_list_img1=data_list_img1,data_list_img2=data_list_img2,data_list_img3=data_list_img3,data_list_num=data_list_num, label_list=label_list, transform=data_transform)

# Set up K-Fold cross-validation, parameter grid, and other configurations
k_folds = 4
num_epochs = 100
device = torch.device('cuda')

param_grid = {
    'criterion': [nn.BCEWithLogitsLoss()],
    'activation': [nn.ReLU()],
    'optimizer': [optim.SGD],
    'learning_rate': [0.00001],
}

kf = StratifiedKFold(n_splits=k_folds)
best_params = None
best_score = 0
param_counter = 1

total_train_time = 0
total_test_time = 0

# 初始化结果列表
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
roc_list = []
roc_curves = []

for params in ParameterGrid(param_grid):
    print(f"Parameter set {param_counter}: {params}")
    fold_counter = 1
    avg_accuracy = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    avg_roc = 0
    total_confusion_matrix = None

    # 在每一个参数设置开始前，清空列表
    accuracy_list.clear()
    precision_list.clear()
    recall_list.clear()
    f1_list.clear()
    roc_list.clear()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    combined_data_list = list(zip(data_list_img1, data_list_img2, data_list_img3, data_list_num))

    for train_idx, val_idx in kf.split(combined_data_list, label_list):
        print(f"Fold {fold_counter}")
        print(f"Train: {len(train_idx)}, Validation: {len(val_idx)}")
        input_size_img1, input_size_img2, input_size_img3, input_size_num = dataset.feature_size()
        num_heads = 8  # 可以设置为其他值
        img_model = ImageModel(params['activation'])
        num_model = NumModel(input_size_num)
        model = FusionModel(img_model, num_model, num_heads=num_heads)

        train_set = Subset(dataset, train_idx)
        val_set= Subset(dataset, val_idx)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
        criterion = params['criterion']
        optimizer_class = params['optimizer']
        val_acc, best_fold_model, train_time, val_time = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, params['learning_rate'], device, num_epochs, patience=10)
        total_train_time += train_time
        total_test_time += val_time

        val_true = np.concatenate([labels.numpy() for (_, _, _, _, labels) in val_loader], axis=0)
        val_preds = []
        val_probs = []
        with torch.no_grad():
            for (img1, img2, img3, num_features, _) in val_loader:
                img1, img2, img3, num_features = img1.to(device), img2.to(device), img3.to(device), num_features.to(device)
                outputs = model(img1, img2, img3, num_features)
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
        confusion_mat = confusion_matrix(val_true, val_preds)
        if total_confusion_matrix is None:
            total_confusion_matrix = confusion_mat
        else:
            total_confusion_matrix += confusion_mat

        # 存储每一个折叠的结果
        accuracy_list.append(val_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        roc_list.append(roc)

        avg_accuracy += val_acc
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
        avg_roc += roc

        fpr, tpr, _ = roc_curve(val_true, val_probs)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        roc_curves.append((fpr, tpr))

        fold_counter += 1

    mean_tpr /= k_folds
    mean_tpr[-1] = 1.0

    roc_curves.append((mean_fpr, mean_tpr))

    param_counter += 1

    avg_accuracy /= k_folds
    avg_precision /= k_folds
    avg_recall /= k_folds
    avg_f1 /= k_folds
    avg_roc /= k_folds

    std_accuracy = np.std(accuracy_list)
    std_precision = np.std(precision_list)
    std_recall = np.std(recall_list)
    std_f1 = np.std(f1_list)
    std_roc = np.std(roc_list)

    if avg_f1 > best_score:
        best_score = avg_f1
        best_params = params
        best_model = best_fold_model

print(f"Total training time for all folds: {total_train_time} seconds.")
print(f"Total testing time for all folds: {total_test_time} seconds.")
print("Best parameters:", best_params)
print(f"Accuracy: {avg_accuracy:.3f}±{std_accuracy:.2f}")
print(f"Precision: {avg_precision:.3f}±{std_precision:.2f}")
print(f"Recall: {avg_recall:.3f}±{std_recall:.2f}")
print(f"F1: {avg_f1:.3f}±{std_f1:.2f}")
print(f"ROC AUC: {avg_roc:.3f}±{std_roc:.2f}")
print("Total confusion matrix:\n", total_confusion_matrix)
