# ImageModel Task
import sys, os
sys.path.append('../utils/')
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from CustomDataset_IMGT import CustomDataset
from CustomModel_IMGT import CustomModel
import pickle

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, learning_rate, device, num_epochs, patience):
    model.to(device)

    # Create the optimizer after moving the model to the target device
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    best_val_loss = float("inf")
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        if no_improvement_epochs >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

        print(f"Epoch {epoch + 1}/{num_epochs}")
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels.float())
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs.view(-1)) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
            # torch.cuda.empty_cache()
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

    return best_val_acc

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
file_list = []
label_list = []
for folder, label in folder_labels.items():
    for root, dirs, files in os.walk(os.path.join(base_path, folder)):
        for file in files:
            file_list.append(os.path.join(feature_path, file[:-6], 'mel_spectrogram.png'))
            label_list.append(label)

all = list(zip(file_list, label_list))
random.shuffle(all)
file_list[:], label_list[:] = zip(*all)

label_list = np.array(label_list)

dataset = CustomDataset(file_list=file_list, label_list=label_list, transform=data_transform)

# Set up K-Fold cross-validation, parameter grid, and other configurations
k_folds = 5
num_epochs = 100
device = torch.device('cuda')

param_grid = {
    'criterion': [nn.BCEWithLogitsLoss()],
    'activation': [nn.ReLU(), nn.Sigmoid(), nn.ELU()],
    'optimizer': [optim.SGD],
    'learning_rate': [0.001],
}

# Perform grid search with cross-validation
kf = KFold(n_splits=k_folds)
best_params = None
best_score = 0
param_counter = 1

# Lists to store metrics per parameter set
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
roc_list = []

for params in ParameterGrid(param_grid):
    print(f"Parameter set {param_counter}: {params}")
    fold_counter = 1

    # Lists to store metrics per fold
    fold_accuracy_list = []
    fold_precision_list = []
    fold_recall_list = []
    fold_f1_list = []
    fold_roc_list = []
    fold_roc_curve = []
    total_confusion_matrix = None

    for train_idx, val_idx in kf.split(dataset):
        print(f"Fold {fold_counter}")
        model = CustomModel(num_heads=4)
        model.activation1 = params['activation']
        model.activation2 = params['activation']

        train_set = Subset(dataset, train_idx)
        val_set= Subset(dataset, val_idx)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        criterion = params['criterion']
        optimizer_class = params['optimizer']
        val_acc = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, params['learning_rate'], device, num_epochs, patience=10)

        val_true = np.concatenate([labels.numpy() for _, labels in val_loader], axis=0)
        val_preds = []
        val_probs = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
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

        # Save fold metrics
        fold_accuracy_list.append(val_acc)
        fold_precision_list.append(precision)
        fold_recall_list.append(recall)
        fold_f1_list.append(f1)
        fold_roc_list.append(roc)
        fold_roc_curve.append((val_true, val_probs))

        fold_counter += 1
    param_counter += 1

    with open("/home/user/xuxiao/ABAFnet/draw/roc_curve_data_melspectrogram.pkl", "wb") as f:
        pickle.dump(fold_roc_curve, f)

    # Compute average and standard deviation for metrics across folds
    avg_accuracy = np.mean(fold_accuracy_list)
    std_accuracy = np.std(fold_accuracy_list)
    avg_precision = np.mean(fold_precision_list)
    std_precision = np.std(fold_precision_list)
    avg_recall = np.mean(fold_recall_list)
    std_recall = np.std(fold_recall_list)
    avg_f1 = np.mean(fold_f1_list)
    std_f1 = np.std(fold_f1_list)
    avg_roc = np.mean(fold_roc_list)
    std_roc = np.std(fold_roc_list)

    # Save parameter set metrics
    accuracy_list.append((avg_accuracy, std_accuracy))
    precision_list.append((avg_precision, std_precision))
    recall_list.append((avg_recall, std_recall))
    f1_list.append((avg_f1, std_f1))
    roc_list.append((avg_roc, std_roc))

    if avg_f1 > best_score:
        best_score = avg_f1
        best_params = params

# Get metrics of the best parameters
best_accuracy = accuracy_list[np.argmax(best_score)]
best_precision = precision_list[np.argmax(best_score)]
best_recall = recall_list[np.argmax(best_score)]
best_f1 = f1_list[np.argmax(best_score)]
best_roc = roc_list[np.argmax(best_score)]

print("Best parameters:", best_params)
print("Best accuracy: %.3f±%.3f" % best_accuracy)
print("Best precision: %.3f±%.3f" % best_precision)
print("Best recall: %.3f±%.3f" % best_recall)
print("Best F1 score: %.3f±%.3f" % best_f1)
print("Best ROC: %.3f±%.3f" % best_roc)
print("Total confusion matrix:\n", total_confusion_matrix)
