# NumModel Task
import sys, os, pickle
sys.path.append('../utils/')
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from CustomDataset_NMT import CustomDataset
from CustomModel_NMT import CustomModel

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, learning_rate, device, num_epochs, patience):
    model.to(device)

    # Create the optimizer after moving the model to the target device
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(num_epochs):
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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs.view(-1)) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_true, val_preds)

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy")
                break

    return best_val_acc

# 加载数据
folder_labels = {
    "NC": 0,
    "Mild": 1,
    "Moderate": 1,
    "Severe": 1,
}
base_path = '/home/user/xuxiao/ABAFnet/audio_data/CNRAC'
df = pd.read_csv('/home/user/xuxiao/ABAFnet/CNRAC_features/emo_large_res.csv')
data_list = []
label_list = []
for folder, label in folder_labels.items():
    for root, dirs, files in os.walk(os.path.join(base_path, folder)):
        for file in files:
            data_list.append(df[df['name'] == file[:-6]].drop(columns=['name','class']).values.tolist()[0])
            label_list.append(label)

dataset = CustomDataset(data_list=data_list, label_list=label_list)

# Set up K-Fold cross-validation, parameter grid, and other configurations
k_folds = 5
num_epochs = 100
device = torch.device('cuda')

param_grid = {
    'criterion': [nn.BCEWithLogitsLoss()],
    'activation': [nn.Sigmoid(), nn.ELU()],
    'optimizer': [optim.SGD, optim.RMSprop],
    'learning_rate': [0.001],
}

# Perform grid search with cross-validation
kf = StratifiedKFold(n_splits=k_folds)
best_params = None
best_score = 0
param_counter = 1
for params in ParameterGrid(param_grid):
    print(f"Parameter set {param_counter}: {params}")
    fold_counter = 1
    acc_scores = []  # Store accuracy scores for each fold
    precision_scores = []  # Store precision scores for each fold
    recall_scores = []  # Store recall scores for each fold
    f1_scores = []  # Store F1 scores for each fold
    roc_scores = []  # Store ROC AUC scores for each fold
    fold_roc_curve = []
    total_confusion_matrix = None
    for train_idx, val_idx in kf.split(data_list, label_list):
        print(f"Fold {fold_counter}")
        feature_size = dataset.feature_size()
        model = CustomModel(num_heads=4, input_size=feature_size)
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

        acc_scores.append(val_acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        roc_scores.append(roc)
        fold_roc_curve.append((val_true, val_probs))

        # Calculate confusion matrix
        confusion_mat = confusion_matrix(val_true, val_preds)
        if total_confusion_matrix is None:
            total_confusion_matrix = confusion_mat
        else:
            total_confusion_matrix += confusion_mat

        fold_counter += 1
    param_counter += 1

    with open("/home/user/xuxiao/ABAFnet/draw/roc_curve_data_emolarge.pkl", "wb") as f:
        pickle.dump(fold_roc_curve, f)

    avg_accuracy = np.mean(acc_scores)
    std_accuracy = np.std(acc_scores)
    avg_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    avg_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_roc = np.mean(roc_scores)
    std_roc = np.std(roc_scores)

    # print(f"Average accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    # print(f"Average precision: {avg_precision:.3f} ± {std_precision:.3f}")
    # print(f"Average recall: {avg_recall:.3f} ± {std_recall:.3f}")
    # print(f"Average F1 score: {avg_f1:.3f} ± {std_f1:.3f}")
    # print(f"Average ROC: {avg_roc:.3f} ± {std_roc:.3f}")

    if avg_f1 > best_score:
        best_score = avg_f1
        best_params = params
        best_std_acc = std_accuracy
        best_std_precision = std_precision
        best_std_recall = std_recall
        best_std_f1 = std_f1
        best_std_roc = std_roc
print("Best parameters:", best_params)
print(f"Best accuracy: {avg_accuracy:.3f} ± {best_std_acc:.3f}")
print(f"Best precision: {avg_precision:.3f} ± {best_std_precision:.3f}")
print(f"Best recall: {avg_recall:.3f} ± {best_std_recall:.3f}")
print(f"Best F1 score: {best_score:.3f} ± {best_std_f1:.3f}")
print(f"Best ROC: {avg_roc:.3f} ± {best_std_roc:.3f}")
print("Total confusion matrix:\n", total_confusion_matrix)
