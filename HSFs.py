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

class MyAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MyAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x, _ = self.multihead_attention(x, x, x)
        return x

class CustomModel(nn.Module):
    def __init__(self, num_heads, input_size):
        self.input_size = input_size
        super(CustomModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.my_attention_1 = MyAttention(embed_dim=64, num_heads=num_heads)
        self.dense_1 = nn.Linear(64, 32)
        self.batch_normalization_1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(32, 16)
        self.batch_normalization_2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.5)
        self.dense_3 = nn.Linear(16, 1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.input_size)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.transpose(0, 1)
        x = self.my_attention_1(x)
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


# Load your data from file
hdf1 = class0 subjects
hdf2 = class1 subjects
hdf = pd.concat([hdf1, hdf2], axis=0).reset_index(drop=True)[['class', 'id']]
features = pd.read_csv('emo_large_res.csv')
header = pd.read_csv('reduced_data.csv').columns
data = pd.merge(hdf,features,left_on='id',right_on='id')[header.append(pd.Index(['class']))]

# Split data into features (data_list) and labels (label_list)
data_list = data.drop(columns=['class']).values.tolist()
label_list = [1 if x != 0 else x for x in data['class'].values.tolist()]

dataset = CustomDataset(data_list=data_list, label_list=label_list)

# Set up K-Fold cross-validation, parameter grid, and other configurations
k_folds = 5
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_grid = {
    'criterion': [nn.BCEWithLogitsLoss()],
    'activation': [nn.ReLU(), nn.Sigmoid(), nn.ELU(), nn.LeakyReLU()],
    'optimizer': [optim.Adam, optim.SGD, optim.RMSprop],
    'learning_rate': [0.001, 0.01],
}

# Continue with the rest of your code for performing grid search with cross-validation.
# Perform grid search with cross-validation
kf = StratifiedKFold(n_splits=k_folds)
best_params = None
best_score = 0
param_counter = 1
for params in ParameterGrid(param_grid):
    print(f"Parameter set {param_counter}: {params}")
    fold_counter = 1
    avg_accuracy = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    avg_roc = 0
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

        # Calculate confusion matrix
        confusion_mat = confusion_matrix(val_true, val_preds)
        if total_confusion_matrix is None:
            total_confusion_matrix = confusion_mat
        else:
            total_confusion_matrix += confusion_mat
        avg_accuracy += val_acc
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
        avg_roc += roc

        fold_counter += 1
    param_counter += 1

    avg_accuracy /= k_folds
    avg_precision /= k_folds
    avg_recall /= k_folds
    avg_f1 /= k_folds
    avg_roc /= k_folds

    if avg_f1 > best_score:
        best_score = avg_f1
        best_params = params
print("Best parameters:", best_params)
print("Best accuracy:", avg_accuracy)
print("Best precision:", avg_precision)
print("Best recall:", avg_recall)
print("Best F1 score:", best_score)
print("Average ROC:", avg_roc)
print("Total confusion matrix:\n", total_confusion_matrix)
