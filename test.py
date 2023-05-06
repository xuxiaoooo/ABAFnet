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


class CustomDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img) if self.transform else img
        label = self.label_list[index]
        return img_transformed, label

class MyAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MyAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x, _ = self.multihead_attention(x, x, x)
        return x

class CustomModel(nn.Module):
    def __init__(self, num_heads):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lstm1 = nn.LSTM(input_size=56 * 56 * 32, hidden_size=128, num_layers=1, batch_first=True)
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
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 32 * 56 * 56)
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
            # torch.cuda.empty_cache()

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

file_list = []
label_list = []
hdf1 = pd.read_excel('HAMD（13-24）-155.xlsx', sheet_name='病人组', engine='openpyxl')
hdf2 = pd.read_excel('HAMD（13-24）-155.xlsx', sheet_name='健康组', engine='openpyxl')
hdf = pd.concat([hdf1, hdf2], axis=0).reset_index(drop=True)[['group', 'standard_id']]
for i in range(len(hdf)):
    file_list.append('/home/user/xuxiao/DeepL/image-features/' + hdf['standard_id'][i] + '/mel_spectrogram.png')
    label_list.append(0 if hdf['group'][i] == 0 else 1)
print('mel_spectrogram')
all = list(zip(file_list, label_list))
random.shuffle(all)
file_list[:], label_list[:] = zip(*all)

label_list = np.array(label_list)

dataset = CustomDataset(file_list=file_list, label_list=label_list, transform=data_transform)

# Set up K-Fold cross-validation, parameter grid, and other configurations
k_folds = 5
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_grid = {
    'criterion': [nn.BCEWithLogitsLoss()],
    'activation': [nn.ReLU(), nn.Sigmoid(), nn.ELU(), nn.LeakyReLU()],
    'optimizer': [optim.SGD],
    'learning_rate': [0.001, 0.01],
}

# Perform grid search with cross-validation
kf = KFold(n_splits=k_folds)
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
    avg_micro_roc = 0
    avg_macro_roc = 0
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

        micro_roc = roc_auc_score(val_true, val_probs, average='micro')
        macro_roc = roc_auc_score(val_true, val_probs, average='macro')

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
        avg_micro_roc += micro_roc
        avg_macro_roc += macro_roc

        fold_counter += 1
    param_counter += 1

    avg_accuracy /= k_folds
    avg_precision /= k_folds
    avg_recall /= k_folds
    avg_f1 /= k_folds
    avg_micro_roc /= k_folds
    avg_macro_roc /= k_folds

    if avg_f1 > best_score:
        best_score = avg_f1
        best_params = params
print("Best parameters:", best_params)
print("Best accuracy:", avg_accuracy)
print("Best precision:", avg_precision)
print("Best recall:", avg_recall)
print("Best F1 score:", best_score)
print("Average micro ROC:", avg_micro_roc)
print("Average macro ROC:", avg_macro_roc)
print("Total confusion matrix:\n", total_confusion_matrix)