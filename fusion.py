import numpy as np
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

class MyAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MyAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x, _ = self.multihead_attention(x, x, x)
        return x

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

class NumModel(nn.Module):
    def __init__(self, input_size_num):
        super().__init__()
        self.fc_num = nn.Linear(input_size_num, 128)

    def forward(self, num_features):
        num_features = self.fc_num(num_features)
        return num_features

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

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, learning_rate, device, num_epochs, patience):
    model.to(device)

    # Create the optimizer after moving the model to the target device
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    epochs_no_improve = 0
    best_model = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # Training
        model.train()
        for (img1, img2, img3, num_features, labels) in train_loader:
            img1, img2, img3, num_features, labels = img1.to(device), img2.to(device), img3.to(device), num_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2, img3, num_features)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

        # Validation
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

    return best_val_acc, best_model

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load your data from file

# Split data into features (data_list) and labels (label_list)
data_list_num = data.drop(columns=['group','standard_id']).values.tolist()
label_list = [1 if x != 0 else x for x in data['group'].values.tolist()]

dataset = CustomDataset(data_list_img1=data_list_img1,data_list_img2=data_list_img2,data_list_img3=data_list_img3,data_list_num=data_list_num, label_list=label_list, transform=data_transform)

# Set up K-Fold cross-validation, parameter grid, and other configurations
k_folds = 5
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_grid = {
    'criterion': [nn.BCEWithLogitsLoss()],
    'activation': [nn.ReLU()],
    'optimizer': [optim.SGD],
    'learning_rate': [0.001],
}

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

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    colors = sns.color_palette("Set2")
    fold_counter = 1
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    combined_data_list = list(zip(data_list_img1, data_list_img2, data_list_img3, data_list_num))

    for train_idx, val_idx in kf.split(combined_data_list, label_list):
        print(f"Fold {fold_counter}")
        input_size_img1, input_size_img2, input_size_img3, input_size_num = dataset.feature_size()
        num_heads = 8  # 可以设置为其他值
        img_model = ImageModel(params['activation'])
        num_model = NumModel(input_size_num)
        model = FusionModel(img_model, num_model, num_heads=num_heads)

        train_set = Subset(dataset, train_idx)
        val_set= Subset(dataset, val_idx)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        criterion = params['criterion']
        optimizer_class = params['optimizer']
        val_acc, best_fold_model = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, params['learning_rate'], device, num_epochs, patience=20)

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
        avg_accuracy += val_acc
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
        avg_roc += roc

        fpr, tpr, _ = roc_curve(val_true, val_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=1, color=colors[fold_counter - 1], linestyle='--', label=f'Fold {fold_counter} (AUC = {roc_auc:.2f})')
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

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
        best_model = best_fold_model

    mean_tpr /= k_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=colors[6], lw=2.5, label=f'Mean (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color=colors[7])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontname="Arial", fontsize=13)
    plt.ylabel('True Positive Rate', fontname="Arial", fontsize=13)
    plt.title('ROC Curve', fontname="Arial", fontsize=14)
    plt.legend(loc="lower right", prop={'family': 'Arial', 'size': 11})
    plt.tight_layout()
    plt.savefig('ROC_curvex.png', dpi=300)

print("Best parameters:", best_params)
print("Best accuracy:", avg_accuracy)
print("Best precision:", avg_precision)
print("Best recall:", avg_recall)
print("Best F1 score:", best_score)
print("Average ROC:", avg_roc)
print("Total confusion matrix:\n", total_confusion_matrix)
torch.save(best_model.state_dict(), 'best_model.pth')
