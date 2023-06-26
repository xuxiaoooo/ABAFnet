# Subtype Classification Task
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
from sklearn.utils import resample
from CustomDataset import CustomDataset
from MyAttention import MyAttention
from ImageModel import ImageModel
from NumModel import NumModel
from FusionModel import FusionModel

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, learning_rate, device, num_epochs, patience):
    start_time = time.time()
    model.to(device)

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        for (img1, img2, img3, num_features, labels) in train_loader:
            img1, img2, img3, num_features, labels = img1.to(device), img2.to(device), img3.to(device), num_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2, img3, num_features)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy")
                break

    train_time = time.time() - start_time
    return best_val_acc, best_model, train_time

def run_experiment(data, labels, experiment_name, num_resamples=10):
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_rocs = []

    for i in range(num_resamples):
        data_resampled, labels_resampled = resample(data, labels, n_samples=len(data), random_state=i, stratify=labels)

        dataset = CustomDataset(data_list_img1=data_resampled[:, 0], data_list_img2=data_resampled[:, 1], data_list_img3=data_resampled[:, 2],
                                data_list_num=data_resampled[:, 3], label_list=labels_resampled, transform=data_transform)

        k_folds = 5
        num_epochs = 100
        device = torch.device('cuda')

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
            fold_counter = 1

            for train_idx, val_idx in kf.split(data_resampled, labels_resampled):
                input_size_img1, input_size_img2, input_size_img3, input_size_num = dataset.feature_size()
                num_heads = 8
                img_model = ImageModel(params['activation'])
                num_model = NumModel(input_size_num)
                model = FusionModel(img_model, num_model, num_heads=num_heads)

                train_set = Subset(dataset, train_idx)
                val_set = Subset(dataset, val_idx)
                train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
                criterion = params['criterion']
                optimizer_class = params['optimizer']
                val_acc, best_fold_model, train_time = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer_class, params['learning_rate'], device, num_epochs, patience=20)

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

                all_accuracies.append(val_acc)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
                all_rocs.append(roc)

                fold_counter += 1
            param_counter += 1

    avg_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    avg_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions)
    avg_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls)
    avg_f1 = np.mean(all_f1s)
    std_f1 = np.std(all_f1s)
    avg_roc = np.mean(all_rocs)
    std_roc = np.std(all_rocs)

    print(f"{experiment_name} results:")
    print(f"Accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"Precision: {avg_precision:.3f} ± {std_precision:.3f}")
    print(f"Recall: {avg_recall:.3f} ± {std_recall:.3f}")
    print(f"F1: {avg_f1:.3f} ± {std_f1:.3f}")
    print(f"ROC AUC: {avg_roc:.3f} ± {std_roc:.3f}")


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

folder_labels = {
    "NC": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
}
base_path = '/home/user/xuxiao/ABAFnet/audio_data/CNRAC'
feature_path = '/home/user/xuxiao/ABAFnet/features/CNRAC_features'
df = pd.read_csv('/home/user/xuxiao/ABAFnet/CNRAC_features/emo_large_res.csv')
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


# Load your data from file
hdf1 = pd.read_excel('HAMD（13-24）-155.xlsx', sheet_name='病人组', engine='openpyxl')
hdf2 = pd.read_excel('HAMD（13-24）-155.xlsx', sheet_name='健康组', engine='openpyxl')
hdf = pd.concat([hdf1, hdf2], axis=0).reset_index(drop=True)[['group', 'standard_id', 'HAMD17_total_score']]
features = pd.read_csv('emo_large_res.csv')
header = pd.read_csv('reduced_data.csv').columns
data = pd.merge(hdf, features, left_on='standard_id', right_on='name')[header.append(pd.Index(['group', 'standard_id', 'HAMD17_total_score']))]

data_list_img1, data_list_img2, data_list_img3 = [], [], []
for i in range(len(hdf)):
    data_list_img1.append('/home/user/xuxiao/DeepL/image-features/' + data['standard_id'][i] + '/envelope.png')
    data_list_img2.append('/home/user/xuxiao/DeepL/image-features/' + data['standard_id'][i] + '/spectrogram.png')
    data_list_img3.append('/home/user/xuxiao/DeepL/image-features/' + data['standard_id'][i] + '/mel_spectrogram.png')

data_list_num = data.drop(columns=['group', 'standard_id', 'HAMD17_total_score']).values.tolist()

# Define the groups
healthy_group = data[data['group'] == 0]
mild_group = data[(data['group'] != 0) & (data['HAMD17_total_score'] >= 8) & (data['HAMD17_total_score'] <= 16)]
moderate_group = data[(data['group'] != 0) & (data['HAMD17_total_score'] >= 17) & (data['HAMD17_total_score'] <= 23)]
severe_group = data[(data['group'] != 0) & (data['HAMD17_total_score'] >= 24)]

# Define the experiments
experiments = [
    ("Healthy vs Mild", healthy_group, mild_group),
    ("Healthy vs Moderate", healthy_group, moderate_group),
    ("Healthy vs Severe", healthy_group, severe_group),
    ("Mild vs Moderate", mild_group, moderate_group),
    ("Mild vs Severe", mild_group, severe_group),
    ("Moderate vs Severe", moderate_group, severe_group),
]

for experiment_name, group1, group2 in experiments:
    group1_data = np.array(list(zip(group1['standard_id'].apply(lambda x: '/home/user/xuxiao/DeepL/image-features/' + x + '/envelope.png'),
                                     group1['standard_id'].apply(lambda x: '/home/user/xuxiao/DeepL/image-features/' + x + '/spectrogram.png'),
                                     group1['standard_id'].apply(lambda x: '/home/user/xuxiao/DeepL/image-features/' + x + '/mel_spectrogram.png'),
                                     group1.drop(columns=['group', 'standard_id', 'HAMD17_total_score']).values.tolist())), dtype=object)
    group1_labels = np.zeros(len(group1))
    
    group2_data = np.array(list(zip(group2['standard_id'].apply(lambda x: '/home/user/xuxiao/DeepL/image-features/' + x + '/envelope.png'),
                                     group2['standard_id'].apply(lambda x: '/home/user/xuxiao/DeepL/image-features/' + x + '/spectrogram.png'),
                                     group2['standard_id'].apply(lambda x: '/home/user/xuxiao/DeepL/image-features/' + x + '/mel_spectrogram.png'),
                                     group2.drop(columns=['group', 'standard_id', 'HAMD17_total_score']).values.tolist())), dtype=object)
    group2_labels = np.ones(len(group2))
    combined_data = np.concatenate([group1_data, group2_data], axis=0)
    combined_labels = np.concatenate([group1_labels, group2_labels], axis=0)

    run_experiment(combined_data, combined_labels, experiment_name)

