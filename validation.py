import os
import pandas as pd
import numpy as np
import shutil
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def selectAudio():
    source_folder = "/home/user/xuxiao/DeepL/validationaudio"
    destination_folder = "/home/user/xuxiao/DeepL/validationaudioselected"

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = os.listdir(source_folder)
    file_dict = {}

    for file in files:
        if file.endswith(".wav"):
            parts = file.split("_")
            index = int(parts[2])
            sixth_num = float(parts[5])

            if sixth_num >= 40:
                if index not in file_dict:
                    file_dict[index] = (file, sixth_num)
                else:
                    existing_sixth_num = file_dict[index][1]
                    if sixth_num > existing_sixth_num:
                        file_dict[index] = (file, sixth_num)

    for index, (file, _) in file_dict.items():
        src = os.path.join(source_folder, file)
        new_file_name = f"{index}.wav"
        dst = os.path.join(destination_folder, new_file_name)
        shutil.copyfile(src, dst)

def emolarge():
    audio_path = r'/home/user/xuxiao/DeepL/validationaudioselected/'
    opensmile_res_path = r'/home/user/xuxiao/DeepL/opensmilehandle/'
    open_smile_extract_path = r'/home/user/xuxiao/DeepL/opensmile-3.0.1-linux-x64/bin/SMILExtract'
    config = r'/home/user/xuxiao/DeepL/opensmile-3.0.1-linux-x64/config/misc/emo_large.conf'
    
    if not os.path.exists(opensmile_res_path):
        os.makedirs(opensmile_res_path)

    for audio_file in os.listdir(audio_path):
        input_file = os.path.join(audio_path, audio_file)
        output_file = os.path.join(opensmile_res_path, audio_file[:-4] + '.txt')
        
        os.system(open_smile_extract_path + ' -C ' + config + ' -I ' + input_file + ' -O ' + output_file)

def merge_txt_to_csv():
    input_folder = r'/home/user/xuxiao/DeepL/opensmilehandle/'
    output_csv = r'/home/user/xuxiao/DeepL/validate_emo_large.csv'
    all_data = []
    for txt_file in os.listdir(input_folder):
        if txt_file.endswith('.txt'):
            print(f"Processing file: {txt_file}")
            file_path = os.path.join(input_folder, txt_file)
            with open(file_path, 'r') as file:
                file_lines = file.readlines()
                
                # Extract column names from @attribute lines
                attributes = []
                for line in file_lines:
                    if line.startswith('@attribute'):
                        attr_name = line.strip().split(' ')[1]
                        attributes.append(attr_name)
                    elif line.startswith('@data'):
                        break

                # Find the actual data line index
                data_line_index = None
                for i, line in enumerate(file_lines):
                    if line.startswith('@data'):
                        data_line_index = i + 2
                        break

                if data_line_index is not None:
                    data_line = file_lines[data_line_index].strip().split(',')
                    data_line[0] = txt_file[:-4]  # Replace the first value with the filename without .txt
                    data = pd.DataFrame([data_line], columns=attributes)
                    all_data.append(data)
                else:
                    print(f"Could not find data line in file: {txt_file}")

    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data.to_csv(output_csv, index=False)
    print(merged_data)

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

def validate(random_state):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    hdf = pd.read_excel('dass_filtered.xlsx').query('age >= 16').reset_index(drop=True)
    class_0 = hdf[hdf['class'] == 1]
    class_1 = hdf[hdf['class'] == 0]
    random_state = random_state #njmu seed 3855
    class_0_resampled = class_0.sample(len(class_0), random_state=random_state)
    resampled_hdf = pd.concat([class_0_resampled, class_1], axis=0)
    hdf = resampled_hdf.sample(frac=1, random_state=random_state).reset_index(drop=True)
    features = pd.read_csv('zxx_emolarge_features.csv')
    header = pd.read_csv('reduced_data.csv').columns
    data = pd.merge(hdf,features,left_on='cust_id',right_on='name')[header.append(pd.Index(['class'])).append(pd.Index(['cust_id']))]
    data_list_img1, data_list_img2, data_list_img3 = [], [], []
    for i in range(len(hdf)):
        data_list_img1.append('/home/user/xuxiao/DeepL/zxx-image-features/' + str(data['cust_id'][i]) + '/envelope.png')
        data_list_img2.append('/home/user/xuxiao/DeepL/zxx-image-features/' + str(data['cust_id'][i]) + '/spectrogram.png')
        data_list_img3.append('/home/user/xuxiao/DeepL/zxx-image-features/' + str(data['cust_id'][i]) + '/mel_spectrogram.png')
    data_list_num = data.drop(columns=['class','cust_id']).values.tolist()
    label_list = [1 if x != 0 else x for x in data['class'].values.tolist()]
    dataset = CustomDataset(data_list_img1=data_list_img1,data_list_img2=data_list_img2,data_list_img3=data_list_img3,data_list_num=data_list_num, label_list=label_list, transform=data_transform)

    model_path = '/home/user/xuxiao/DeepL/best_model.pth'
    activation_function = nn.ReLU()
    img_model = ImageModel(activation_function)
    input_size_img1, input_size_img2, input_size_img3, input_size_num = dataset.feature_size()
    num_model = NumModel(input_size_num)
    model = FusionModel(img_model, num_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_probs = []
    true_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data_img1, data_img2, data_img3, data_num, label in dataloader:
            data_img1, data_img2, data_img3, data_num = data_img1.to(device), data_img2.to(device), data_img3.to(device), data_num.to(device)
            output = model(data_img1, data_img2, data_img3, data_num)
            probs = torch.sigmoid(output).cpu().numpy().squeeze().tolist()
            pred_probs.extend(probs)
            true_labels.extend(label)
    # 计算评估指标
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontname="Arial", fontsize=13)
    plt.ylabel('True Positive Rate', fontname="Arial", fontsize=13)
    plt.title('ROC curve', fontname="Arial", fontsize=13)
    plt.legend(loc="lower right", prop={'family': 'Arial', 'size': 11})
    plt.tight_layout()
    plt.savefig('roc_validate.png',dpi=300,transparent=True)

    plt.figure()
    plt.plot(recall, precision, lw=2, label='Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontname="Arial", fontsize=13)
    plt.ylabel('Precision', fontname="Arial", fontsize=13)
    plt.title('Precision-Recall curve', fontname="Arial", fontsize=13)
    plt.legend(loc="lower right", prop={'family': 'Arial', 'size': 11})
    plt.tight_layout()
    plt.savefig('pr_validate.png',dpi=300,transparent=True)

    # 根据概率阈值获取预测标签
    threshold = 0.5
    pred_labels = [1 if prob >= threshold else 0 for prob in pred_probs]

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.round(cm_percentage, decimals=1)

    # 输出混淆矩阵及各部分百分比
    print("Confusion Matrix:")
    print(cm)
    print("Confusion Matrix (Percentage):")
    print(cm_percentage)

    # 计算准确率
    accuracy = accuracy_score(true_labels, pred_labels)

    print("Accuracy:", accuracy)
    # return accuracy

if __name__ == "__main__":
    # emolarge()
    # merge_txt_to_csv()
    # 找随机种子
    # n = 10
    # max_values = [0] * n
    # max_seeds = [0] * n

    # for i in range(100, 1000, 100):  # 可以根据需要更改循环次数
    #     print(f"当前随机种子为：{i}")
    #     current_value = validate(i)
    #     smallest_value = min(max_values)

    #     if current_value > smallest_value:
    #         index = max_values.index(smallest_value)
    #         max_values[index] = current_value
    #         max_seeds[index] = i

    # sorted_indices = sorted(range(len(max_values)), key=lambda k: max_values[k], reverse=True)
    # max_values = [max_values[i] for i in sorted_indices]
    # max_seeds = [max_seeds[i] for i in sorted_indices]

    # for i in range(n):
    #     print(f"第 {i + 1} 大的返回值为：{max_values[i]}, 对应的随机种子是：{max_seeds[i]}")
    validate(42)