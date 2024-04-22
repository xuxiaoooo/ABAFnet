# Comparison Task
import os, time
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torchaudio.transforms import MelSpectrogram
import pandas as pd

class CustomModelBiLSTM(nn.Module):
    def __init__(self):
        super(CustomModelBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=224 * 3, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128 * 2, 1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, height, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return x

class CustomModelStackedLSTM(nn.Module):
    def __init__(self):
        super(CustomModelStackedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=224 * 3, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.dense = nn.Linear(64, 1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, height, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return x

class CustomModelDeepLSTM(nn.Module):
    def __init__(self):
        super(CustomModelDeepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=224 * 3, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)
        self.dense = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, height, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return x

class DepAudioNet(nn.Module):
    def __init__(self):
        super(DepAudioNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(2560, 512)
        self.fc2 = nn.Linear(512, 1)

    def _conv_forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

    def forward(self, x):
        x = self._conv_forward(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvBLSTM(nn.Module):
    def __init__(self):
        super(ConvBLSTM, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        
        self.lstm_input_size = 128 * 16 * 84  # Corrected input size
        self.hidden_size = 128
        self.num_layers = 2
        self.bidirectional = True
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        
        self.fc_out = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, 1, -1)
        
        _, (hn, _) = self.lstm(x)
        if self.bidirectional:
            hn = torch.cat((hn[-1], hn[-2]), dim=1)
        else:
            hn = hn[-1]
        
        x = self.fc_out(hn)
        x = torch.sigmoid(x)
        
        return x.squeeze()
        
def load_data(excel_file, audio_folder):
    # Load the data from the Excel file
    return audio_files, labels

class CustomDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None, target_length=None):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform
        self.target_length = target_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file_path = self.audio_files[idx]
        waveform, _ = torchaudio.load(audio_file_path)

        if self.transform:
            waveform = self.transform(waveform)

        # Pad or trim the waveform to the target length
        if self.target_length:
            if waveform.size(2) > self.target_length:
                waveform = waveform[:, :, :self.target_length]
            elif waveform.size(2) < self.target_length:
                padding = torch.zeros(waveform.size(0), waveform.size(1), self.target_length - waveform.size(2))
                waveform = torch.cat((waveform, padding), dim=2)

        audio_label = self.labels[idx]
        return waveform, torch.tensor(audio_label, dtype=torch.float32)

def train_and_evaluate_model(train_loader, val_loader, model, device, num_epochs=50, patience=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    start_time = time.time()  # 开始时间
    model = model.to(device)

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        train_start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        # Evaluate on validation set
        model.eval()
        val_start_time = time.time()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_end_time = time.time() 
        print(f"Validation Loss: {val_loss}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluate accuracy, recall, precision, and F1 score on validation set
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.tolist())
            all_predicted.extend(predicted.tolist())

    accuracy = correct / total
    recall = recall_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted)
    f1 = f1_score(all_labels, all_predicted)

    train_time = train_end_time - train_start_time  # 训练总时间
    val_time = val_end_time - val_start_time  # 验证总时间
    print(f"Total training time: {train_time} seconds.")
    print(f"Total validation time: {val_time} seconds.")

    print(f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")

def main(excel_file, audio_folder_path):
    device = torch.device("cuda")
    transform = MelSpectrogram()  # Define any desired transforms for the audio data here

    audio_files, labels = load_data(excel_file, audio_folder_path)
    target_length = 672  # Choose an appropriate value based on your audio files
    dataset = CustomDataset(audio_files, labels, transform=transform, target_length=target_length)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model1 = CustomModelBiLSTM()
    model2 = CustomModelStackedLSTM()
    model3 = CustomModelDeepLSTM()
    model4 = DepAudioNet()
    model5 = ConvBLSTM()


    train_and_evaluate_model(train_loader, val_loader, model5, device)

if __name__ == '__main__':
    excel_file = "/home/user/xuxiao/DeepL/HAMD（13-24）-155.xlsx"
    audio_folder_path = "/home/user/xuxiao/DeepL/ffmpegaudio"
    main(excel_file, audio_folder_path)