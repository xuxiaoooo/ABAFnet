from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

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
