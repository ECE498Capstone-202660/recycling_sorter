import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import shutil
import csv
import random

class WasteDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(data_dir, batch_size=32):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    def make_loader(split, transform):
        dataset = WasteDataset(
            csv_file=os.path.join(data_dir, split, f"{split}_labels.csv"),
            img_dir=os.path.join(data_dir, split, "images"),
            transform=transform
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, pin_memory=True)

    return make_loader("train", transform_train), make_loader("val", transform_eval), make_loader("test", transform_eval)
