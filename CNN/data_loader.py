import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

CATEGORY_WEIGHTS = {
    0: (30, 60),     # Cardboard
    1: (100, 200),   # Glass
    2: (10, 30),     # Metal
    3: (4, 6),       # Paper
    4: (10, 30),     # Plastic
    5: (5, 25),      # Trash
}

class WasteDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        weight = float(self.data.iloc[idx, 2])
        if self.transform:
            image = self.transform(image)
        return (image, torch.tensor([weight], dtype=torch.float32)), label

def _inject_weight_column(csv_path):
    df = pd.read_csv(csv_path)
    if "weight" in df.columns:
        return  # already injected

    weights = []
    for label in df["label"]:
        low, high = CATEGORY_WEIGHTS[label]
        weights.append(round(random.uniform(low, high), 2))
    df["weight"] = weights
    df.to_csv(csv_path, index=False)

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
        csv_file = os.path.join(data_dir, split, f"{split}_labels.csv")
        _inject_weight_column(csv_file)
        dataset = WasteDataset(
            csv_file=csv_file,
            img_dir=os.path.join(data_dir, split, "images"),
            transform=transform
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, pin_memory=True)

    return make_loader("train", transform_train), make_loader("val", transform_eval), make_loader("test", transform_eval)
