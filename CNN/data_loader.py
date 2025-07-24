import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# Updated CATEGORY_WEIGHTS and removed Cardboard
CATEGORY_WEIGHTS = {
    0: (200, 280),   # Glass
    1: (14, 18),     # Metal
    2: (5, 12),      # Paper
    3: (24, 35),     # Plastic
    4: (3, 8),      # Trash
}

class WasteDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.min_weight = self.data['weight'].min()
        self.max_weight = self.data['weight'].max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        weight = float(self.data.iloc[idx, 2])
        # Normalize weight to [0, 1]
        norm_weight = 5 * (weight - self.min_weight) / (self.max_weight - self.min_weight + 1e-8)
        if self.transform:
            image = self.transform(image)
        return (image, torch.tensor([norm_weight], dtype=torch.float32)), label

def _inject_weight_column(csv_path):
    df = pd.read_csv(csv_path)
    if "weight" in df.columns:
        return  # already injected
    weights = []
    for label in df["label"]:
        low, high = CATEGORY_WEIGHTS[label]
        mu = (low + high) / 2
        sigma = (high - low) / 6  # 99.7% values within [low, high]
        # Sample from Gaussian and clip to [low, high]
        w = np.random.normal(mu, sigma)
        w = min(max(w, low), high)
        weights.append(round(w, 2))
    df["weight"] = weights
    df.to_csv(csv_path, index=False)

def get_data_loaders(data_dir, batch_size=32):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),  # Random crop and resize
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])  

    transform_eval = transforms.Compose([
        transforms.Resize((256, 256)),
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