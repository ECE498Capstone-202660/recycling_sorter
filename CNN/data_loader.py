import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

class WasteDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32):
    """
    Create data loaders for training, validation and test sets.
    
    Args:
        data_dir (string): Directory with all the data.
        batch_size (int): Batch size for the data loaders.
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for each set
    """
    # Training data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])

    # Validation and test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])

    # Create datasets
    train_dataset = WasteDataset(
        csv_file=os.path.join(data_dir, 'train', 'train_labels.csv'),
        img_dir=os.path.join(data_dir, 'train', 'images'),
        transform=train_transform
    )

    val_dataset = WasteDataset(
        csv_file=os.path.join(data_dir, 'val', 'val_labels.csv'),
        img_dir=os.path.join(data_dir, 'val', 'images'),
        transform=val_transform
    )

    test_dataset = WasteDataset(
        csv_file=os.path.join(data_dir, 'test', 'test_labels.csv'),
        img_dir=os.path.join(data_dir, 'test', 'images'),
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader 