import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import shutil
import csv
import random

# 🔹 Custom dataset class
class WasteDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, stage1=False, recycle_only=False):
        self.data = pd.read_csv(csv_file)
        if recycle_only:
            self.data = self.data[self.data["label"].isin([0, 1, 2, 3, 4])]
        self.img_dir = img_dir
        self.transform = transform
        self.stage1 = stage1  # If True: label becomes 0 (recycle) or 1 (trash)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.stage1:
            label = 0 if label in [0, 1, 2, 3, 4] else 1  # Recycle vs Trash

        if self.transform:
            image = self.transform(image)

        return image, label


# 🔹 Splitting images into train/val/test
def split_data(original_images_dir, data_dir, train_ratio=0.7, val_ratio=0.2):
    class_mapping = {
        "cardboard": 0,
        "glass": 1,
        "metal": 2,
        "paper": 3,
        "plastic": 4,
        "trash": 5
    }

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    for class_name, class_id in class_mapping.items():
        class_dir = os.path.join(original_images_dir, class_name)
        if os.path.exists(class_dir):
            images = [img for img in os.listdir(class_dir) if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            random.shuffle(images)

            train_count = int(len(images) * train_ratio)
            val_count = int(len(images) * val_ratio)

            train_images = images[:train_count]
            val_images = images[train_count:train_count + val_count]
            test_images = images[train_count + val_count:]

            for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
                split_dir = os.path.join(data_dir, split)
                img_dir = os.path.join(split_dir, "images")
                for img_file in split_images:
                    src = os.path.join(class_dir, img_file)
                    dst = os.path.join(img_dir, img_file)
                    shutil.copy(src, dst)


# 🔹 Generate CSVs from image folder structure
def generate_csv(data_dir, split):
    class_mapping = {
        "cardboard": 0,
        "glass": 1,
        "metal": 2,
        "paper": 3,
        "plastic": 4,
        "trash": 5
    }

    split_dir = os.path.join(data_dir, split)
    img_dir = os.path.join(split_dir, "images")
    csv_path = os.path.join(split_dir, f"{split}_labels.csv")

    data = []
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            category = img_file.split("_")[0].lower()
            if category in class_mapping:
                label = class_mapping[category]
                data.append([img_file, label])

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])
        writer.writerows(data)


# 🔹 General-purpose loader (supports both Stage 1 and Stage 2)
def get_data_loaders(data_dir, batch_size=32, stage1=False, recycle_only=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = WasteDataset(
        csv_file=os.path.join(data_dir, "train", "train_labels.csv"),
        img_dir=os.path.join(data_dir, "train", "images"),
        transform=transform,
        stage1=stage1,
        recycle_only=recycle_only
    )

    val_dataset = WasteDataset(
        csv_file=os.path.join(data_dir, "val", "val_labels.csv"),
        img_dir=os.path.join(data_dir, "val", "images"),
        transform=val_transform,
        stage1=stage1,
        recycle_only=recycle_only
    )

    test_dataset = WasteDataset(
        csv_file=os.path.join(data_dir, "test", "test_labels.csv"),
        img_dir=os.path.join(data_dir, "test", "images"),
        transform=val_transform,
        stage1=stage1,
        recycle_only=recycle_only
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
