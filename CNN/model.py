import torch
import torch.nn as nn
import torch.nn.functional as F

class MaterialClassifier(nn.Module):
    def __init__(self, num_classes=6):  # For stage 2 (recyclables)
        super(MaterialClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28

            nn.AdaptiveAvgPool2d((1, 1))  # Output size: [B, 128, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 128]
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TrashRecycleClassifier(nn.Module):
    def __init__(self):
        super(TrashRecycleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.AdaptiveAvgPool2d((1, 1))  # Output size: [B, 64, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
