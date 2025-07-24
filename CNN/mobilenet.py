import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class MaterialClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(MaterialClassifier, self).__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 64, downsample=True),      # 128x128
            ResidualBlock(64, 128, downsample=True),    # 64x64
            ResidualBlock(128, 256, downsample=True),   # 32x32
            ResidualBlock(256, 512, downsample=True),   # 16x16
            ResidualBlock(512, 512, downsample=True),   # 8x8
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, weight):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, weight), dim=1)
        return self.classifier(x)