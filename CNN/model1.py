import torch
import torch.nn as nn

class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout=0.1):
        super().__init__()
        stride = 2 if downsample else 1
        mid_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))

        out += identity
        return self.relu(out)

class MaterialClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(MaterialClassifier, self).__init__()

        # Feature extractor with better residual blocks and less aggressive downsampling
        self.features = nn.Sequential(
            ImprovedResidualBlock(3, 64, downsample=True),     # 128 -> 64
            ImprovedResidualBlock(64, 128, downsample=True),   # 64 -> 32
            ImprovedResidualBlock(128, 256, downsample=False), # 32 -> 32
            ImprovedResidualBlock(256, 512, downsample=False), # 32 -> 32
            ImprovedResidualBlock(512, 512, downsample=True),  # 32 -> 16
        )

        # Global Average Pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Weight input embedding
        self.weight_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
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

        w = self.weight_embed(weight)  # shape: (batch_size, 64)

        x = torch.cat((x, w), dim=1)
        return self.classifier(x)
