import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch and stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_ch))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=stride, padding=1, groups=hidden_ch, bias=False))
        layers.append(nn.BatchNorm2d(hidden_ch))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)

class MaterialClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.cfg = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        layers = []
        in_ch = 32
        for t, c, n, s in self.cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
        self.features = nn.Sequential(*layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_ch, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
