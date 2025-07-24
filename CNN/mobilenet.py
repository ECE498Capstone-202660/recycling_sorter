import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetWithMass(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(MobileNetWithMass, self).__init__()
        # Load MobileNetV2 with pretrained weights
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)

        # Remove the original classification head
        self.backbone.classifier = nn.Identity()

        # New classifier that takes in image features + 1D mass input
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, mass):
        # x: (B, 3, H, W), mass: (B, 1)
        x = self.backbone(x)           # Output shape: (B, 1280)
        x = torch.cat((x, mass), dim=1)  # Concatenate mass → shape: (B, 1281)
        return self.classifier(x)
