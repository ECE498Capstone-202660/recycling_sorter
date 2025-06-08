import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                     stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=12):
        super(CustomMobileNet, self).__init__()
        
        # Initial convolution layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 64, stride=1)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512, stride=1)
        )
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Stage 1: 32 -> 64
        x = self.stage1(x)
        
        # Stage 2: 64 -> 128
        x = self.stage2(x)
        
        # Stage 3: 128 -> 256
        x = self.stage3(x)
        
        # Stage 4: 256 -> 512
        x = self.stage4(x)
        
        # Final processing
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def get_mobilenet_model(num_classes=12):
    """Helper function to create and return the model"""
    return CustomMobileNet(num_classes=num_classes) 