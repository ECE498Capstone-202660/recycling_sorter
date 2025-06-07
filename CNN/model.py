import torch
import torch.nn as nn
import torch.nn.functional as F

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=12):  # 12 categories for waste classification with rebate status
        super(WasteClassifier, self).__init__()
        
        # First Convolutional Block
        # This block processes the grayscale input image (1 channel) and extracts basic features
        # - First conv layer: Increases channels from 1 to 32, capturing basic patterns
        # - Second conv layer: Maintains 32 channels, refining the basic features
        # - MaxPool: Reduces spatial dimensions by half, making the network more robust to small variations
        # - Dropout: Prevents overfitting by randomly deactivating 25% of neurons
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: grayscale image (1 channel)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Second Convolutional Block
        # This block processes the features from block1 and extracts more complex patterns
        # - First conv layer: Doubles channels from 32 to 64, capturing more complex features
        # - Second conv layer: Maintains 64 channels, refining these complex features
        # - MaxPool: Further reduces spatial dimensions, increasing receptive field
        # - Dropout: Continues to prevent overfitting
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Third Convolutional Block
        # This block processes the features from block2 and extracts high-level patterns
        # - First conv layer: Doubles channels from 64 to 128, capturing high-level features
        # - Second conv layer: Maintains 128 channels, refining these high-level features
        # - MaxPool: Further reduces spatial dimensions, increasing receptive field
        # - Dropout: Continues to prevent overfitting
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Fourth Convolutional Block
        # This block processes the features from block3 and extracts the most abstract patterns
        # - First conv layer: Doubles channels from 128 to 256, capturing the most abstract features
        # - Second conv layer: Maintains 256 channels, refining these abstract features
        # - MaxPool: Final reduction of spatial dimensions
        # - Dropout: Final dropout layer before fully connected layers
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Calculate the size of the flattened features
        # For input size 32x32, after 4 maxpool layers (each reducing size by half)
        # Final feature map size is 2x2 with 256 channels
        self._to_linear = 256 * 2 * 2  # 256 channels * 2 height * 2 width
        
        # Fully Connected Layers
        # - First layer: Reduces dimensions from flattened features to 512
        # - Dropout: Prevents overfitting in fully connected layers
        # - Final layer: Maps to number of classes (12 waste categories with rebate status)
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Process input through convolutional blocks
        # Each block extracts increasingly complex features
        x = self.conv_block1(x)  # Extract basic features
        x = self.conv_block2(x)  # Extract complex features
        x = self.conv_block3(x)  # Extract high-level features
        x = self.conv_block4(x)  # Extract abstract features
        
        # Flatten the feature maps and pass through fully connected layers
        x = x.view(-1, self._to_linear)
        x = self.fc(x)
        
        return x
