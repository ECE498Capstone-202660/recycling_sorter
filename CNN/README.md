# CNN Module for Waste Classification

This module implements a Convolutional Neural Network (CNN) for classifying waste items into 12 distinct categories based on material type and rebate status.

## Model Architecture

The CNN architecture is designed for efficient waste classification with the following components:

### Convolutional Blocks
- **Block 1**: 2x Conv2D(32) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
- **Block 2**: 2x Conv2D(64) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
- **Block 3**: 2x Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout(0.25)

### Fully Connected Layers
- Dense(512) + ReLU + Dropout(0.5)
- Dense(12) for classification

## Data Processing

### Image Preprocessing
- Resize to 32x32 pixels
- Normalize using ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Data Augmentation (Training Only)
- Random horizontal flips
- Random rotations (±10 degrees)
- Color jittering (brightness and contrast)

## Training Configuration

### Hyperparameters
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 1e-4
- Epochs: 50
- Optimizer: Adam
- Learning rate scheduler: ReduceLROnPlateau
  - Factor: 0.1
  - Patience: 5 epochs

### Training Process
1. Data loading with augmentation
2. Forward pass through CNN
3. Loss computation (CrossEntropyLoss)
4. Backpropagation
5. Parameter updates
6. Validation after each epoch
7. Model checkpointing based on validation accuracy

## Model Files

- `model.py`: CNN architecture definition
- `data_loader.py`: Dataset and data loading implementation
- `train.py`: Training loop and model optimization
- `test.py`: Model evaluation and inference

## Usage

### Training
```python
python train.py
```

### Testing
```python
python test.py
```

### Making Predictions
```python
from test import load_model, predict_single_image

# Load model
model = load_model('checkpoints/best_model.pth', device)

# Predict
class_id, probabilities = predict_single_image(model, 'path_to_image.jpg', device)
```

## Performance Metrics

The model is evaluated using:
- Training accuracy
- Validation accuracy
- Test accuracy
- Per-class accuracy
- Confusion matrix

## Model Checkpoints

Checkpoints are saved in the `checkpoints` directory and include:
- Model state dictionary
- Optimizer state
- Current epoch
- Best validation accuracy

## Dependencies

Required Python packages:
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.19.2
- pandas>=1.2.4
- Pillow>=8.2.0
- tqdm>=4.61.0
- scikit-learn>=0.24.2 