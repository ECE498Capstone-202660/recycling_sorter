import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from model import WasteClassifier
from data_loader import get_data_loaders

def load_model(model_path, device):
    model = WasteClassifier(num_classes=12).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def predict_single_image(model, image_path, device):
    # Define the same transform used in validation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return predicted.item(), probabilities[0].cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'Using device: GPU (cuda)')
    else:
        print(f'Using device: CPU')
        print('WARNING: CUDA GPU is not available. Testing will be much slower on CPU.')
    model = WasteClassifier(num_classes=12).to(device)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    data_dir = 'data'
    _, _, test_loader = get_data_loaders(data_dir, batch_size=32)
    test_model(model, test_loader, device)

    # Example of predicting a single image
    image_path = 'path_to_your_image.jpg'  # Update this path
    if os.path.exists(image_path):
        predicted_class, probabilities = predict_single_image(model, image_path, device)
        print(f'\nPrediction for {image_path}:')
        print(f'Predicted class: {predicted_class}')
        print('Class probabilities:')
        for i, prob in enumerate(probabilities):
            print(f'Class {i}: {prob:.4f}')

if __name__ == '__main__':
    main() 