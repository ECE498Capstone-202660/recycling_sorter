import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from model import WasteClassifier
from data_loader import get_data_loaders
import os

# Define category information
CATEGORIES = {
    0: "Non-Recyclable",
    1: "Plastic 10¢ rebate",
    2: "Plastic 25¢ rebate",
    3: "Plastic No rebate",
    4: "Glass 10¢ rebate",
    5: "Glass 25¢ rebate",
    6: "Glass No rebate",
    7: "Glass High-value ($1 rebate)",
    8: "Metal 10¢ rebate",
    9: "Metal 25¢ rebate",
    10: "Metal No rebate",
    11: "Recyclable Paper/Cardboard"
}

def plot_loss_curves(history):
    """Plot training and validation loss and accuracy curves."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda'):
    """Train the model."""
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)
    
    return history

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """Evaluate the model on test set."""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    return all_preds, all_labels

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    data_dir = 'data'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=32)
    
    # Create model
    model = WasteClassifier(num_classes=12).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device
    )
    
    # Plot training curves
    plot_loss_curves(history)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
    
    # Plot confusion matrix
    category_names = list(CATEGORIES.values())
    plot_confusion_matrix(all_labels, all_preds, category_names)
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=category_names))
    
    # Save category information
    with open('category_info.txt', 'w') as f:
        f.write("Category Information:\n")
        f.write("-" * 50 + "\n")
        for idx, (name, info) in CATEGORIES.items():
            f.write(f"Class {idx}: {name}\n")
        f.write("\nRebate Information:\n")
        f.write("-" * 50 + "\n")
        f.write("10¢ rebate: Small plastic/glass bottles, small aluminum cans\n")
        f.write("25¢ rebate: Large plastic/glass bottles, large aluminum cans\n")
        f.write("$1 rebate: Wine bottles, liquor bottles (750mL+)\n")
        f.write("No rebate: Non-recyclable items, food containers, paper/cardboard\n")

if __name__ == '__main__':
    main()