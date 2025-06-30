import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from model import MaterialClassifier, TrashRecycleClassifier
from data_loader import get_data_loaders
import os
import time

# Categories
CATEGORIES_STAGE1 = ["Recycle", "Trash"]
CATEGORIES_STAGE2 = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]

# 🔧 Compute class weights
def compute_class_weights(data_dir, num_classes, stage1=False, recycle_only=False):
    class_counts = [0] * num_classes
    for split in ["train", "val", "test"]:
        csv_file = os.path.join(data_dir, split, f"{split}_labels.csv")
        if os.path.exists(csv_file):
            labels = pd.read_csv(csv_file)["label"]
            for label in labels:
                if recycle_only and label > 4:
                    continue
                if stage1:
                    label = 0 if label in [0,1,2,3,4] else 1
                if 0 <= label < num_classes:
                    class_counts[label] += 1
    total = sum(class_counts)
    return torch.tensor([total / c if c > 0 else 0 for c in class_counts], dtype=torch.float)

# 🔧 Plotting functions
def plot_loss_curves(history, stage):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{stage} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{stage} Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{stage.replace(' ', '_').lower()}_curves.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, stage):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{stage} Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{stage.replace(' ', '_').lower()}_confusion.png")
    plt.close()

# 🔧 Training & evaluation
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
            t_correct += (out.argmax(1) == y).sum().item()
            t_total += y.size(0)
        train_acc = 100. * t_correct / t_total

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item()
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += y.size(0)
        val_acc = 100. * v_correct / v_total

        history['train_loss'].append(t_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        scheduler.step(v_loss / len(val_loader))

        print(f"Epoch {epoch+1:02d}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, LR={optimizer.param_groups[0]['lr']:.6f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✔️ New best model saved at epoch {epoch+1} with Val Acc={val_acc:.2f}%")
    return history

def evaluate_model(model, test_loader, criterion, device='cuda'):
    model.eval()
    test_loss, preds, trues = 0, [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_loss += criterion(out, y).item()
            preds.extend(out.argmax(1).cpu().tolist())
            trues.extend(y.cpu().tolist())
    acc = 100. * sum(p == t for p, t in zip(preds, trues)) / len(trues)
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {acc:.2f}%")
    return preds, trues

# 🔧 Main entry point
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "C:/Users/10640/recycling_sorter_sw/CNN/data"
    print(f"[INFO] Device: {device}")

    # 🔷 Stage 1: Trash vs Recycle
    print("\n🔷 Stage 1: Trash vs Recycle")
    train_loader1, val_loader1, test_loader1 = get_data_loaders(data_dir, batch_size=32, stage1=True)
    weights1 = compute_class_weights(data_dir, num_classes=2, stage1=True)
    model1 = TrashRecycleClassifier().to(device)
    criterion1 = nn.CrossEntropyLoss(weight=weights1.to(device))
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=5)

    hist1 = train_model(model1, train_loader1, val_loader1, criterion1, optimizer1, scheduler1, device=device)
    plot_loss_curves(hist1, "Stage 1 Trash vs Recycle")
    model1.load_state_dict(torch.load('best_model.pth'))
    preds1, trues1 = evaluate_model(model1, test_loader1, criterion1, device)
    plot_confusion_matrix(trues1, preds1, CATEGORIES_STAGE1, "Stage 1 Trash vs Recycle")
    print("\n" + classification_report(trues1, preds1, target_names=CATEGORIES_STAGE1))

    # 🔶 Stage 2: Recyclable Types
    print("\n🔶 Stage 2: Recyclable Types")
    train_loader2, val_loader2, test_loader2 = get_data_loaders(data_dir, batch_size=32, recycle_only=True)
    weights2 = compute_class_weights(data_dir, num_classes=5, recycle_only=True)
    model2 = MaterialClassifier(num_classes=5).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=weights2.to(device))
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.1, patience=5)

    hist2 = train_model(model2, train_loader2, val_loader2, criterion2, optimizer2, scheduler2, device=device)
    plot_loss_curves(hist2, "Stage 2 Recyclable Types")
    model2.load_state_dict(torch.load('best_model.pth'))
    preds2, trues2 = evaluate_model(model2, test_loader2, criterion2, device)
    plot_confusion_matrix(trues2, preds2, CATEGORIES_STAGE2, "Stage 2 Recyclable Types")
    print("\n" + classification_report(trues2, preds2, target_names=CATEGORIES_STAGE2))


if __name__ == '__main__':
    main()
