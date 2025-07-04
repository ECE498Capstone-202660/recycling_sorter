import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from mobilenet import MaterialClassifier
from data_loader import get_data_loaders
import os
import pandas as pd

CATEGORIES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

def compute_class_weights(data_dir, num_classes=6):
    counts = [0] * num_classes
    for split in ["train", "val", "test"]:
        csv_file = os.path.join(data_dir, split, f"{split}_labels.csv")
        if os.path.exists(csv_file):
            labels = pd.read_csv(csv_file)["label"]
            for label in labels:
                if 0 <= label < num_classes:
                    counts[label] += 1
    total = sum(counts)
    return torch.tensor([total / c if c > 0 else 0 for c in counts], dtype=torch.float)

def plot_loss_curves(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{title} Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_training_curves.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=80):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for (x, w), y in train_loader:
            x, w, y = x.to(device), w.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, w)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = 100. * correct / total

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for (x, w), y in val_loader:
                x, w, y = x.to(device), w.to(device), y.to(device)
                out = model(x, w)
                val_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = 100. * correct / total
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:02d}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_cm_model.pth")
            print(f" New best model saved with Val Acc={val_acc:.2f}%")

    return history

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for (x, w), y in test_loader:
            x, w, y = x.to(device), w.to(device), y.to(device)
            out = model(x, w)
            test_loss += criterion(out, y).item()
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    acc = 100. * sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {acc:.2f}%")
    return preds, labels

def main():
    data_dir = "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=32)
    class_weights = compute_class_weights(data_dir).to(device)

    model = MaterialClassifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
    plot_loss_curves(history, "One Stage Classification")

    model.load_state_dict(torch.load("best_cm_model.pth"))
    preds, labels = evaluate_model(model, test_loader, criterion, device)
    plot_confusion_matrix(labels, preds, CATEGORIES, "One Stage Classification")
    print("\n" + classification_report(labels, preds, target_names=CATEGORIES))

if __name__ == '__main__':
    main()
