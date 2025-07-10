from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from .dataset import ImageDataset
from .models import get_model, get_transforms


def create_loaders(train_real: str, train_fake: str, val_real: str, val_fake: str,
                   batch_size: int, train_transform, val_transform):
    train_dataset = ImageDataset(train_real, train_fake, transform=train_transform)
    val_dataset = ImageDataset(val_real, val_fake, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataset, val_dataset, train_loader, val_loader


def compute_class_weights(dataset, device):
    num_real = len(dataset.real_images)
    num_fake = len(dataset.fake_images)
    total_samples = num_real + num_fake
    weight_real = total_samples / (2 * num_real)
    weight_fake = total_samples / (2 * num_fake)
    return torch.tensor([weight_real, weight_fake], dtype=torch.float, device=device)


def train_model(model_name: str, train_real: str, train_fake: str, val_real: str, val_fake: str,
                epochs: int = 10, batch_size: int = 32, lr: float = 0.001,
                device: Optional[str] = None, save_path: Optional[str] = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = get_model(model_name).to(device)
    train_t, val_t = get_transforms(model_name)
    train_dataset, val_dataset, train_loader, val_loader = create_loaders(
        train_real, train_fake, val_real, val_fake, batch_size, train_t, val_t
    )
    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_accuracy = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_acc:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        roc_auc = roc_auc_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_acc:.4f}, ROC-AUC: {roc_auc:.4f}")
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = model.state_dict()
        scheduler.step()

    if save_path and best_state is not None:
        torch.save(best_state, save_path)
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    return best_val_accuracy

