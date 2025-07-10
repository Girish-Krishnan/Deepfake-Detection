import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import os
import glob
from deepfake_detection.dataset import ImageDataset
import numpy as np

# Add random seeds
torch.manual_seed(0)
np.random.seed(0)


# Paths to training and validation directories
train_real_dir = 'train_real'
train_fake_dir = 'train_fake'
valid_real_dir = 'valid_real'
valid_fake_dir = 'valid_fake'

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Dataset initialization
train_dataset = ImageDataset(train_real_dir, train_fake_dir, transform=train_transform)
val_dataset = ImageDataset(valid_real_dir, valid_fake_dir, transform=val_transform)

# Calculate class weights
num_real = len(glob.glob(os.path.join(train_real_dir, '*.png')))
num_fake = len(glob.glob(os.path.join(train_fake_dir, '*.png')))
total_samples = num_real + num_fake
weight_real = total_samples / (2 * num_real)
weight_fake = total_samples / (2 * num_fake)
class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float)

# Weighted sampler
weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# DataLoaders
batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.inception_v3(pretrained=True)
model.aux_logits = False  # Disable auxiliary logits for standard training
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)

# Optimizer, scheduler, and criterion
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4, eps=1e-08)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training loop
best_val_accuracy = 0.0
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    roc_auc = roc_auc_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model.state_dict()

    scheduler.step()

# Save the final best model
torch.save(best_model, "best_inceptionv3_model.pth")
print("Training complete. Best validation accuracy:", best_val_accuracy)
