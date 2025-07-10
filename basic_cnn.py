import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Add random seeds
torch.manual_seed(0)
np.random.seed(0)

# Custom Dataset class for separate training and validation directories
class CustomDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = glob.glob(os.path.join(real_dir, '*.png'))
        self.fake_images = glob.glob(os.path.join(fake_dir, '*.png'))
        self.transform = transform
        # Combine image paths and labels (0 for real, 1 for fake)
        self.data = [(img_path, 0) for img_path in self.real_images] + \
                    [(img_path, 1) for img_path in self.fake_images]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Paths to training and validation directories
train_real_dir = 'train_real'
train_fake_dir = 'train_fake'
valid_real_dir = 'valid_real'
valid_fake_dir = 'valid_fake'

# Transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Define a simple CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Create datasets for training and validation
    train_dataset = CustomDataset(train_real_dir, train_fake_dir, transform=transform)
    val_dataset = CustomDataset(valid_real_dir, valid_fake_dir, transform=transform)

    # DataLoaders for training and validation with optimized settings
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for better training
        num_workers=4,  # Adjust based on CPU capacity
        pin_memory=True,  # Helpful if using GPU
        persistent_workers=True  # Keeps workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Instantiate model, loss function, and optimizer
    device = torch.device('mps')
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Count the number of real and fake images in the training set
    num_real = len(glob.glob(os.path.join(train_real_dir, '*.png')))
    num_fake = len(glob.glob(os.path.join(train_fake_dir, '*.png')))
    print(f"Number of real images: {num_real}")
    print(f"Number of fake images: {num_fake}")

    # Compute class weights (inversely proportional to the class size)
    total_samples = num_real + num_fake
    weight_real = total_samples / (2 * num_real)
    weight_fake = total_samples / (2 * num_fake)

    # Convert to tensor and move to device
    class_weights_tensor = torch.tensor([weight_real, weight_fake], dtype=torch.float).to(device)
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
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
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({"Loss": total_loss / len(train_loader), "Training Accuracy": correct / total})

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Training Accuracy: {correct / total:.4f}")

    # Validation on validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {correct / total:.4f}")

