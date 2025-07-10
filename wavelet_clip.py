import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
from Wavelet_CLIP.training.detectors.clip_detector_wavelet import CLIPDetectorWavelet  # Importing Wavelet-CLIP model
from Wavelet_CLIP.training.optimizor.SAM import SAM  # Importing SAM optimizer

# Add random seeds
torch.manual_seed(0)

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
        return {'image': image, 'label': torch.tensor(label)}

# Paths to training and validation directories
train_real_dir = 'train_real'
train_fake_dir = 'train_fake'
valid_real_dir = 'valid_real'
valid_fake_dir = 'valid_fake'

# Transformations for data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

if __name__ == "__main__":
    # Create datasets for training and validation
    train_dataset = CustomDataset(train_real_dir, train_fake_dir, transform=train_transform)
    val_dataset = CustomDataset(valid_real_dir, valid_fake_dir, transform=val_transform)

    # DataLoaders for training and validation with optimized settings
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load Wavelet-CLIP model
    device = torch.device('mps')
    config = {'loss_func': 'cross_entropy'}
    model = CLIPDetectorWavelet(config).to(device)

    num_real = len(train_dataset.real_images)
    num_fake = len(train_dataset.fake_images)
    total_samples = num_real + num_fake

    # Weight inversely proportional to class frequency
    weight_real = total_samples / (2 * num_real)
    weight_fake = total_samples / (2 * num_fake)
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float).to(device)

    # Set up SAM optimizer and learning rate scheduler
    base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # Define loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop with SAM optimizer and validation tracking
    best_val_accuracy = 0.0
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            data_dict = {k: v.to(device) for k, v in batch.items()}
            
            # Define closure function
            def closure():
                optimizer.zero_grad()
                pred_dict = model(data_dict)  # Forward pass
                loss = criterion(pred_dict['cls'], data_dict['label'])
                loss.backward()
                return loss, pred_dict  # Return both loss and predictions

            # First SAM step with closure
            loss, pred_dict = closure()  # Retrieve both loss and pred_dict
            optimizer.step(lambda: closure()[0])  # Step only uses the loss part of closure
            
            if loss is not None:
                total_loss += loss.item()  # Accumulate the loss value

            # Track accuracy
            _, predicted = torch.max(pred_dict['cls'], 1)
            correct += (predicted == data_dict['label']).sum().item()
            total += data_dict['label'].size(0)

            progress_bar.set_postfix({"Loss": total_loss / len(train_loader), "Training Accuracy": correct / total})

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Training Accuracy: {correct / total:.4f}")

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                data_dict = {k: v.to(device) for k, v in batch.items()}
                pred_dict = model(data_dict)
                _, predicted = torch.max(pred_dict['cls'], 1)
                val_correct += (predicted == data_dict['label']).sum().item()
                val_total += data_dict['label'].size(0)
        val_accuracy = val_correct / val_total
        print(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy:.4f}")

        # Update learning rate scheduler
        scheduler.step(val_accuracy)

        # Check for early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()  # Save the best model

    # Load the best model and report final validation accuracy
    model.load_state_dict(best_model)
    model.eval()
    final_val_correct = 0
    final_val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            data_dict = {k: v.to(device) for k, v in batch.items()}
            pred_dict = model(data_dict)
            _, predicted = torch.max(pred_dict['cls'], 1)
            final_val_correct += (predicted == data_dict['label']).sum().item()
            final_val_total += data_dict['label'].size(0)
    final_val_accuracy = final_val_correct / final_val_total
    print(f"Final Best Validation Accuracy: {final_val_accuracy:.4f}")
