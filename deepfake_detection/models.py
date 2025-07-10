import torch
import torch.nn as nn
from torchvision import models, transforms
import timm


class CNNModel(nn.Module):
    """Simple CNN baseline."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Adaptive pooling ensures the fully connected layer sees a fixed
        # feature size regardless of the input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def get_model(name: str):
    name = name.lower()
    if name == "basic_cnn":
        model = CNNModel()
    elif name == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "inception":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
    elif name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    elif name == "xception":
        model = timm.create_model("xception", pretrained=True, num_classes=2)
    elif name == "wavelet_clip":
        from Wavelet_CLIP.training.detectors.clip_detector_wavelet import CLIPDetectorWavelet
        config = {"loss_func": "cross_entropy"}
        model = CLIPDetectorWavelet(config)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def get_transforms(name: str):
    """Return consistent train and validation transforms for all models."""
    _ = name  # the parameter is kept for backward compatibility
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_t, val_t

