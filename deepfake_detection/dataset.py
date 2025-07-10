import glob
import os
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Dataset that loads images from real and fake directories."""

    def __init__(self, real_dir: str, fake_dir: str, transform: Optional[callable] = None):
        self.real_images = glob.glob(os.path.join(real_dir, "*.png"))
        self.fake_images = glob.glob(os.path.join(fake_dir, "*.png"))
        self.transform = transform
        self.data = [(p, 0) for p in self.real_images] + [(p, 1) for p in self.fake_images]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

