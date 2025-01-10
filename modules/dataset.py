import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from pathlib import Path
import PIL.Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt

class PixelArtDataset(Dataset):
    def __init__(self, image_dir, image_size=32):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.png"))
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path).convert('RGB')
        return self.transform(image)
