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

def save_samples(generator, epoch, latent_dim, num_samples, device, save_dir):
    """Helper function to save sample generator images"""
    # Create directory if it doesn't exist
    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Generate sample images
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = generator(z)
        
        # Denormalize
        samples = (samples + 1) / 2
        
        # Save grid of images
        save_path = samples_dir / f"samples_epoch_{epoch}.png"
        grid = utils.make_grid(samples, nrow=int(np.sqrt(num_samples)), padding=2, normalize=False)
        utils.save_image(grid, save_path)
    
    generator.train()
