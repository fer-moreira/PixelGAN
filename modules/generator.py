import torch
import torch.nn as nn

class PixelArtGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_colors=16, image_size=32):
        super().__init__()
        
        self.image_size = image_size
        self.initial_size = image_size // 8
        
        # More structured initial projection
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512 * self.initial_size * self.initial_size),
            nn.LeakyReLU(0.2)
        )
        
        # Improved generator architecture with residual connections
        self.main = nn.ModuleList([
            # First block
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            ),
            # Second block
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            ),
            # Third block
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Pixel-perfect final layers
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 1),
            nn.Tanh()
        )
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Color quantization
        self.num_colors = num_colors
    
    def quantize_colors(self, x):
        # Scale to [0, num_colors-1]
        x = ((x + 1) / 2) * (self.num_colors - 1)
        # Quantize
        x = torch.round(x)
        # Scale back to [-1, 1]
        x = (x / (self.num_colors - 1)) * 2 - 1
        return x
    
    def forward(self, z):
        # Initial projection
        x = self.fc(z)
        x = x.view(-1, 512, self.initial_size, self.initial_size)
        
        # Main generation pathway
        for block in self.main[:-1]:
            x = block(x)
        
        # Apply attention on final upsampling
        features = self.main[-1](x)
        attention = self.attention(features)
        x = features * attention
        
        # Final layers and color quantization
        x = self.final(x)
        x = self.quantize_colors(x)
        return x
