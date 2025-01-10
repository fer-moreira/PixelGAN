import torch.nn as nn

class PixelArtDiscriminator(nn.Module):
    def __init__(self, image_size=32):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, size):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.LayerNorm([out_channels, size, size]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                nn.LayerNorm([out_channels, size//2, size//2]),
                nn.LeakyReLU(0.2)
            )
        
        # Track image size as it gets downsampled
        current_size = image_size
        
        self.main = nn.Sequential(
            # Input layer
            discriminator_block(3, 64, current_size),
            # Feature extraction layers
            discriminator_block(64, 128, current_size//2),
            discriminator_block(128, 256, current_size//4),
            discriminator_block(256, 512, current_size//8),
        )
        
        # Calculate final feature map size
        final_size = image_size // 16
        
        self.adversarial = nn.Sequential(
            nn.Linear(512 * final_size * final_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        features = self.main(x)
        features = features.view(features.size(0), -1)
        return self.adversarial(features)