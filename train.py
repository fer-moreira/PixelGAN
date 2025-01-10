import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

# pixel art related
from modules.generator import PixelArtGenerator
from modules.descriminator import PixelArtDiscriminator
from modules.dataset import PixelArtDataset
from modules.utils import save_samples

def train_pixel_art_model(
    image_dir,
    output_dir,
    image_size=32,
    latent_dim=100,
    num_colors=16,
    batch_size=32,
    num_epochs=200,
    lr=0.0001,
    beta1=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Create output directories
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create models
    generator = PixelArtGenerator(latent_dim, num_colors, image_size).to(device)
    discriminator = PixelArtDiscriminator(image_size).to(device)
    
    # Setup optimizers with different learning rates
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr*0.5, betas=(beta1, 0.999))
    
    # Learning rate scheduling
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, num_epochs)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, num_epochs)
    
    # Create dataset and dataloader
    dataset = PixelArtDataset(image_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Lists to store losses for plotting
    g_losses = []
    d_losses = []
    
    print(f"Starting training on device: {device}")
    print(f"Number of training images: {len(dataset)}")
    
    # Training loop
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train discriminator
            d_optimizer.zero_grad()
            
            # Train with real
            real_validity = discriminator(real_images)
            d_real_loss = F.binary_cross_entropy_with_logits(
                real_validity, 
                torch.ones_like(real_validity) * 0.9  # Label smoothing
            )
            
            # Train with fake
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_validity = discriminator(fake_images.detach())
            d_fake_loss = F.binary_cross_entropy_with_logits(
                fake_validity,
                torch.zeros_like(fake_validity) * 0.1  # Label smoothing
            )
            
            # Gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1).to(device)
            interpolates = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
            d_interpolates = discriminator(interpolates)
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True,
                retain_graph=True
            )[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            
            d_loss = d_real_loss + d_fake_loss + gradient_penalty
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            if i % 5 == 0:  # Update generator less frequently
                g_optimizer.zero_grad()
                fake_validity = discriminator(fake_images)
                g_loss = F.binary_cross_entropy_with_logits(
                    fake_validity,
                    torch.ones_like(fake_validity)
                )
                g_loss.backward()
                g_optimizer.step()
            
                # Store losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Step [{i}/{len(dataloader)}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')
        
        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()
        
    
    torch.save({
        'epoch': num_epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_scheduler_state_dict': g_scheduler.state_dict(),
        'd_scheduler_state_dict': d_scheduler.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
    }, models_dir / "trained_model.pth")
    
    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'loss_plot.png')
    plt.close()

if __name__ == "__main__":
    print("Pixel Art GAN Training")
    print("-" * 40)
    
    # Get input directory
    while True:
        image_dir = input("Enter the path to your training images directory: ").strip()
        if Path(image_dir).exists():
            break
        print("Directory not found. Please enter a valid path.")
    
    # Get output directory
    output_dir = input("Enter the path to save models and samples: ").strip()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get training parameters with defaults
    try:
        image_size = int(input("Enter image size (default 32): ") or 32)
        num_colors = int(input("Enter number of colors (default 16): ") or 16)
        batch_size = int(input("Enter batch size (default 32): ") or 32)
        num_epochs = int(input("Enter number of epochs (default 200): ") or 200)
        lr = float(input("Enter learning rate (default 0.0001): ") or 0.0001)
    except ValueError:
        print("Invalid input detected. Using default values.")
        image_size = 32
        num_colors = 16
        batch_size = 32
        num_epochs = 200
        lr = 0.0001
    
    print("\nStarting training with these parameters:")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Number of colors: {num_colors}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print("-" * 40)
    
    # Ask for confirmation
    confirm = input("\nProceed with training? (y/n): ").lower().strip

    if confirm == "y":
        train_pixel_art_model(
            image_dir,
            output_dir,
            image_size,
            100,
            num_colors,
            batch_size,
            num_epochs,
            lr
        )
    else:
        pass