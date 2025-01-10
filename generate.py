import torch
from pathlib import Path
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import time

# Import the Generator architecture from the training script
# Assuming it's saved in a file called 'model.py'
from train import PixelArtGenerator

def generate_pixel_art(
    model_path,
    output_dir,
    num_images=1,
    image_size=32,
    num_colors=16,
    latent_dim=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the generator
    generator = PixelArtGenerator(latent_dim, num_colors, image_size).to(device)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Set to evaluation mode
    generator.eval()
    
    print(f"Generating {num_images} images...")
    
    with torch.no_grad():
        for i in range(num_images):
            # Generate random noise
            noise = torch.randn(1, latent_dim).to(device)
            
            # Generate image
            fake_image = generator(noise)
            
            # Scale to 0-255 range and quantize colors
            image = (fake_image * 255).clamp(0, 255).to(torch.uint8)
            
            # Save the image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_image(fake_image.data, 
                      output_dir / f"pixel_art_{timestamp}_{i+1}.png", 
                      normalize=True)
            
            print(f"Generated image {i+1}/{num_images}")

if __name__ == "__main__":
    print("Pixel Art Generator")
    print("-" * 40)
    
    # Get model path
    while True:
        model_path = input("Enter the path to your trained model (.pth file): ").strip()
        if Path(model_path).exists():
            break
        print("Model file not found. Please enter a valid path.")
    
    # Get output directory
    output_dir = input("Enter the directory to save generated images: ").strip()
    
    # Get generation parameters
    try:
        num_images = int(input("How many images to generate? (default 1): ") or 1)
        image_size = int(input("Image size (must match training size, default 32): ") or 32)
        num_colors = int(input("Number of colors (must match training, default 16): ") or 16)
    except ValueError:
        print("Invalid input detected. Using default values.")
        num_images = 1
        image_size = 32
        num_colors = 16
    
    print("\nGeneration settings:")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of images: {num_images}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Number of colors: {num_colors}")
    print("-" * 40)
    
    # Ask for confirmation
    confirm = input("\nProceed with generation? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Generation cancelled.")
        exit()
    
    try:
        generate_pixel_art(
            model_path=model_path,
            output_dir=output_dir,
            num_images=num_images,
            image_size=image_size,
            num_colors=num_colors
        )
        print("\nGeneration completed successfully!")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during generation: {str(e)}")