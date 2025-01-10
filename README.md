# PixelGAN
Trying to make a GAN that generate pixel art character



## Setup

    # Check if NVIDIA driver is properly installed in WSL 2
    nvidia-smi

    # Check CUDA version
    nvcc --version

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    pip install -r requirements.txt
    
    # For better performance
    pip install ninja
    
    # For monitoring training
    pip install wandb