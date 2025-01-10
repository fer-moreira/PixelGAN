import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
import argparse
from pathlib import Path
from src.logger import logging

def load_trained_model(model_path, device="cuda"):
    # Load the original SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    # Load your trained UNet weights
    state_dict = load_file(model_path)
    pipe.unet.load_state_dict(state_dict)
    
    return pipe

def generate_images(
    model_path,
    prompt,
    output_dir,
    num_images=1,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=256,
    width=256,
    device="cuda"
):
    pipe = load_trained_model(model_path, device)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Generating {num_images} images with prompt: {prompt}")
    
    # Generate images
    with torch.autocast(device):
        for i in range(num_images):
            image = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            # Save the image
            image_path = f"{output_dir}/generated_{i+1}.png"
            image.save(image_path)
            logging.info(f"Saved image to {image_path}")

if __name__ == "__main__":

    model_path = input("Path to your trained model safetensor file: ")
    prompt = input("Text prompt for image generation: ")
    output_dir = input("Directory to save generated images: ")
    num_images = int(input("Number of images to generate (1): ") or 1)
    guidance_scale = float(input("Number of images to generate (7.5): ") or 7.5)
    num_inference_steps = int(input("Number of images to generate (50): ") or 50)
    height = int(input("Number of images to generate (512): ") or 512)
    width = int(input("Number of images to generate (512): ") or 512)
    device = str(input("Device to use for generation (cuda* \ cpu): ") or 'cuda')

    logging.debug(f"""Current training settings:
- Model Path: {model_path}
- Prompt: {prompt}
- Output: {output_dir}
- Num Images: {num_images}
- Guidance: {guidance_scale}
- Inference Step: {num_inference_steps}
- height: {height}
- width: {width}
- Device: {device}
    """)

    try:
        generate_images(
            model_path,
            prompt,
            output_dir,
            num_images,
            guidance_scale,
            num_inference_steps,
            height,
            width,
            device
        )
        
    except KeyboardInterrupt:
        logging.info("User canceled generation, exiting...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")