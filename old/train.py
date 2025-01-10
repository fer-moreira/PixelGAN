import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from pathlib import Path
import PIL.Image
import numpy as np
from torchvision import transforms
from safetensors.torch import save_file
import argparse
import sys

from src.logger import logging


class PixelArtDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, image_size=512):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.png"))
        self.tokenizer = tokenizer

        # Load captions from file
        self.captions = {}
        with open(caption_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split(',', 1)
                self.captions[img_name] = caption

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.captions.get(image_path.name, None)

        if not caption:
            raise Exception(f"Missing caption in csv file for image: {image_path.name}")

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": encoding.input_ids[0]
        }

def train_pixel_art_model(
    image_dir,
    caption_file,
    output_dir,
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-5,
    save_steps=500
):
    # Initialize accelerator
    accelerator = Accelerator()


    # Load models
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    )

    # Move models to device before freezing
    device = accelerator.device
    text_encoder = text_encoder.to(device)
    vae = vae.to(accelerator.device)
    unet = unet.to(device)

    # Freeze CLIP text encoder and VAE
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # Create dataset and dataloader
    dataset = PixelArtDataset(image_dir, caption_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # Prepare for training using accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # Training loop
    global_step = 0

    logging.info("Starting training")

    for epoch in range(0, num_epochs):
        unet.train()
        for batch in dataloader:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Get latent representation
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
            noisy_latents = noise + timesteps.reshape(-1, 1, 1, 1) * latents

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            logging.info(f"Step {global_step}, Loss: {loss.item():.4f}")

            if global_step % save_steps == 0:
                logging.info(f"Trying to save model at step {global_step}")

                model_path = f"{output_dir}/model_{global_step}.safetensors"
                unwrapped_unet = accelerator.unwrap_model(unet)
                state_dict = unwrapped_unet.state_dict()
                save_file(state_dict, model_path)
                
                logging.info(f"Saved model at step {global_step}")
    
    logging.info(f"Trying to save model at end")
    
    model_path = f"{output_dir}/model_final.safetensors"
    unwrapped_unet = accelerator.unwrap_model(unet)
    state_dict = unwrapped_unet.state_dict()
    save_file(state_dict, model_path)

    logging.info(f"Model saved at end of training... 100% done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing pixel art images")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to caption file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_steps", type=int, default=500)

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.debug(f"""Current training settings:
- Input caption file: {args.caption_file}
- Input: {args.image_dir}
- Output: {args.output_dir}
- Epochs: {args.num_epochs}
- Batchs: {args.batch_size}
- Learning rate: {args.learning_rate}
- Save steps: {args.save_steps}
    """)

    try:
        train_pixel_art_model(**vars(args))
    except KeyboardInterrupt:
        logging.info("User canceled training, exiting...")
        sys.exit()
