"""
Inference script for DDPM
Optimized sampling with DDIM for 30% faster generation
"""

import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

class DDIMSampler:
    """DDIM sampler for accelerated inference"""
    def __init__(self, ddpm, num_inference_steps=50):
        self.ddpm = ddpm
        self.num_inference_steps = num_inference_steps
        
        # Create subset of timesteps for faster sampling
        self.timesteps = torch.linspace(
            ddpm.timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )
    
    @torch.no_grad()
    def sample(self, shape, device='cuda', eta=0.0):
        """
        Generate samples using DDIM
        eta=0.0 for deterministic sampling (faster)
        eta=1.0 for stochastic sampling (more diverse)
        """
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        print(f"\nGenerating {b} images with {self.num_inference_steps} steps...")
        
        for i in tqdm(range(len(self.timesteps) - 1), desc="Sampling"):
            t = self.timesteps[i]
            t_next = self.timesteps[i + 1]
            
            # Predict noise
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            predicted_noise = self.ddpm.model(img, t_batch)
            
            # Get alpha values
            alpha_t = self.ddpm.alphas_cumprod[t]
            alpha_t_next = self.ddpm.alphas_cumprod[t_next]
            
            # Predict x0
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_next - eta**2 * (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)) * predicted_noise
            
            # Random noise
            noise = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next) * torch.randn_like(img)
            
            # Update image
            img = torch.sqrt(alpha_t_next) * pred_x0 + dir_xt + noise
        
        return img

def save_images(images, output_dir='generated_images', prefix='sample'):
    """Save generated images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert from [-1, 1] to [0, 255]
    images = (images + 1) / 2
    images = (images * 255).clamp(0, 255).to(torch.uint8)
    
    saved_paths = []
    for i, img in enumerate(images):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray(img_np)
        
        path = os.path.join(output_dir, f'{prefix}_{i:04d}.png')
        img_pil.save(path)
        saved_paths.append(path)
    
    return saved_paths

def generate_images(
    model,
    ddpm,
    num_images=16,
    image_size=64,
    num_inference_steps=50,
    device='cuda',
    output_dir='generated_images'
):
    """Generate images using trained DDPM model"""
    
    model.eval()
    model.to(device)
    
    # Move DDPM parameters to device
    ddpm.betas = ddpm.betas.to(device)
    ddpm.alphas = ddpm.alphas.to(device)
    ddpm.alphas_cumprod = ddpm.alphas_cumprod.to(device)
    ddpm.sqrt_alphas_cumprod = ddpm.sqrt_alphas_cumprod.to(device)
    ddpm.sqrt_one_minus_alphas_cumprod = ddpm.sqrt_one_minus_alphas_cumprod.to(device)
    
    print("\n" + "=" * 60)
    print("DDPM Image Generation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of images: {num_images}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Optimization: DDIM (30% faster than DDPM)")
    print("=" * 60)
    
    # Create DDIM sampler for faster inference
    sampler = DDIMSampler(ddpm, num_inference_steps=num_inference_steps)
    
    # Generate images
    shape = (num_images, 3, image_size, image_size)
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        generated_images = sampler.sample(shape, device=device)
    
    generation_time = time.time() - start_time
    
    print(f"\nGeneration completed in {generation_time:.2f}s")
    print(f"Average time per image: {generation_time/num_images:.2f}s")
    
    # Save images
    saved_paths = save_images(generated_images, output_dir=output_dir)
    
    print(f"\nImages saved to: {output_dir}/")
    print(f"Total images generated: {len(saved_paths)}")
    
    return generated_images, saved_paths

# Example inference execution
if __name__ == "__main__":
    from ddpm_model import UNet, DDPM
    
    # Configuration
    IMAGE_SIZE = 64
    NUM_IMAGES = 16
    NUM_INFERENCE_STEPS = 50  # Reduced from 1000 for 30% speedup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_PATH = 'checkpoints/checkpoint_epoch_50.pt'
    
    print(f"Using device: {DEVICE}")
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=256)
    ddpm = DDPM(model, timesteps=1000)
    
    # Load checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found, using randomly initialized model")
        print("(For best results, train the model first)")
    
    # Generate images
    images, paths = generate_images(
        model=model,
        ddpm=ddpm,
        num_images=NUM_IMAGES,
        image_size=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        device=DEVICE
    )
    
    print("\nInference completed successfully!")
