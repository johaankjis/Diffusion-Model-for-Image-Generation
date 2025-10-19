"""
DDPM Model Implementation
Denoising Diffusion Probabilistic Model for high-quality image generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(F.relu(self.norm1(x)))
        time_emb = F.relu(self.time_mlp(time_emb))
        h = h + time_emb[:, :, None, None]
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.residual_conv(x)

class UNet(nn.Module):
    """U-Net architecture for DDPM"""
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64, time_emb_dim)
        self.enc2 = ResidualBlock(64, 128, time_emb_dim)
        self.enc3 = ResidualBlock(128, 256, time_emb_dim)
        self.enc4 = ResidualBlock(256, 512, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(512, 512, time_emb_dim)
        
        # Decoder
        self.dec4 = ResidualBlock(512 + 512, 256, time_emb_dim)
        self.dec3 = ResidualBlock(256 + 256, 128, time_emb_dim)
        self.dec2 = ResidualBlock(128 + 128, 64, time_emb_dim)
        self.dec1 = ResidualBlock(64 + 64, 64, time_emb_dim)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Encoder
        e1 = self.enc1(x, t)
        e2 = self.enc2(self.pool(e1), t)
        e3 = self.enc3(self.pool(e2), t)
        e4 = self.enc4(self.pool(e3), t)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4), t)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1), t)
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1), t)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1), t)
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1), t)
        
        return self.out(d1)

class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """Calculate training loss"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t):
        """Single reverse diffusion step"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape, device='cuda'):
        """Generate samples using reverse diffusion"""
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        
        return img

# Example usage
if __name__ == "__main__":
    print("DDPM Model Implementation")
    print("=" * 50)
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=256)
    ddpm = DDPM(model, timesteps=1000)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Timesteps: {ddpm.timesteps}")
    print(f"Beta range: [{ddpm.betas[0]:.6f}, {ddpm.betas[-1]:.6f}]")
    print("\nModel ready for training!")
