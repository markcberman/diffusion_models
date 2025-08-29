import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
ITERATIONS_PER_EPOCH = 1000
LEARNING_RATE = 2e-4
BETA_START = 0.0001
BETA_END = 0.02
NUM_TIMESTEPS = 1000
IMG_SIZE = 32
CHANNELS = 3

# DDIM parameters
ETA = 0.0  # Deterministic sampling
ALPHA_BAR = 0.5  # DDIM alpha parameter

class SinusoidalPositionEmbeddings(nn.Module):
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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Initial convolution
        self.conv0 = nn.Conv2d(c_in, 64, 3, padding=1)
        
        # Downsampling
        self.down1 = Block(64, 128, time_dim)
        self.down2 = Block(128, 256, time_dim)
        self.down3 = Block(256, 512, time_dim)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Upsampling
        self.up1 = Block(512, 256, time_dim, up=True)
        self.up2 = Block(256, 128, time_dim, up=True)
        self.up3 = Block(128, 64, time_dim, up=True)
        
        # Final convolution
        self.output = nn.Conv2d(64, c_out, 1)
        
    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)
        
        # Initial conv
        x0 = self.conv0(x)
        
        # Downsampling
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)
        x3 = self.down3(x2, t)
        
        # Bottleneck
        x3 = self.bottleneck(x3)
        
        # Upsampling with skip connections
        x = self.up1(x3, t)
        x = self.up2(x, t)
        x = self.up3(x, t)
        
        return self.output(x)

class DDIM:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear schedule for beta
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # DDIM specific parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x, t, t_index, eta=0.0):
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_cumprod_t = 1.0 / torch.sqrt(self.alphas_cumprod[t_index])
        
        # Model prediction
        model_output = model(x, t)
        pred_epsilon = model_output
        
        # DDIM sampling
        pred_x_start = (x - sqrt_one_minus_alphas_cumprod_t * pred_epsilon) * sqrt_recip_alphas_cumprod_t
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x)
            sigma_t = eta * torch.sqrt(betas_t)
            pred_x_start = pred_x_start + sigma_t * noise
            
        return pred_x_start
    
    @torch.no_grad()
    def sample(self, model, shape, eta=0.0):
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Sample from t=T to t=0
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, eta)
            
        return x

def train_epoch(model, dataloader, optimizer, diffusion, device):
    model.train()
    total_loss = 0
    
    for i, (images, _) in enumerate(dataloader):
        if i >= ITERATIONS_PER_EPOCH:
            break
            
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        
        # Add noise
        noise = torch.randn_like(images)
        noisy_images = diffusion.q_sample(images, t, noise)
        
        # Predict noise
        predicted_noise = model(noisy_images, t)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"  Iteration {i}, Loss: {loss.item():.6f}")
    
    return total_loss / min(len(dataloader), ITERATIONS_PER_EPOCH)

def main():
    # Data loading
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Initialize model and diffusion
    model = UNet().to(device)
    diffusion = DDIM()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training for {NUM_EPOCHS} epochs with {ITERATIONS_PER_EPOCH} iterations per epoch")
    print(f"Total iterations: {NUM_EPOCHS * ITERATIONS_PER_EPOCH}")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        avg_loss = train_epoch(model, dataloader, optimizer, diffusion, device)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.6f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'ddim_model_epoch_{epoch + 1}.pth')
            print(f"Model saved at epoch {epoch + 1}")
    
    # Final model save
    torch.save(model.state_dict(), 'ddim_model_final.pth')
    print("Training completed! Final model saved.")
    
    # Sampling
    print("\nGenerating samples from trained model...")
    model.eval()
    
    # Generate 16 samples
    samples = diffusion.sample(model, (16, CHANNELS, IMG_SIZE, IMG_SIZE), eta=ETA)
    
    # Denormalize samples
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Display samples
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = samples[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('ddim_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Samples generated and saved as 'ddim_samples.png'")

if __name__ == "__main__":
    main()
