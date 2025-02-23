import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time

from glf_msa_decoder.decoder import IPv6Decoder


class SixDiffusion:
    def __init__(self, T=200, beta_start=0.0001, beta_end=0.02, d_model=512, schedule_type='cosine', schedule_offset=0.008):
        self.T = T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Schedule configuration
        if schedule_type == 'cosine':
            self._setup_cosine_schedule(schedule_offset)
        else:  # linear
            self._setup_linear_schedule(beta_start, beta_end)
        
        # Precompute coefficients
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        
        # Initialize model
        self.model = IPv6Decoder(d_model=d_model).to(self.device)

    def _setup_linear_schedule(self, beta_start, beta_end):
        """Linear noise schedule"""
        self.betas = torch.linspace(beta_start, beta_end, self.T, device=self.device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_cumprod[:-1]])

    def _setup_cosine_schedule(self, s=0.008):
        """Cosine noise schedule (improved stability)"""
        steps = self.T + 1
        x = torch.linspace(0, self.T, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / self.T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1.0
        
        # Derive actual schedule parameters
        self.alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.betas = 1. - self.alphas
        self.alpha_cumprod = alphas_cumprod[1:]  # Align with T steps
        self.alpha_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_cumprod[:-1]])

        # Move all to device
        self.alphas = self.alphas.to(self.device)
        self.betas = self.betas.to(self.device)
        self.alpha_cumprod = self.alpha_cumprod.to(self.device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(self.device)

    def forward_diffusion(self, x0, t):
        """Combined forward process for both schedules"""
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.alpha_cumprod[t][:, None].sqrt()
        sqrt_one_minus_alpha_cumprod = (1. - self.alpha_cumprod[t][:, None]).sqrt()
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise
    
    def training_step(self, x0_batch):
        B = x0_batch.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        
        x_t, true_noise = self.forward_diffusion(x0_batch, t)
        pred_noise = self.model(x_t, t)
        
        # Combine losses more efficiently
        mse_loss = F.mse_loss(pred_noise, true_noise)
        x0_pred = (x_t - self.sqrt_one_minus_alphas_cumprod[t][:, None] * pred_noise) / self.sqrt_alphas_cumprod[t][:, None]
        recon_loss = F.mse_loss(x0_pred, x0_batch)
        
        print("Loss: ", mse_loss, recon_loss)
        
        return mse_loss + recon_loss

    def fit(self, train_dataset, epochs=200, lr=0.001, batch_size=128):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for x0_batch, in pbar:
                    x0_batch = x0_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    loss = self.training_step(x0_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
            
            print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")

    @torch.no_grad()
    def sample(self, num_samples, return_all_steps=False):
        self.model.eval()
        x = torch.randn(num_samples, 32, device=self.device)
        steps = []
        
        for t in reversed(range(1, self.T)):
            t_batch = torch.full((num_samples,), t, device=self.device)
            pred_noise = self.model(x, t_batch)
            
            # Use precomputed parameters for reverse process
            x = self._reverse_step(x, t, pred_noise)
            
            if return_all_steps:
                steps.append(self.reverse_normalize(x.clone()))
        
        return steps if return_all_steps else self.reverse_normalize(x)

    def _reverse_step(self, x, t, pred_noise):
        beta_t = self.betas[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Main reverse step calculation
        x = sqrt_recip_alpha_t * (x - beta_t * pred_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if t > 0:
            x += torch.sqrt(self.posterior_variance[t-1]) * torch.randn_like(x)
        
        return x

    @staticmethod
    def reverse_normalize(tensor):
        return torch.clamp(((tensor + 1) * 7.5).round(), 0, 15).int()