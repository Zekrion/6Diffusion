import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from glf_msa_decoder.decoder import IPv6Decoder
import time

from tqdm import tqdm

class SixDiffusion:
    def __init__(self, T=2000, beta_start=1e-6, beta_end=0.01, d_model=512):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move precomputed alpha values to device once
        self.alpha_cumprod = self.alpha_cumprod.to(self.device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)

        # Load IPv6 Diffusion Model
        self.model = IPv6Decoder(d_model=d_model).to(self.device)

    def transform_ipv6_to_tokens(self, ipv6_addr):
        """
        Convert an IPv6 address to a sequence of integer tokens (32 tokens for 32 nibbles).
        """
        hex_rep = ipv6_addr.replace(":", "")
        tokens = [int(ch, 16) for ch in hex_rep]

        # Normalize from [0..15] to [-1..+1]
        tokens = (tokens - 7.5) / 7.5
        tokens = tokens.clamp(-1.0, 1.0)

        return tokens

    def forward_diffusion_batch(self, x0_tokens_batch, t_batch):
        """
        Applies forward diffusion to a batch of IPv6 addresses.
        """
        B = x0_tokens_batch.shape[0]
        x0 = x0_tokens_batch.to(torch.float32).to(self.device)
        noise = torch.randn_like(x0, device=self.device)  # [B, 32]

        # ✅ Fix: Remove redundant `torch.tensor()` and use `.clone().detach().to(...)`
        t_batch = t_batch.clone().detach().to(dtype=torch.long, device=self.device)

        alpha_bars = torch.gather(self.alpha_cumprod, 0, t_batch).view(B, 1)

        # Apply diffusion noise
        xt = torch.sqrt(alpha_bars) * x0 + torch.sqrt(1 - alpha_bars) * noise
        return xt, noise

    def compute_true_posterior(self, x0, x_t, t):
        """
        Computes the true posterior q(x_{t-1} | x_t, x0).
        """
        alpha_t = torch.gather(self.alphas, 0, t).view(-1, 1)  # ✅ Reshape to [B,1]
        alpha_t_bar = torch.gather(self.alpha_cumprod, 0, t).view(-1, 1)  # ✅ Reshape to [B,1]

        t_prev = torch.clamp(t - 1, min=0)  # Prevent out-of-bounds
        alpha_t_bar_prev = torch.gather(self.alpha_cumprod, 0, t_prev).view(-1, 1)  # ✅ Reshape to [B,1]
        beta_t = torch.gather(self.betas, 0, t).view(-1, 1)  # ✅ Reshape to [B,1]

        # Now all tensors have shape [B,1] and can broadcast with x0 / x_t [B, 32]
        mu_q = ((torch.sqrt(alpha_t_bar_prev) * beta_t / (1 - alpha_t_bar)) * x0 +
                (torch.sqrt(alpha_t) * (1 - alpha_t_bar_prev) / (1 - alpha_t_bar)) * x_t)
        
        sigma_q = torch.clamp((1 - alpha_t_bar_prev) * beta_t / (1 - alpha_t_bar), min=1e-6)

        return mu_q, sigma_q

    def compute_learned_posterior(self, x_t, t):
        """
        Uses the learned model to estimate the posterior p_theta(x_{t-1} | x_t).
        """
        alpha_t = torch.gather(self.alphas, 0, t).view(-1, 1)  # ✅ Reshape to [B,1]
        alpha_t_bar = torch.gather(self.alpha_cumprod, 0, t).view(-1, 1)  # ✅ Reshape to [B,1]
        beta_t = torch.gather(self.betas, 0, t).view(-1, 1)  # ✅ Reshape to [B,1]
        alpha_t_sqrt = torch.sqrt(alpha_t)
        alpha_bar_sqrt = torch.sqrt(alpha_t_bar)

        predicted_noise = self.model(x_t, t)  # [B, 32]

        # Ensure the predicted x0 uses correctly shaped scalars
        x0_pred = (x_t - torch.sqrt(1 - alpha_t_bar) * predicted_noise) / alpha_bar_sqrt

        t_prev = torch.clamp(t - 1, min=0)  # Avoid out-of-bounds
        alpha_t_bar_prev = torch.gather(self.alpha_cumprod, 0, t_prev).view(-1, 1)  # ✅ Reshape to [B,1]

        # Now, `mu_theta` can correctly broadcast with `[B, 32]`
        mu_theta = ((torch.sqrt(alpha_t_bar_prev) * beta_t / (1 - alpha_t_bar)) * x0_pred +
                    (torch.sqrt(alpha_t) * (1 - alpha_t_bar_prev) / (1 - alpha_t_bar)) * x_t)

        sigma_theta = torch.clamp(beta_t, min=1e-6)

        return mu_theta, sigma_theta, predicted_noise

    def training_step(self, x0_tokens_batch):
        """
        Performs one training step using the paper's two-term loss:
        1) MSE of predicted noise vs. true noise
        2) MSE of x_0 vs. one-step denoised x_1
        """
        B = x0_tokens_batch.shape[0]
        
        # 1. Sample timesteps in [1..T].
        t_batch = torch.randint(1, self.T, (B,), dtype=torch.long, device=self.device)
        
        # 2. Forward diffusion to get x_t and the true noise
        #    'forward_diffusion_batch' should return:
        #       x_t  = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*noise
        #       true_noise = noise
        x_t_batch, true_noise = self.forward_diffusion_batch(x0_tokens_batch, t_batch)
        
        # 3. Model predicts noise from (x_t, t).  That is your “denoising network”
        predicted_noise = self.model(x_t_batch, t_batch)   # shape [B, num_dims]
        
        # -- Term 2 in the paper: noise-prediction MSE.
        kl_like_loss = torch.nn.functional.mse_loss(predicted_noise, true_noise)
        
        # 4. Compute x_1 “reconstruction.”  Typically for t=1, you invert one step:
        #    x0_pred = (x_t - sqrt(1 - alpha_bar_t)*predicted_noise) / sqrt(alpha_bar_t).
        alpha_t_bar = torch.gather(self.alpha_cumprod, 0, t_batch).to(self.device)
        alpha_t_bar_sqrt = torch.sqrt(alpha_t_bar).unsqueeze(-1)        # shape [B,1]
        one_minus_alpha_bar_sqrt = torch.sqrt(1.0 - alpha_t_bar).unsqueeze(-1) 
        
        x0_pred = ( 
            x_t_batch - one_minus_alpha_bar_sqrt * predicted_noise
        ) / alpha_t_bar_sqrt
        
        # -- Term 3 in the paper: MSE( x_0, x_1 )
        #    The paper basically uses x0_pred as the "x_1" that is supposed to match x_0.
        #    (They interpret it as the one-step denoised version).
        x0_target = x0_tokens_batch.float().to(self.device)
        recon_loss = torch.nn.functional.mse_loss(x0_pred, x0_target)
        
        # Final training loss is sum of these two terms
        loss = kl_like_loss + recon_loss
        
        return loss

    def fit(self, train_dataset, epochs=10, lr=1e-3, batch_size=512):
        """
        Trains the model using Adam optimizer with tqdm progress bar.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            start_time = time.time()
            running_loss = 0.0

            # ✅ Wrap data_loader in tqdm to show progress bar
            with tqdm(data_loader, desc=f"Epoch {ep+1}/{epochs}", unit="batch") as pbar:
                for batch in pbar:
                    x0_batch = batch[0]

                    loss = self.training_step(x0_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # ✅ Update tqdm progress bar with live loss value
                    pbar.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(data_loader)
            epoch_time = time.time() - start_time
            print(f"[Epoch {ep+1}/{epochs}] loss={avg_loss:.4f} | time={epoch_time:.2f}s")

    def sample_ipv6(self, num_samples=1):
        """
        Generates IPv6 addresses by reverse diffusion.
        """
        self.model.eval()
        with torch.no_grad():
            x_t = torch.randn(num_samples, 32).to(self.device)  # Start from Gaussian noise
            for t in reversed(range(0, self.T, 5)):  # Sampling every 5 steps for efficiency
                t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=self.device)
                mu_theta, sigma_theta, _ = self.compute_learned_posterior(x_t, t_tensor)
                x_t = mu_theta + torch.sqrt(sigma_theta) * torch.randn_like(x_t)
        return inverse_normalize(x_t)
    
def inverse_normalize(tokens):
    # tokens in [-1..+1], map back to [0..15]
    tokens = (tokens * 7.5) + 7.5
    # clamp to [0..15], round to nearest integer
    tokens = torch.clamp(tokens, 0, 15)
    return torch.round(tokens)