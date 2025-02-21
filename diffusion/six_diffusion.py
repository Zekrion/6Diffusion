# diffusion/six_diffusion.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.ipv6_denoiser import IPv6Denoiser

class SixDiffusion:
    """
    The diffusion “wrapper” that handles:
      - forward_diffusion (adds noise)
      - reverse sampling
      - training (MSE with x_0)
    """
    def __init__(self,
                 T=2000,
                 beta_start=1e-6,
                 beta_end=0.01,
                 d_model=512):
        self.T = T
        self.betas = np.linspace(beta_start, beta_end, T).astype(np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IPv6Denoiser(d_model=d_model).to(self.device)

    def transform_ipv6_to_tokens(self, ipv6_addr):
        """
        Convert e.g. "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        into [2,0,0,1, 0,d,b,8, ...] => decimal ints in [0..15].
        """
        hex_rep = ipv6_addr.replace(":", "")
        tokens = [int(ch, 16) for ch in hex_rep]
        return tokens

    def forward_diffusion(self, x0_tokens, t):
        """
        x0_tokens: shape (32,) in [0..15]
        returns x_t as float shape (32,)
        """
        x0 = x0_tokens.astype(np.float32)
        noise = np.random.randn(32).astype(np.float32)
        alpha_bar = self.alpha_cumprod[t]
        xt = np.sqrt(alpha_bar)*x0 + np.sqrt(1-alpha_bar)*noise
        return xt

    def forward_diffusion_batch(self, x0_tokens_batch, t_batch):
        """
        Vectorized approach
        x0_tokens_batch: (B,32) in [0..15]
        t_batch: (B,)
        returns: (B,32) float
        """
        B = x0_tokens_batch.shape[0]
        x0 = x0_tokens_batch.astype(np.float32)
        noise = np.random.randn(B,32).astype(np.float32)

        alpha_bars = self.alpha_cumprod[t_batch]  # shape (B,)
        alpha_bars = alpha_bars[:,None]           # (B,1)
        xt = np.sqrt(alpha_bars)*x0 + np.sqrt(1-alpha_bars)*noise
        return xt

    def training_step(self, x0_tokens_batch):
        """
        1) Sample t in [0..T-1]
        2) x_t = forward_diffusion
        3) model(x_t_approx, t)-> x0_pred
        4) MSE with original x0_tokens
        """
        B = x0_tokens_batch.shape[0]
        t_batch = np.random.randint(0, self.T, size=(B,))
        x_t_batch = self.forward_diffusion_batch(x0_tokens_batch, t_batch)  # (B,32)

        # clamp/round x_t to [0..15] for discrete tokens
        approx_tokens = np.clip(np.round(x_t_batch), 0, 15).astype(np.int64)

        x_t_tensor = torch.tensor(approx_tokens, dtype=torch.long, device=self.device)
        t_tensor   = torch.tensor(t_batch, dtype=torch.long, device=self.device)
        x0_target  = torch.tensor(x0_tokens_batch, dtype=torch.float32, device=self.device)

        x0_pred = self.model(x_t_tensor, t_tensor)  # (B,32) float
        # MSE vs the original tokens (cast to float)
        loss = ((x0_pred - x0_target)**2).mean()
        return loss

    def fit(self, train_dataset, epochs=10, lr=1e-3, batch_size=512):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        num_samples = len(train_dataset)
        num_batches = (num_samples + batch_size - 1)//batch_size

        for ep in range(epochs):
            perm = np.random.permutation(num_samples)
            running_loss = 0.0
            for i in range(num_batches):
                idx = perm[i*batch_size:(i+1)*batch_size]
                x0_batch = train_dataset[idx]  # shape (B,32)

                loss = self.training_step(x0_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss/num_batches
            print(f"[Epoch {ep+1}/{epochs}] loss={avg_loss:.4f}")

    def sample_ipv6(self, n_samples=1, skip_step=5):
        """
        Start from x_T ~ N(0,I), do steps down to 0 in increments of skip_step
        Return final tokens in shape (n_samples,32)
        """
        self.model.eval()
        x_t = np.random.randn(n_samples, 32).astype(np.float32)
        timesteps = list(range(self.T-1, -1, -skip_step))
        if timesteps[-1] != 0:
            timesteps.append(0)

        with torch.no_grad():
            for t in timesteps:
                # clamp to [0..15] to create discrete approx
                approx_tokens = np.clip(np.round(x_t), 0, 15).astype(np.int64)

                x_t_tensor = torch.tensor(approx_tokens, dtype=torch.long, device=self.device)
                t_tensor   = torch.tensor([t]*n_samples, dtype=torch.long, device=self.device)

                x0_pred = self.model(x_t_tensor, t_tensor)  # shape (n_samples,32)

                alpha_bar = self.alpha_cumprod[t] if t>0 else 1.0
                # For brevity, do a simplified DDIM formula (no random noise):
                # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
                # we skip details. This is a minimal illustration.
                if t>0:
                    alpha_bar_prev = self.alpha_cumprod[t-1]
                else:
                    alpha_bar_prev = 1.0

                sqrt_abp = np.sqrt(alpha_bar_prev)
                x_t = sqrt_abp * x0_pred.cpu().numpy()

        # clamp final to [0..15]
        final_tokens = np.clip(np.round(x_t), 0, 15).astype(np.int64)
        return final_tokens