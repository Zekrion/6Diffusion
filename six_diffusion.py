import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class SixDiffusion:
    def __init__(self, T=1000, beta_start=1e-6, beta_end=0.01):
        """
        Initialize the diffusion model.
        :param T: Number of diffusion steps
        :param beta_start: Start of noise schedule
        :param beta_end: End of noise schedule
        """
        self.T = T
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transform_ipv6(self, ipv6_address):
        """
        Convert an IPv6 address to a word vector.
        :param ipv6_address: IPv6 address in standard format
        :return: List of hexadecimal characters as numeric vectors
        """
        hex_rep = ipv6_address.replace(":", "")
        return [int(char, 16) for char in hex_rep]

    def forward_diffusion(self, x0, t):
        """
        Perform forward diffusion by adding noise.
        :param x0: Original vector
        :param t: Diffusion step
        :return: Noisy vector
        """
        noise = np.random.normal(0, 1, len(x0))
        xt = np.sqrt(self.alpha_cumprod[t]) * np.array(x0) + np.sqrt(1 - self.alpha_cumprod[t]) * noise
        return xt

    def reverse_process(self, xt, model, t):
        """
        Perform reverse process (denoising).
        :param xt: Noisy vector
        :param model: Trained denoising model
        :param t: Reverse diffusion step
        :return: Denoised vector
        """
        with torch.no_grad():
            xt_tensor = torch.tensor(xt, dtype=torch.float32).to(self.device)
            denoised = model(xt_tensor, torch.tensor([t]).to(self.device))
        return denoised.cpu().numpy()

# Example usage
if __name__ == "__main__":
    six_diffusion = SixDiffusion()
    ipv6_sample = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    word_vector = six_diffusion.transform_ipv6(ipv6_sample)
    noisy_vector = six_diffusion.forward_diffusion(word_vector, t=500)
    print("Original Vector:", word_vector)
    print("Noisy Vector:", noisy_vector)