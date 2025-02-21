# main.py

import numpy as np
from diffusion.six_diffusion import SixDiffusion

def main():
    # example addresses
    sample_addresses = [
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        "2400:cb00:2048:1::c629:d7a2",
        "2001:4860:4860::8888",
        # ...
    ]
    # init diffusion model
    diffusion = SixDiffusion(T=2000, beta_start=1e-6, beta_end=0.01, d_model=512)

    # convert them to tokens
    tokenized = []
    for addr in sample_addresses:
        tokens = diffusion.transform_ipv6_to_tokens(addr)
        tokenized.append(tokens)
    tokenized = np.array(tokenized, dtype=np.int64)  # shape (N,32)

    # do a small training run
    diffusion.fit(train_dataset=tokenized, epochs=2, lr=1e-3, batch_size=2)

    # sample
    final_tokens = diffusion.sample_ipv6(n_samples=2, skip_step=5)
    print("Sampled final tokens:\n", final_tokens)

if __name__ == "__main__":
    main()