import torch
import numpy as np
from diffusion.six_diffusion import SixDiffusion
import ipaddress

def load_ipv6_dataset(file_path):
    """
    Reads IPv6 addresses from a .txt file, expands them, and converts them into a tensor dataset.
    Skips invalid lines and logs warnings.
    """
    ipv6_addresses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            addr = line.strip()
            if not addr or "#" in addr:  # Skip invalid lines
                print(f"⚠️ Skipping invalid line: {addr}")
                continue

            try:
                full_ipv6 = str(ipaddress.IPv6Address(addr).exploded)  # Expand compressed IPv6
                ipv6_addresses.append(full_ipv6)
            except ValueError:
                print(f"❌ Error: Invalid IPv6 address format: {addr}")

    # Convert IPv6 addresses into tokenized integer representation
    def transform_ipv6_to_tokens(ipv6_addr):
        hex_rep = ipv6_addr.replace(":", "").lower()  # Remove colons & lowercase
        try:
            tokens = [int(ch, 16) for ch in hex_rep]  # Convert each hex digit to int (0-15)
        except ValueError:
            print(f"❌ Error: Invalid characters in IPv6 address {ipv6_addr}")
            return None
        return tokens

    tokenized_data = [transform_ipv6_to_tokens(addr) for addr in ipv6_addresses]
    tokenized_data = [t for t in tokenized_data if t is not None]  # Remove None values

    if not tokenized_data:
        raise ValueError("No valid IPv6 addresses found in the dataset!")

    return torch.tensor(tokenized_data, dtype=torch.float32)

def tokens_to_ipv6(tokens):
    """
    Converts a list of 32 integer tokens (nibbles) back to an IPv6 string.
    """
    hex_string = "".join([hex(int(t))[2:] for t in tokens])  # Convert to hex
    hex_string = hex_string.zfill(32)  # Ensure it’s always 32 nibbles

    ipv6_words = [hex_string[i:i+4] for i in range(0, len(hex_string), 4)]  # Split into 8 groups
    ipv6_str = ":".join(ipv6_words)

    # Compress to shortest IPv6 representation
    return str(ipaddress.IPv6Address(ipv6_str))

### === Main Function === ###
def main():
    # Load dataset
    dataset = load_ipv6_dataset("data/sample.txt")

    train_dataset = torch.utils.data.TensorDataset(dataset)

    # Initialize diffusion model
    diffusion_model = SixDiffusion(T=2000, d_model=512)

    # Train model
    print("Starting Training...")
    diffusion_model.fit(train_dataset, epochs=10, lr=1e-3, batch_size=512)

    # Generate new IPv6 addresses
    print("\nGenerating IPv6 Addresses...")
    generated_samples = diffusion_model.sample_ipv6(num_samples=5)

    # Print generated IPv6 addresses
    for i, sample in enumerate(generated_samples):
        ipv6_addr = tokens_to_ipv6(sample.cpu().numpy())
        print(f"Generated IPv6 #{i+1}: {ipv6_addr}")

### === Run the Script === ###
if __name__ == "__main__":
    main()