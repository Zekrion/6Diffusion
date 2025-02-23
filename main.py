import torch
import numpy as np
from diffusion.six_diffusion import SixDiffusion
import ipaddress

from torch.utils.data import TensorDataset, Subset

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
                print(f"Skipping invalid line: {addr}")
                continue

            try:
                full_ipv6 = str(ipaddress.IPv6Address(addr).exploded)  # Expand compressed IPv6
                ipv6_addresses.append(full_ipv6)
            except ValueError:
                print(f"Error: Invalid IPv6 address format: {addr}")

    # Convert IPv6 addresses into tokenized integer representation
    def transform_ipv6_to_tokens(ipv6_addr):
        hex_rep = ipv6_addr.replace(":", "").lower()  # Remove colons & lowercase
        try:
            tokens = [int(ch, 16) for ch in hex_rep]  # Convert each hex digit to int (0-15)
            normalized_tokens = [(token / 7.5) - 1 for token in tokens]  # Scale to [-1, 1]
        except ValueError:
            print(f"Error: Invalid characters in IPv6 address {ipv6_addr}")
            return None
        return normalized_tokens

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
    
    # Set seed for reproducibility
    torch.manual_seed(42)

    train_dataset = torch.utils.data.TensorDataset(dataset)

    ### Subset Dataset #####################################################
    
    # Get the number of samples in the dataset
    num_samples = len(train_dataset)

    # Generate random indices with a fixed seed
    number_samples = 1000
    random_indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))[:number_samples]

    # Create a subset with the selected indices
    subset_dataset = Subset(train_dataset, random_indices)
    
    ### Toy Dataset ##########################################################
    
    sample_address = [5] * 32  # creates a list [1, 1, ..., 1] of length 32

    # Create a list of 1000 copies of the sample address
    toy_list = [sample_address for _ in range(1000)]

    # Convert the list to a tensor of shape (1000, 32)
    sample_tensor = torch.tensor(toy_list, dtype=torch.float32)

    # Create the TensorDataset
    toy_dataset = TensorDataset(sample_tensor)
    
    ### Training and Inference ################################################

    # Initialize diffusion model
    diffusion_model = SixDiffusion(T=2000, d_model=512)

    # Train model
    print("Starting Training...")
    diffusion_model.fit(subset_dataset, epochs=30, lr=0.001, batch_size=25)

    # Generate new IPv6 addresses
    print("\nGenerating IPv6 Addresses...")
    generated_samples = diffusion_model.sample(num_samples=1)

    # Print generated IPv6 addresses
    for i, sample in enumerate(generated_samples):
        ipv6_addr = tokens_to_ipv6(sample.cpu().detach().numpy())
        print(f"Generated IPv6 #{i+1}: {ipv6_addr}")

### === Run the Script === ###
if __name__ == "__main__":
    main()