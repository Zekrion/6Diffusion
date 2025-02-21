input_file = r"D:\CodingProjects\Python\6Diffusion\data\responsive-addresses.txt"
output_file = r"D:\CodingProjects\Python\6Diffusion\data\sample.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for _ in range(100000):
        line = infile.readline()
        if not line:  # Stop if file has fewer than 100,000 lines
            break
        outfile.write(line)