#!/usr/bin/env python3
import subprocess
import os

# --------------------------
# Define input/output files (use relative paths safely)
# --------------------------
input_file = "../data/Sample_Dataset.csv"         # Path to your input dataset
output_file = "../data/workload_dataset_sample.csv"  # Path to save preprocessed dataset

# Default machine configuration
cores = 16
mem_mb = 65536  # 64 GB

# --------------------------
# Build the command string
# --------------------------
command = f'python preprocess.py --input "{input_file}" --output "{output_file}" --cores {cores} --mem_mb {mem_mb}'

# --------------------------
# Run the command
# --------------------------
print(f"🚀 Running command:\n{command}\n")

result = subprocess.run(command, shell=True, capture_output=True, text=True)

# --------------------------
# Print results
# --------------------------
if result.returncode == 0:
    print("✅ Command executed successfully!\n")
    print("----- OUTPUT -----")
    print(result.stdout)
else:
    print("❌ Command failed!\n")
    print("----- ERROR -----")
    print(result.stderr)

print(f"\nExit Code: {result.returncode}")
