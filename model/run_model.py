#!/usr/bin/env python3
import subprocess
import os

# --------------------------
# Define input/output files (use relative paths safely)
# --------------------------
input_file = "../data/workload_dataset_sample.csv"      # Preprocessed dataset input
model_output = "../model/orchestrator_model.pkl"         # Output trained model

# --------------------------
# Build the command string
# --------------------------
command = f'python train_model.py --input "{input_file}" --output "{model_output}"'

# --------------------------
# Run the command
# --------------------------
print(f" Running model training command:\n{command}\n")

result = subprocess.run(command, shell=True, capture_output=True, text=True)

# --------------------------
# Print results
# --------------------------
if result.returncode == 0:
    print("[OK] Model training executed successfully!\n")
    print("----- OUTPUT -----")
    print(result.stdout)
else:
    print("[ERROR] Model training failed!\n")
    print("----- ERROR -----")
    print(result.stderr)

print(f"\nExit Code: {result.returncode}")
