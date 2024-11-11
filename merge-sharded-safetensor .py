import torch
from safetensors.torch import load_file, save_file
import os
import argparse
import re

# Settings
FIRST_SHARD_PATH = r"diffusion_pytorch_model-00001-of-00003.safetensors"  # Default first shard path, relative to this script's directory. You can also use an absolute file path
OUTPUT_MODEL_NAME = ""  # Optional model name for output (leave empty to use the input name)
OVERWRITE_EXISTING = False  # Set to True to allow overwriting the output model file if it already exists
PURGE_SHARD_AFTER_LOADING = True  # Set to True to free memory (purge) after each shard is loaded and combined

# Function to find all shard files starting from the first shard
def find_shard_files(first_shard):
    base_path, base_filename = os.path.split(first_shard)
    if not base_path: # Use current directory if no base path is provided
        base_path = os.getcwd()
    base_filename_prefix = base_filename.split("-")[0]
    
    shard_files = []
    shard_index = 1
    
    # Check for shard files dynamically based on the naming pattern
    while True:
        shard_file_pattern = f"{base_filename_prefix}-{str(shard_index).zfill(5)}-of-*.safetensors"
        matched_files = [f for f in os.listdir(base_path) if re.match(shard_file_pattern.replace('*', '.*'), f)]
        
        if matched_files:
            shard_files.extend([os.path.join(base_path, f) for f in matched_files])
            shard_index += 1
        else:
            break
    
    shard_files.sort()
    print(f"\nFound {len(shard_files)} shard files\n")
    return shard_files


# Parse optional arguments for the first shard and output model name
parser = argparse.ArgumentParser(description="Combine sharded model into one file")
parser.add_argument("--first-shard", type=str, help="Path to the first shard file")
parser.add_argument("--output-model-name", type=str, help="Optional output model name (defaults to first shard name without '-x-of-y')")
parser.add_argument("--overwrite", action="store_true", help="Flag to allow overwriting the existing output model file")
parser.add_argument("--purge-shard", action="store_true", help="Flag to purge (free) memory after loading each shard")
args = parser.parse_args()

# Override FIRST_SHARD_PATH with command-line argument if provided
if args.first_shard:
    FIRST_SHARD_PATH = args.first_shard

# Override OUTPUT_MODEL_NAME if provided, otherwise default behavior
if args.output_model_name:
    OUTPUT_MODEL_NAME = args.output_model_name
else:
    base_path, base_filename = os.path.split(FIRST_SHARD_PATH)
    match = re.match(r"^(.*?)-\d{5}-of-\d{5}", base_filename)
    if match:
        OUTPUT_MODEL_NAME = match.group(1)
    else:
        OUTPUT_MODEL_NAME = base_filename

# Handle --overwrite argument. Set to True if passed, otherwise default to the existing variable
if args.overwrite:
    OVERWRITE_EXISTING = True

# Handle --purge-shard argument: Set to True if passed
if args.purge_shard:
    PURGE_SHARD_AFTER_LOADING = True

# Ensure FIRST_SHARD_PATH uses forward slashes for compatibility
FIRST_SHARD_PATH = FIRST_SHARD_PATH.replace("\\", "/")

# Find all shard files based on the first shard path
shard_files = find_shard_files(FIRST_SHARD_PATH)

# Check if shard files were found
if not shard_files:
    print(f"Error: No shard files found for the provided first shard: {FIRST_SHARD_PATH}")
    exit(1)

# Define the output path for the single combined file
output_path = f"{OUTPUT_MODEL_NAME}.safetensors"

# Check if the file already exists
if os.path.exists(output_path):
    if OVERWRITE_EXISTING:
        print(f"\nFile {output_path} already exists. Overwriting...\n")
    else:
        print(f"\nError: {output_path} already exists. Set --overwrite to overwrite, or change the variable in the Settings-section of the script\n")
        exit(1)

# Load each shard and combine into a single state dictionary
combined_state_dict = {}
print("\nLoading and combining shard files...\n")
for idx, shard in enumerate(shard_files, 1):
    print(f"  Loading shard {idx}/{len(shard_files)}: {shard}")
    
    # Load the state dict from the shard file
    try:
        state_dict = load_file(shard)
    except Exception as e:
        print(f"    Error loading shard {shard}: {e}")
        continue
    
    combined_state_dict.update(state_dict)
    print(f"    Shard {idx}/{len(shard_files)} loaded and combined.")
    
    # Optionally purge memory after loading each shard
    if PURGE_SHARD_AFTER_LOADING:
        del state_dict
        torch.cuda.empty_cache()
        print(f"    Shard {idx} purged from memory.")

# Check if any tensors were added to the combined_state_dict
if len(combined_state_dict) == 0:
    print("\nError: No tensors were combined. The model might not have loaded properly.\n")
    exit(1)

print("\nSaving the combined model to a single .safetensors file...\n")

# Save the combined state dictionary as a single .safetensors file
try:
    save_file(combined_state_dict, output_path)
    print(f"\nCombined model saved to {output_path}\n")
except Exception as e:
    print(f"\nError saving the model: {e}\n")
    exit(1)
