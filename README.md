# Merge Sharded Safetensors
This tool quickly and easily merges safetensor files that are split into parts.

If you run across a .safetensors model that is split into pieces, and you want to combine it, this tool might be able to do it for you.

![image](https://github.com/user-attachments/assets/359a095f-0080-4693-8f60-e6eabf77c6bf)


#### Input
> diffusion_pytorch_model-00001-of-00003.safetensors
> 
> diffusion_pytorch_model-00002-of-00003.safetensors
> 
> diffusion_pytorch_model-00003-of-00003.safetensors

#### Output
> diffusion_pytorch_model.safetensors

# Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/merge-sharded-safetensor`
2. (Optional) Create a virtual environment for your setup. Feel free to use the `venv_create.bat` for a simple windows setup.
3. Activate your venv.
4. Run `pip install -r requirements.txt` (this is done automatically with the `venv_create.bat`).

# Usage
The tool can be used in 3 ways:
1. Edit `merge-sharded-safetensor.py` and enter the path to the first shard file in the variable `FIRST_SHARD_PATH`. Then run `py merge-sharded-safetensor.py`.
2. Run `py merge-sharded-safetensor.py` with the `FIRST_SHARD_PATH` variable empty. The script will then ask for the path to the first shard path.
3. Run `py merge-sharded-safetensor.py --first-shard "path/to/your/first-shard-file.safetensors"`

