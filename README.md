# RAE for (Cross-Embodiment) Alignment

This repo is based on [Diffusion Transformers with Representation Autoencoders (RAE)](https://github.com/bytetriper/RAE). For clarity and conciseness, I removed their original README content, but I suggest you read the original one first before you read my updates.

## Setup Guide
```
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

# Install PyTorch 2.2.0 with CUDA 12.1
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb
uv pip install "numpy<2" transformers einops omegaconf

# The following steps are additional steps for our alignment training codebase
pip install -U albumentations
pip install -U datasets
pip install matplotlib
python -m pip install "numpy<2" --upgrade
```