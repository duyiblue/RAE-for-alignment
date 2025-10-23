# RAE for (Cross-Embodiment) Alignment

This repo is based on [Diffusion Transformers with Representation Autoencoders (RAE)](https://github.com/bytetriper/RAE). For clarity and conciseness, I removed their original README content, but I suggest you read the original one first before you read my updates.

## Overview
Goal: Finetune a (source) encoder model, so that its output aligns with that of another (target) encoder model. The source and target models receive paired inputs, where two images are "identical" in essence (depending how you define essence), but differ in domain or appearance. For example:
- **VOC mask experiment**: Source model receives images overlayed with semi-transparent segmentation maps, while target model receives the corresponding original images.
- **Pose encoder experiment**: Source model receives images of robot embodiment A, while target model receives images of robot embodiment B in the same pose. (Here the "essence" of an image is defined as the pose of the robot arm in the image.)

We adopt the RAE codebase, where DINOv2 serves as the encoder. We can finetune it with LoRA. A key contribution of RAE is that it provides a decoder for DINOv2, which helps us with validation and visualization (how well our alignment is doing).

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
uv pip install --no-deps torchmetrics==1.3.1
uv pip install --no-deps lightning-utilities
uv pip install --no-deps "pytorch-lightning==2.2.*"
```