"""
Test script to reconstruct an arbitrary image using RAE encoder-decoder.
Creates a side-by-side comparison of original and reconstructed images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src directory to Python path so relative imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE

from dataset.dataset_factory import get_alignment_dataloader

def load_image_from_path(image_path: Path) -> torch.Tensor:
    """Load image and convert to tensor."""
    image = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor

def load_image_from_dataset(dataset: str = "voc") -> torch.Tensor:
    """Load an image from the dataset."""
    train_loader, _ = get_alignment_dataloader(
        dataset=dataset,
        batch_size=1,
    )
    batch = next(iter(train_loader))
    # Return the target image, which is already in [B, C, H, W] format normalized to [0, 1]
    return batch["target_img"]

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    tensor = tensor.squeeze(0).cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def create_side_by_side(original: torch.Tensor, reconstructed: torch.Tensor) -> Image.Image:
    """Create a side-by-side comparison image."""
    orig_img = tensor_to_pil(original)
    recon_img = tensor_to_pil(reconstructed)
    
    # Ensure both images have the same size
    if orig_img.size != recon_img.size:
        recon_img = recon_img.resize(orig_img.size, Image.BICUBIC)
    
    # Create side-by-side canvas
    width, height = orig_img.size
    combined = Image.new('RGB', (width * 2, height))
    combined.paste(orig_img, (0, 0))
    combined.paste(recon_img, (width, 0))
    
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test RAE reconstruction with side-by-side comparison."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1/pretrained/DINOv2-B.yaml"),
        help="Path to the YAML config with a stage_1 section.",
    ) # For higher resolution, use configs/stage1/pretrained/DINOv2-B_512.yaml
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path to reconstruct (mutually exclusive with --from-dataset).",
    )
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="Load image from dataset instead of file path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",
        choices=["voc", "robot"],
        help="Which dataset to load from (only used with --from-dataset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: outputs/<image_name>_comparison.png).",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Validate mutually exclusive options
    if args.from_dataset and args.image is not None:
        raise ValueError("Cannot specify both --image and --from-dataset")
    if not args.from_dataset and args.image is None:
        raise ValueError("Must specify either --image or --from-dataset")

    # Validate input image if loading from path
    if not args.from_dataset and not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # Set output path
    if args.output is None:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = "comparison.png"
        args.output = output_dir / output_filename
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load RAE model from config
    print(f"Loading RAE model from config: {args.config}")
    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    print("RAE model loaded successfully")

    # Load and process image
    if args.from_dataset:
        print(f"Loading image from {args.dataset} dataset...")
        original = load_image_from_dataset(dataset=args.dataset).to(device)
    else:
        print(f"Loading image: {args.image}")
        original = load_image_from_path(args.image).to(device)
    print(f"Image shape: {tuple(original.shape)}")

    # Encode and decode
    print("Encoding image...")
    with torch.no_grad():
        latent = rae.encode(original)
        print(f"Latent shape: {tuple(latent.shape)}")
        
        print("Decoding latent...")
        reconstructed = rae.decode(latent)
        reconstructed = reconstructed.clamp(0.0, 1.0)
        print(f"Reconstructed shape: {tuple(reconstructed.shape)}")

    # Create side-by-side comparison
    print("Creating side-by-side comparison...")
    comparison = create_side_by_side(original, reconstructed)
    
    # Save result
    comparison.save(args.output)
    print(f"\n✓ Saved comparison to: {args.output.resolve()}")
    print(f"  Original (left) | Reconstructed (right)")


if __name__ == "__main__":
    main()

