#!/usr/bin/env python
"""
Evaluation script for alignment training checkpoints.

Loads a trained alignment checkpoint and evaluates it on a single source image:
- Encodes source image with trained source encoder
- Decodes latent with frozen target decoder
- Saves reconstructed image
- Optionally computes metrics if target image is provided
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from datetime import datetime
from PIL import Image
from torchvision import transforms
import numpy as np

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from torchmetrics.image import StructuralSimilarityIndexMeasure
from disc import LPIPS
from alignment_training import AlignmentTrainer


def load_image_from_path(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Load image from path and convert to tensor.
    
    Args:
        image_path: Path to image file
        device: torch device
        
    Returns:
        Image tensor in shape (1, C, H, W) with values in [0, 1]
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W) in [0, 1]
    return tensor.to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor in shape (1, C, H, W) or (C, H, W)
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def evaluate_single_sample(
    checkpoint_path: Path,
    source_image_path: Path,
    target_image_path: Path | None,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """
    Evaluate alignment checkpoint on a single sample.
    
    Args:
        checkpoint_path: Path to alignment training checkpoint (.ckpt)
        source_image_path: Path to source domain image
        target_image_path: Path to target domain image (optional, for metrics)
        output_dir: Directory to save outputs
        device: torch device
        
    Returns:
        Dictionary with results (metrics if target image provided)
    """
    # Load checkpoint
    print(f"Loading alignment checkpoint from: {checkpoint_path}")
    model = AlignmentTrainer.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    print("✓ Checkpoint loaded successfully")
    
    # Extract models (use full RAE models, not just encoder/decoder)
    source_model = model.source_model
    target_model = model.target_model
    
    # Ensure models are in eval mode (should already be, but make explicit)
    source_model.eval()
    target_model.eval()
    print("✓ Extracted source and target models")
    
    # Load source image
    print(f"Loading source image: {source_image_path}")
    source_image = load_image_from_path(source_image_path, device)
    print(f"  Source image shape: {tuple(source_image.shape)}")
    
    # Load target image if provided
    target_image = None
    if target_image_path is not None:
        print(f"Loading target image: {target_image_path}")
        target_image = load_image_from_path(target_image_path, device)
        print(f"  Target image shape: {tuple(target_image.shape)}")
    
    # Encode source image with source model's encode method
    print("Encoding source image with trained source encoder...")
    with torch.no_grad():
        source_latent = source_model.encode(source_image)
        print(f"  Latent shape: {tuple(source_latent.shape)}")
    
    # Decode with target model's decode method
    print("Decoding latent with frozen target decoder...")
    with torch.no_grad():
        reconstructed = target_model.decode(source_latent)
        reconstructed = reconstructed.clamp(0.0, 1.0)
        print(f"  Reconstructed shape: {tuple(reconstructed.shape)}")
    
    # Resize reconstruction to match source image if needed
    if reconstructed.shape != source_image.shape:
        print(f"Resizing reconstruction from {tuple(reconstructed.shape)} to {tuple(source_image.shape)}")
        reconstructed = F.interpolate(
            reconstructed,
            size=(source_image.shape[2], source_image.shape[3]),
            mode='bicubic',
            align_corners=False
        )
    
    # Save reconstructed image
    print("Saving reconstructed image...")
    recon_image_path = output_dir / "reconstructed.png"
    recon_pil = tensor_to_pil(reconstructed)
    recon_pil.save(recon_image_path)
    print(f"✓ Saved reconstruction to: {recon_image_path}")
    
    # Also save source image for comparison
    source_image_copy_path = output_dir / "source_image.png"
    source_pil = tensor_to_pil(source_image)
    source_pil.save(source_image_copy_path)
    print(f"✓ Saved source image copy to: {source_image_copy_path}")
    
    # Compute metrics if target image provided
    results = {
        "checkpoint_path": str(checkpoint_path),
        "source_image_path": str(source_image_path),
        "target_image_path": str(target_image_path) if target_image_path else None,
        "source_image_shape": list(source_image.shape),
        "latent_shape": list(source_latent.shape),
        "reconstructed_shape": list(reconstructed.shape),
    }
    
    if target_image is not None:
        print("\nComputing metrics against target image...")
        
        # Encode target image to get target latent (for latent loss)
        print("Encoding target image with target encoder...")
        with torch.no_grad():
            target_latent = target_model.encode(target_image)
            print(f"  Target latent shape: {tuple(target_latent.shape)}")
        
        # Resize target to match reconstruction if needed
        if target_image.shape != reconstructed.shape:
            print(f"Resizing target from {tuple(target_image.shape)} to {tuple(reconstructed.shape)}")
            target_image_resized = F.interpolate(
                target_image,
                size=(reconstructed.shape[2], reconstructed.shape[3]),
                mode='bicubic',
                align_corners=False
            )
        else:
            target_image_resized = target_image
        
        # Save target image for comparison
        target_image_copy_path = output_dir / "target_image.png"
        target_pil = tensor_to_pil(target_image_resized)
        target_pil.save(target_image_copy_path)
        print(f"✓ Saved target image copy to: {target_image_copy_path}")
        
        # Create side-by-side comparison
        comparison = create_comparison_grid(source_image, target_image_resized, reconstructed)
        comparison_path = output_dir / "comparison.png"
        comparison.save(comparison_path)
        print(f"✓ Saved comparison grid to: {comparison_path}")
        
        # Compute losses
        with torch.no_grad():
            # Latent loss (MSE between source and target latents)
            latent_loss = F.mse_loss(source_latent, target_latent).item()
            
            # Reconstruction losses
            l1_loss = F.l1_loss(reconstructed, target_image_resized).item()
            l2_loss = F.mse_loss(reconstructed, target_image_resized).item()
            
            # SSIM
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            ssim_value = ssim_metric(reconstructed, target_image_resized).item()
            ssim_loss = 1.0 - ssim_value
            
            # LPIPS
            lpips_metric = LPIPS().to(device).eval()
            lpips_value = lpips_metric(reconstructed, target_image_resized).mean().item()
        
        results.update({
            "metrics": {
                "latent_loss": latent_loss,
                "recon_l1_loss": l1_loss,
                "recon_l2_loss": l2_loss,
                "recon_ssim_value": ssim_value,
                "recon_ssim_loss": ssim_loss,
                "recon_lpips": lpips_value,
            }
        })
        
        print("\nMetrics:")
        print(f"  Latent Loss:  {latent_loss:.6f} (MSE between source and target latents)")
        print(f"  L1 Loss:      {l1_loss:.6f}")
        print(f"  L2 Loss:      {l2_loss:.6f}")
        print(f"  SSIM:         {ssim_value:.6f}")
        print(f"  SSIM Loss:    {ssim_loss:.6f} (1 - SSIM, lower is better)")
        print(f"  LPIPS:        {lpips_value:.6f}")
    
    # Save results to JSON
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {results_path}")
    
    return results


def create_comparison_grid(
    source: torch.Tensor,
    target: torch.Tensor,
    reconstructed: torch.Tensor
) -> Image.Image:
    """
    Create a comparison grid: [Source | Target | Reconstructed]
    
    Args:
        source: Source image tensor
        target: Target image tensor
        reconstructed: Reconstructed image tensor
        
    Returns:
        PIL Image with side-by-side comparison
    """
    source_pil = tensor_to_pil(source)
    target_pil = tensor_to_pil(target)
    recon_pil = tensor_to_pil(reconstructed)
    
    # Ensure all images have the same size
    width, height = source_pil.size
    if target_pil.size != (width, height):
        target_pil = target_pil.resize((width, height), Image.BICUBIC)
    if recon_pil.size != (width, height):
        recon_pil = recon_pil.resize((width, height), Image.BICUBIC)
    
    # Create grid: [Source | Target | Reconstructed]
    combined = Image.new('RGB', (width * 3, height))
    combined.paste(source_pil, (0, 0))
    combined.paste(target_pil, (width, 0))
    combined.paste(recon_pil, (width * 2, 0))
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate alignment training checkpoint on a single sample'
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to alignment training checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--source_image',
        type=Path,
        required=True,
        help='Path to source domain image'
    )
    
    # Optional arguments
    parser.add_argument(
        '--target_image',
        type=Path,
        default=None,
        help='Path to target domain image (optional, for computing metrics)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Output directory (default: outputs/eval_YYYYMMDD_HHMMSS)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    assert args.checkpoint.exists(), f"Checkpoint not found: {args.checkpoint}"
    assert args.source_image.exists(), f"Source image not found: {args.source_image}"
    if args.target_image is not None:
        assert args.target_image.exists(), f"Target image not found: {args.target_image}"
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"eval_{timestamp}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir.resolve()}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run evaluation
    print("\n" + "="*60)
    print("ALIGNMENT CHECKPOINT EVALUATION")
    print("="*60 + "\n")
    
    results = evaluate_single_sample(
        checkpoint_path=args.checkpoint,
        source_image_path=args.source_image,
        target_image_path=args.target_image,
        output_dir=args.output_dir,
        device=device,
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"All outputs saved to: {args.output_dir.resolve()}")
    print("="*60)


if __name__ == '__main__':
    main()

