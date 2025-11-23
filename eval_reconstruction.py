"""
Evaluate RAE reconstruction quality on entire dataset.
Computes L1, L2, SSIM, and LPIPS metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src directory to Python path so relative imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE
from disc import LPIPS

from dataset.dataset_factory import get_alignment_dataloader

class MetricsAccumulator:
    """Accumulate metrics over batches."""
    
    def __init__(self):
        self.l1_sum = 0.0
        self.l2_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.count = 0
    
    def update(self, l1: float, l2: float, ssim: float, lpips: float, batch_size: int):
        self.l1_sum += l1 * batch_size
        self.l2_sum += l2 * batch_size
        self.ssim_sum += ssim * batch_size
        self.lpips_sum += lpips * batch_size
        self.count += batch_size
    
    def get_averages(self) -> dict[str, float]:
        assert self.count > 0, "No samples processed"
        return {
            "L1": self.l1_sum / self.count,
            "L2": self.l2_sum / self.count,
            "SSIM": self.ssim_sum / self.count,
            "LPIPS": self.lpips_sum / self.count,
        }


def evaluate_reconstruction(
    rae: RAE,
    lpips_fn: LPIPS,
    ssim_fn: StructuralSimilarityIndexMeasure,
    dataloader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate reconstruction on entire dataset.
    
    Args:
        rae: RAE model
        lpips_fn: LPIPS metric
        ssim_fn: SSIM metric from torchmetrics
        dataloader: DataLoader for dataset
        device: torch device
    
    Returns:
        Dictionary with average metrics
    """
    rae.eval()
    lpips_fn.eval()
    
    accumulator = MetricsAccumulator()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get target images
            original = batch["target_img"].to(device)
            batch_size = original.size(0)
            
            # Encode and decode
            latent = rae.encode(original)
            reconstructed = rae.decode(latent)
            reconstructed = reconstructed.clamp(0.0, 1.0)
            
            # Compute metrics
            l1_loss = F.l1_loss(reconstructed, original).item()
            l2_loss = F.mse_loss(reconstructed, original).item()
            ssim_similarity = ssim_fn(reconstructed, original).item()
            ssim_loss = 1.0 - ssim_similarity  # Convert to loss (lower is better)
            lpips_value = lpips_fn(original, reconstructed).mean().item()
            
            accumulator.update(l1_loss, l2_loss, ssim_loss, lpips_value, batch_size)
    
    return accumulator.get_averages()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAE reconstruction quality on entire dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1/pretrained/DINOv2-B.yaml"),
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",
        choices=["voc", "robot"],
        help="Which dataset to evaluate on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate on (train or val).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Load RAE model from config
    print(f"Loading RAE model from config: {args.config}")
    rae_config, *_ = parse_configs(args.config)
    assert rae_config is not None, f"No stage_1 section found in config {args.config}"

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    print("RAE model loaded successfully")

    # Load LPIPS metric
    print("Loading LPIPS metric...")
    lpips_fn = LPIPS().to(device)
    lpips_fn.eval()
    print("LPIPS loaded successfully")

    # Load SSIM metric
    print("Loading SSIM metric...")
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    print("SSIM loaded successfully")

    # Load dataset
    print(f"Loading {args.dataset} dataset ({args.split} split)...")
    train_loader, val_loader = get_alignment_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )
    dataloader = train_loader if args.split == "train" else val_loader
    print(f"Dataset loaded: {len(dataloader.dataset)} samples")

    # Evaluate
    print("\nStarting evaluation...")
    metrics = evaluate_reconstruction(
        rae=rae,
        lpips_fn=lpips_fn,
        ssim_fn=ssim_fn,
        dataloader=dataloader,
        device=device,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RECONSTRUCTION METRICS")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"Config: {args.config}")
    print("-" * 60)
    print(f"L1 Loss:      {metrics['L1']:.6f}")
    print(f"L2 Loss:      {metrics['L2']:.6f}")
    print(f"SSIM Loss:    {metrics['SSIM']:.6f}  (1 - SSIM, lower is better)")
    print(f"LPIPS:        {metrics['LPIPS']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

