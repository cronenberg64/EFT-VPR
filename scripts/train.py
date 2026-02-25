"""
Unified Training Entry Point for EFT-VPR.

Usage:
    python scripts/train.py --phase encoder --config configs/default.yaml
    python scripts/train.py --phase transformer --config configs/default.yaml
    python scripts/train.py --phase finetune --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_encoder(config: dict, data_path: str, device: torch.device):
    """Train the SNN Encoder with triplet loss."""
    from src.models.snn_encoder import SNNEncoder, SNNEncoderConfig
    from src.training.train_encoder import EncoderTrainer
    from src.data.sequence_dataset import create_dataloader

    # Build model
    encoder_config = SNNEncoderConfig.from_dict(config)
    model = SNNEncoder(encoder_config)
    logger.info(f"Model parameters: {model.get_num_parameters()}")

    # Build data loaders
    data_dir = Path(data_path)
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        logger.error(f"No .h5 files found in {data_dir}")
        logger.info("Run preprocessing first: python scripts/preprocess.py ...")
        return

    # Split: 80% train, 20% val
    split_idx = max(1, int(len(h5_files) * 0.8))
    train_files = h5_files[:split_idx]
    val_files = h5_files[split_idx:] if split_idx < len(h5_files) else None

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    train_loader = create_dataloader(
        h5_paths=train_files,
        sequence_length=data_cfg.get("sequence_length", 10),
        stride=data_cfg.get("stride", 1),
        batch_size=training_cfg.get("encoder_batch_size", 64),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        augment=True,
        normalize="minmax",
    )

    val_loader = None
    if val_files:
        val_loader = create_dataloader(
            h5_paths=val_files,
            sequence_length=data_cfg.get("sequence_length", 10),
            batch_size=training_cfg.get("encoder_batch_size", 64),
            shuffle=False,
            augment=False,
            normalize="minmax",
        )

    # Train
    trainer = EncoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=config.get("paths", {}).get("checkpoints", "checkpoints"),
    )

    results = trainer.train()
    logger.info(f"Training complete: {results}")


def main():
    parser = argparse.ArgumentParser(description="EFT-VPR Training")
    parser.add_argument(
        "--phase", type=str, required=True,
        choices=["encoder", "transformer", "finetune"],
        help="Training phase",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs:
        config.setdefault("training", {})[f"{args.phase}_epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})[f"{args.phase}_batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})[f"{args.phase}_lr"] = args.lr

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dispatch
    if args.phase == "encoder":
        train_encoder(config, args.data, device)
    elif args.phase == "transformer":
        logger.info("Transformer training — Phase 3 (not yet implemented)")
    elif args.phase == "finetune":
        logger.info("End-to-end fine-tuning — Phase 3 (not yet implemented)")


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
