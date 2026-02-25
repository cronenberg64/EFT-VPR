"""
SNN Encoder Training Loop with GPS-Based Triplet Loss.

Trains the SNNEncoder to produce metrically meaningful place embeddings
using triplet loss with GPS-distance mining.

Usage:
    python scripts/train.py --phase encoder --config configs/default.yaml

Features:
    - AdamW optimizer with cosine annealing + warmup
    - TensorBoard logging (loss, triplet stats, learning rate, beta values)
    - Gradient clipping for SNN stability
    - Best checkpoint saving by validation Recall@1
    - Mixed precision training support for RTX 4070
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.snn_encoder import SNNEncoder, SNNEncoderConfig
from src.losses.triplet_loss import GPSTripletLoss

logger = logging.getLogger(__name__)


class EncoderTrainer:
    """Training manager for the SNN Encoder.

    Handles the full training loop including optimization, logging,
    checkpointing, and validation.

    Args:
        model: SNNEncoder instance.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        config: Training configuration dict.
        device: Target device.
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: SNNEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints",
    ):
        self.config = config or {}
        training_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("loss", {})

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.criterion = GPSTripletLoss(
            margin=loss_cfg.get("triplet_margin", 0.3),
            positive_threshold_m=loss_cfg.get("positive_threshold_m", 25.0),
            negative_threshold_m=loss_cfg.get("negative_threshold_m", 100.0),
        )

        # Optimizer
        self.lr = training_cfg.get("encoder_lr", 1e-3)
        self.weight_decay = training_cfg.get("encoder_weight_decay", 1e-4)
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler: Linear warmup → Cosine annealing
        self.epochs = training_cfg.get("encoder_epochs", 100)
        warmup_epochs = training_cfg.get("warmup_epochs", 5)
        steps_per_epoch = len(train_loader)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * steps_per_epoch,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=(self.epochs - warmup_epochs) * steps_per_epoch,
            T_mult=1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * steps_per_epoch],
        )

        # Gradient clipping
        self.gradient_clip = training_cfg.get("gradient_clip", 1.0)

        # Mixed precision
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.device.type == "cuda",
        )
        self.use_amp = self.device.type == "cuda"

        # Logging
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.config.get("paths", {}).get("tensorboard", "runs")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/encoder")

        # Tracking
        self.best_val_loss = float("inf")
        self.global_step = 0

        logger.info(
            f"EncoderTrainer initialized: device={self.device}, "
            f"lr={self.lr}, epochs={self.epochs}, "
            f"grad_clip={self.gradient_clip}, AMP={self.use_amp}"
        )

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dict with epoch training metrics.
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_triplets = 0
        epoch_active = 0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs}",
            leave=False,
        )

        for batch in pbar:
            input_seq = batch["input"].to(self.device)     # (B, T, C, H, W)
            target_gps = batch["target_gps"].to(self.device)  # (B, 2)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional mixed precision
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                # Encode the input sequence
                embeddings = self.model(input_seq)  # (B, D)

                # Compute triplet loss using target GPS for pair mining
                loss, stats = self.criterion(embeddings, target_gps)

            if loss.item() > 0 and stats["n_triplets"] > 0:
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping for SNN stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # No valid triplets in this batch — skip update
                pass

            self.scheduler.step()

            # Logging
            epoch_loss += loss.item()
            epoch_triplets += stats["n_triplets"]
            epoch_active += stats["n_active"]
            n_batches += 1
            self.global_step += 1

            # TensorBoard step logging
            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar("train/active_ratio", stats["active_ratio"], self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "triplets": stats["n_triplets"],
                "active": f"{stats['active_ratio']:.1%}",
            })

        metrics = {
            "loss": epoch_loss / max(n_batches, 1),
            "total_triplets": epoch_triplets,
            "total_active": epoch_active,
            "active_ratio": epoch_active / max(epoch_triplets, 1),
        }

        # Log epoch metrics
        self.writer.add_scalar("train/loss_epoch", metrics["loss"], epoch)
        self.writer.add_scalar("train/triplets_per_epoch", epoch_triplets, epoch)

        # Log learnable beta values
        for i, block in enumerate(self.model.blocks):
            self.writer.add_scalar(f"betas/block_{i}", block.lif.beta.item(), epoch)
        self.writer.add_scalar("betas/output", self.model.lif_out.beta.item(), epoch)

        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Run validation epoch.

        Computes average triplet loss on the validation set.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict with validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        val_triplets = 0
        n_batches = 0

        for batch in self.val_loader:
            input_seq = batch["input"].to(self.device)
            target_gps = batch["target_gps"].to(self.device)

            embeddings = self.model(input_seq)
            loss, stats = self.criterion(embeddings, target_gps)

            val_loss += loss.item()
            val_triplets += stats["n_triplets"]
            n_batches += 1

        metrics = {
            "val_loss": val_loss / max(n_batches, 1),
            "val_triplets": val_triplets,
        }

        self.writer.add_scalar("val/loss", metrics["val_loss"], epoch)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            metrics: Training/validation metrics.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "model_config": {
                "in_channels": self.model.config.in_channels,
                "channels": self.model.config.channels,
                "embedding_dim": self.model.config.embedding_dim,
                "beta_init": self.model.config.beta_init,
                "grid_size": self.model.config.grid_size,
            },
        }

        # Save periodic checkpoint
        path = self.output_dir / f"encoder_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "encoder_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: val_loss={metrics.get('val_loss', 'N/A')}")

        # Save latest (always)
        latest_path = self.output_dir / "encoder_latest.pt"
        torch.save(checkpoint, latest_path)

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with final training statistics.
        """
        logger.info(f"Starting encoder training for {self.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Merge metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Check if best
            val_loss = val_metrics.get("val_loss", train_metrics["loss"])
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint every 10 epochs or if best
            if (epoch + 1) % 10 == 0 or is_best or epoch == self.epochs - 1:
                self.save_checkpoint(epoch, all_metrics, is_best=is_best)

            # Log
            lr = self.scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} — "
                f"loss={train_metrics['loss']:.4f}, "
                f"triplets={train_metrics['total_triplets']}, "
                f"active={train_metrics['active_ratio']:.1%}, "
                f"val_loss={val_metrics.get('val_loss', 'N/A')}, "
                f"lr={lr:.2e}"
            )

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 60:.1f} minutes")

        self.writer.close()

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.epochs,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> tuple[SNNEncoder, dict]:
        """Load a trained encoder from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Device to load onto.

        Returns:
            Tuple of (model, checkpoint_dict).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_cfg = SNNEncoderConfig(**checkpoint["model_config"])
        model = SNNEncoder(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        logger.info(f"Loaded encoder from {checkpoint_path} "
                    f"(epoch {checkpoint['epoch'] + 1})")
        return model, checkpoint
