"""
Forecasting Transformer Training Loop.

Trains the ForecastingTransformer to predict next embeddings using
Temporal Contrastive Loss (InfoNCE). Supports two training stages:

  Stage 1: Frozen encoder — only transformer weights are updated.
  Stage 2: End-to-end fine-tuning — encoder unfrozen with reduced LR.

Usage:
    python scripts/train.py --phase transformer --config configs/default.yaml
"""

import logging
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
from src.models.forecasting_transformer import ForecastingTransformer, TransformerConfig
from src.losses.temporal_contrastive import TemporalContrastiveLoss
from src.training.train_encoder import EncoderTrainer

logger = logging.getLogger(__name__)


class TransformerTrainer:
    """Training manager for the Forecasting Transformer.

    Handles frozen-encoder training and optional end-to-end fine-tuning.

    Args:
        encoder: Pre-trained SNNEncoder.
        transformer: ForecastingTransformer instance.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        config: Full configuration dict.
        device: Target device.
        output_dir: Directory for checkpoints and logs.
        freeze_encoder: If True, encoder weights are frozen (Stage 1).
    """

    def __init__(
        self,
        encoder: SNNEncoder,
        transformer: ForecastingTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints",
        freeze_encoder: bool = True,
    ):
        self.config = config or {}
        training_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("loss", {})

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = encoder.to(self.device)
        self.transformer = transformer.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.freeze_encoder = freeze_encoder

        # Freeze encoder if Stage 1
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder FROZEN — only transformer weights will be trained")
        else:
            logger.info("End-to-end mode — encoder + transformer both trainable")

        # Loss function
        self.criterion = TemporalContrastiveLoss(
            temperature=loss_cfg.get("temperature", 0.07),
        )

        # Optimizer — only transformer params if frozen, else both
        self.lr = training_cfg.get("transformer_lr", 1e-3)
        if freeze_encoder:
            params = list(self.transformer.parameters())
        else:
            finetune_lr = training_cfg.get("finetune_lr", 1e-4)
            params = [
                {"params": self.transformer.parameters(), "lr": self.lr},
                {"params": self.encoder.parameters(), "lr": finetune_lr},
            ]

        self.optimizer = AdamW(
            params,
            lr=self.lr,
            weight_decay=training_cfg.get("transformer_weight_decay", 1e-4),
        )

        # Scheduler
        self.epochs = training_cfg.get("transformer_epochs", 100)
        if not freeze_encoder:
            self.epochs = training_cfg.get("finetune_epochs", 50)

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
            T_0=max(1, (self.epochs - warmup_epochs) * steps_per_epoch),
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
        phase_name = "transformer_frozen" if freeze_encoder else "finetune"
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{phase_name}")

        # Tracking
        self.best_val_loss = float("inf")
        self.global_step = 0

    def _encode_batch(self, input_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of input sequences through the SNN encoder.

        The input has shape (B, T, C, H, W) with T = seq_len + 1.
        We need:
          - Input embeddings: encode bins [0..T-2] → seq of embeddings for transformer
          - Target embedding: encode bin [T-1] → ground truth for loss

        But we encode all T bins at once for efficiency, treating each bin
        as a single-step sequence through the encoder.

        Args:
            input_seq: Raw event bins, shape (B, T, C, H, W) where T includes
                       both input sequence AND the target bin.

        Returns:
            input_embeddings: Shape (B, T-1, D) — embeddings of the input bins.
            target_embedding: Shape (B, D) — embedding of the target bin.
        """
        B, T, C, H, W = input_seq.shape

        # Reshape to encode all bins at once: (B*T, 1, C, H, W)
        # Each bin is treated as a single-timestep sequence
        all_bins = input_seq.reshape(B * T, C, H, W)

        if self.freeze_encoder:
            with torch.no_grad():
                all_embeddings = self.encoder.encode_single(all_bins)  # (B*T, D)
        else:
            all_embeddings = self.encoder.encode_single(all_bins)

        # Reshape back: (B, T, D)
        all_embeddings = all_embeddings.reshape(B, T, -1)

        # Split: input is [0..T-2], target is [T-1]
        input_embeddings = all_embeddings[:, :-1, :]  # (B, T-1, D)
        target_embedding = all_embeddings[:, -1, :]   # (B, D)

        return input_embeddings, target_embedding

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Dict with epoch metrics.
        """
        self.transformer.train()
        if not self.freeze_encoder:
            self.encoder.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs}",
            leave=False,
        )

        for batch in pbar:
            # input has shape (B, seq_len, C, H, W) — the event bins
            # target has shape (B, C, H, W) — the next bin
            input_bins = batch["input"].to(self.device)
            target_bin = batch["target"].to(self.device)

            # Combine input + target into one sequence for encoding
            full_seq = torch.cat(
                [input_bins, target_bin.unsqueeze(1)], dim=1
            )  # (B, seq_len+1, C, H, W)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                # Encode all bins
                input_embeddings, target_embedding = self._encode_batch(full_seq)

                # Transformer predicts next embedding
                predicted = self.transformer(input_embeddings)  # (B, D)

                # Temporal Contrastive Loss
                loss, stats = self.criterion(predicted, target_embedding)

            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Clip gradients
            if self.freeze_encoder:
                torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(), self.gradient_clip
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(self.transformer.parameters()) + list(self.encoder.parameters()),
                    self.gradient_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            epoch_loss += loss.item()
            epoch_accuracy += stats.get("accuracy", 0.0)
            n_batches += 1
            self.global_step += 1

            # TensorBoard step logging
            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar("train/accuracy_step", stats.get("accuracy", 0), self.global_step)
                self.writer.add_scalar("train/temperature", stats["temperature"], self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{stats.get('accuracy', 0):.1%}",
                "pos_sim": f"{stats['mean_positive_sim']:.3f}",
            })

        metrics = {
            "loss": epoch_loss / max(n_batches, 1),
            "accuracy": epoch_accuracy / max(n_batches, 1),
        }

        self.writer.add_scalar("train/loss_epoch", metrics["loss"], epoch)
        self.writer.add_scalar("train/accuracy_epoch", metrics["accuracy"], epoch)

        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.encoder.eval()
        self.transformer.eval()

        val_loss = 0.0
        val_accuracy = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_bins = batch["input"].to(self.device)
            target_bin = batch["target"].to(self.device)

            full_seq = torch.cat([input_bins, target_bin.unsqueeze(1)], dim=1)
            input_embeddings, target_embedding = self._encode_batch(full_seq)
            predicted = self.transformer(input_embeddings)
            loss, stats = self.criterion(predicted, target_embedding)

            val_loss += loss.item()
            val_accuracy += stats.get("accuracy", 0.0)
            n_batches += 1

        metrics = {
            "val_loss": val_loss / max(n_batches, 1),
            "val_accuracy": val_accuracy / max(n_batches, 1),
        }

        self.writer.add_scalar("val/loss", metrics["val_loss"], epoch)
        self.writer.add_scalar("val/accuracy", metrics["val_accuracy"], epoch)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint with both encoder and transformer states."""
        phase = "frozen" if self.freeze_encoder else "finetune"
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "transformer_state_dict": self.transformer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "phase": phase,
        }

        path = self.output_dir / f"transformer_{phase}_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.output_dir / f"transformer_{phase}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model: val_loss={metrics.get('val_loss', 'N/A')}")

        latest_path = self.output_dir / f"transformer_{phase}_latest.pt"
        torch.save(checkpoint, latest_path)

    def train(self) -> dict:
        """Run the full training loop."""
        phase = "frozen encoder" if self.freeze_encoder else "end-to-end fine-tuning"
        logger.info(f"Starting transformer training ({phase}) for {self.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            all_metrics = {**train_metrics, **val_metrics}

            val_loss = val_metrics.get("val_loss", train_metrics["loss"])
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % 10 == 0 or is_best or epoch == self.epochs - 1:
                self.save_checkpoint(epoch, all_metrics, is_best=is_best)

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} — "
                f"loss={train_metrics['loss']:.4f}, "
                f"acc={train_metrics['accuracy']:.1%}, "
                f"val_loss={val_metrics.get('val_loss', 'N/A')}, "
                f"val_acc={val_metrics.get('val_accuracy', 'N/A')}"
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
        encoder_config: Optional[SNNEncoderConfig] = None,
        transformer_config: Optional[TransformerConfig] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[SNNEncoder, ForecastingTransformer, dict]:
        """Load trained encoder + transformer from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint.
            encoder_config: Override encoder config (uses saved if None).
            transformer_config: Override transformer config.
            device: Target device.

        Returns:
            Tuple of (encoder, transformer, checkpoint_dict).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = checkpoint.get("config", {})

        if encoder_config is None:
            encoder_config = SNNEncoderConfig.from_dict(config)
        if transformer_config is None:
            transformer_config = TransformerConfig.from_dict(config)

        encoder = SNNEncoder(encoder_config).to(device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])

        transformer = ForecastingTransformer(transformer_config).to(device)
        transformer.load_state_dict(checkpoint["transformer_state_dict"])

        encoder.eval()
        transformer.eval()

        logger.info(f"Loaded encoder + transformer from {checkpoint_path}")
        return encoder, transformer, checkpoint
