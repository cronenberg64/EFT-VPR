"""
EFT-VPR Inference Pipeline.

Two modes of Visual Place Recognition:

  1. **Standard VPR** (baseline): Single-frame matching.
     Event grid → Encoder → FAISS → GPS.
     No temporal reasoning—each bin is matched independently.

  2. **Forecasting VPR** (full pipeline): Temporal prediction.
     Event sequence → Encoder → Transformer (predicts ẑ_{t+1}) → FAISS → GPS.
     Uses accumulated temporal context for more robust matching.

Both modes share the same FAISS map database and encoder.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.event_binning import BinConfig, bin_events, bin_events_to_tensor
from src.models.snn_encoder import SNNEncoder
from src.models.forecasting_transformer import ForecastingTransformer
from src.vpr.map_database import MapDatabase

logger = logging.getLogger(__name__)


class EFTVPRPipeline:
    """Full EFT-VPR inference pipeline.

    Handles the end-to-end flow from event data to GPS localization:
        Event Stream → Binning → SNN Encoder → Transformer → FAISS → GPS

    Also supports Standard VPR (no transformer) as a baseline.

    Args:
        encoder: Pre-trained SNNEncoder.
        transformer: Pre-trained ForecastingTransformer.
        map_db: Populated MapDatabase.
        bin_config: Event binning configuration.
        device: Compute device.
        sequence_length: Number of bins in the temporal context window.
    """

    def __init__(
        self,
        encoder: SNNEncoder,
        transformer: ForecastingTransformer,
        map_db: MapDatabase,
        bin_config: Optional[BinConfig] = None,
        device: Optional[torch.device] = None,
        sequence_length: int = 10,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = encoder.to(self.device).eval()
        self.transformer = transformer.to(self.device).eval()
        self.map_db = map_db
        self.bin_config = bin_config or BinConfig()
        self.sequence_length = sequence_length

        # Rolling buffer for temporal context
        self._embedding_buffer: list[torch.Tensor] = []

        logger.info(
            f"EFTVPRPipeline initialized: seq_len={sequence_length}, "
            f"map_size={map_db.size}, device={self.device}"
        )

    def reset(self):
        """Reset the temporal context buffer."""
        self._embedding_buffer.clear()

    @torch.no_grad()
    def _encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a single event grid into an embedding.

        Args:
            grid: Event grid of shape (C, H, W) or (1, C, H, W).

        Returns:
            Embedding of shape (D,).
        """
        if grid.ndim == 3:
            grid = grid.unsqueeze(0)
        grid = grid.to(self.device)
        emb = self.encoder.encode_single(grid)  # (1, D)
        return emb.squeeze(0)

    @torch.no_grad()
    def localize_standard(
        self,
        grid: torch.Tensor,
        top_k: int = 5,
    ) -> list[dict]:
        """Standard VPR: single-frame matching (baseline).

        Encodes the current event grid and searches the FAISS map
        without any temporal context.

        Args:
            grid: Event grid of shape (C, H, W).
            top_k: Number of top matches to return.

        Returns:
            List of top_k match dicts with 'gps', 'similarity', etc.
        """
        emb = self._encode_grid(grid)
        results = self.map_db.search(
            emb.unsqueeze(0).cpu().numpy(),
            top_k=top_k,
        )
        return results[0]

    @torch.no_grad()
    def localize_forecasting(
        self,
        grid: torch.Tensor,
        top_k: int = 5,
    ) -> list[dict]:
        """Forecasting VPR: temporal prediction with transformer.

        Adds the current grid's embedding to the temporal buffer,
        then uses the transformer to predict the *next* embedding.
        Matches the predicted embedding against the FAISS map.

        Until the buffer has enough history (< sequence_length bins),
        falls back to standard single-frame matching.

        Args:
            grid: Event grid of shape (C, H, W).
            top_k: Number of top matches to return.

        Returns:
            List of top_k match dicts.
        """
        emb = self._encode_grid(grid)
        self._embedding_buffer.append(emb.cpu())

        # Not enough context yet — fall back to standard
        if len(self._embedding_buffer) < self.sequence_length:
            results = self.map_db.search(
                emb.unsqueeze(0).cpu().numpy(),
                top_k=top_k,
            )
            return results[0]

        # Use the last `sequence_length` embeddings
        context = torch.stack(
            self._embedding_buffer[-self.sequence_length:]
        ).unsqueeze(0).to(self.device)  # (1, T, D)

        # Transformer predicts the next embedding
        predicted = self.transformer(context)  # (1, D)

        # Search FAISS with predicted embedding
        results = self.map_db.search(
            predicted.cpu().numpy(),
            top_k=top_k,
        )

        # Trim buffer to avoid unbounded growth
        max_buffer = self.sequence_length * 2
        if len(self._embedding_buffer) > max_buffer:
            self._embedding_buffer = self._embedding_buffer[-max_buffer:]

        return results[0]

    @torch.no_grad()
    def localize_batch_standard(
        self,
        grids: torch.Tensor,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Batch standard VPR for evaluation.

        Args:
            grids: Event grids of shape (N, C, H, W).
            top_k: Top matches per query.

        Returns:
            List of N result lists.
        """
        grids = grids.to(self.device)
        embeddings = self.encoder.encode_single(grids)  # (N, D)
        return self.map_db.search(embeddings.cpu().numpy(), top_k=top_k)

    @torch.no_grad()
    def localize_batch_forecasting(
        self,
        sequences: torch.Tensor,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Batch forecasting VPR for evaluation.

        Each sequence is encoded through the SNN encoder (per-bin),
        then the transformer predicts the next embedding.

        Args:
            sequences: Event grid sequences, shape (N, T, C, H, W).
            top_k: Top matches per query.

        Returns:
            List of N result lists.
        """
        B, T, C, H, W = sequences.shape
        sequences = sequences.to(self.device)

        # Encode all bins: (B*T, C, H, W) → (B*T, D) → (B, T, D)
        all_grids = sequences.reshape(B * T, C, H, W)
        all_emb = self.encoder.encode_single(all_grids)
        emb_seq = all_emb.reshape(B, T, -1)

        # Transformer prediction
        predicted = self.transformer(emb_seq)  # (B, D)

        return self.map_db.search(predicted.cpu().numpy(), top_k=top_k)

    def from_events(
        self,
        events_df,
        mode: str = "forecasting",
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Run VPR from raw event data.

        End-to-end: raw events → binning → encoding → localization.

        Args:
            events_df: DataFrame with x, y, timestamp, polarity columns.
            mode: 'standard' or 'forecasting'.
            top_k: Top matches per bin.

        Returns:
            List of result lists, one per bin.
        """
        self.reset()

        # Bin events
        grids_tensor, _ = bin_events_to_tensor(events_df, self.bin_config)

        all_results = []
        for i in range(grids_tensor.shape[0]):
            grid = grids_tensor[i]
            if mode == "standard":
                results = self.localize_standard(grid, top_k=top_k)
            else:
                results = self.localize_forecasting(grid, top_k=top_k)
            all_results.append(results)

        return all_results

    def get_pipeline_info(self) -> dict:
        """Return pipeline configuration summary."""
        return {
            "encoder_params": self.encoder.get_num_parameters(),
            "transformer_params": self.transformer.get_num_parameters(),
            "map_size": self.map_db.size,
            "sequence_length": self.sequence_length,
            "device": str(self.device),
            "buffer_size": len(self._embedding_buffer),
        }


class StandardVPRBaseline:
    """Standalone Standard VPR baseline for comparison.

    Simple single-frame embedding matching without any temporal modeling.
    Used in evaluation to compare against the full EFT-VPR pipeline.

    Args:
        encoder: Pre-trained SNNEncoder.
        map_db: Populated MapDatabase.
        device: Compute device.
    """

    def __init__(
        self,
        encoder: SNNEncoder,
        map_db: MapDatabase,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = encoder.to(self.device).eval()
        self.map_db = map_db

    @torch.no_grad()
    def localize(
        self,
        grids: torch.Tensor,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Match event grids against the map.

        Args:
            grids: Shape (N, C, H, W).
            top_k: Top matches per query.

        Returns:
            List of N result lists.
        """
        grids = grids.to(self.device)
        embeddings = self.encoder.encode_single(grids)
        return self.map_db.search(embeddings.cpu().numpy(), top_k=top_k)

    @torch.no_grad()
    def localize_from_embeddings(
        self,
        embeddings: np.ndarray | torch.Tensor,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Search from pre-computed embeddings.

        Args:
            embeddings: Shape (N, D).
            top_k: Top matches.

        Returns:
            List of N result lists.
        """
        return self.map_db.search(embeddings, top_k=top_k)
