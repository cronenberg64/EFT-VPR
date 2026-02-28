"""
Sensor Dropout Robustness Testing for EFT-VPR.

Simulates sensor failures by dropping N consecutive event frames and
forcing the Forecasting Transformer to predict its path autoregressively.
Compares localization quality between:

  - Full data (no dropout)
  - Dropout with autoregressive hallucination
  - Standard VPR baseline (which completely fails during dropout)

This is the key differentiator for EFT-VPR: the transformer enables
continued localization even when the event camera feed is interrupted.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.snn_encoder import SNNEncoder
from src.models.forecasting_transformer import ForecastingTransformer
from src.vpr.map_database import MapDatabase
from src.losses.triplet_loss import haversine_distance

logger = logging.getLogger(__name__)


@dataclass
class DropoutResult:
    """Results from a single dropout test.

    Attributes:
        n_dropped: Number of frames dropped.
        drop_start_idx: Index where dropout begins in the sequence.
        gps_predicted: Predicted GPS for each dropped frame, shape (N, 2).
        gps_ground_truth: Actual GPS for each dropped frame, shape (N, 2).
        localization_errors_m: Per-frame error in meters, shape (N,).
        mean_error_m: Average localization error over the dropout window.
        max_error_m: Maximum localization error during dropout.
        recall_at_25m: Fraction of predictions within 25m of ground truth.
    """
    n_dropped: int
    drop_start_idx: int
    gps_predicted: np.ndarray
    gps_ground_truth: np.ndarray
    localization_errors_m: np.ndarray
    mean_error_m: float
    max_error_m: float
    recall_at_25m: float


class SensorDropoutTest:
    """Simulates sensor dropout and measures localization robustness.

    During a dropout window of N frames:
      - Standard VPR produces NO predictions (sensor is offline)
      - Forecasting VPR uses autoregressive transformer predictions
        to continue localization despite missing sensor data

    The test measures how well the predicted GPS coordinates match
    the ground truth GPS during the dropout period.

    Args:
        encoder: Pre-trained SNNEncoder.
        transformer: Pre-trained ForecastingTransformer.
        map_db: Populated MapDatabase.
        device: Compute device.
        sequence_length: Transformer context window size.
    """

    def __init__(
        self,
        encoder: SNNEncoder,
        transformer: ForecastingTransformer,
        map_db: MapDatabase,
        device: Optional[torch.device] = None,
        sequence_length: int = 10,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = encoder.to(self.device).eval()
        self.transformer = transformer.to(self.device).eval()
        self.map_db = map_db
        self.sequence_length = sequence_length

    @torch.no_grad()
    def _encode_grids(self, grids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of grids into embeddings.

        Args:
            grids: Shape (N, C, H, W).

        Returns:
            Embeddings, shape (N, D).
        """
        grids = grids.to(self.device)
        return self.encoder.encode_single(grids)

    @torch.no_grad()
    def run_dropout_test(
        self,
        grids: torch.Tensor,
        gps_coords: np.ndarray,
        n_drop: int,
        drop_start: Optional[int] = None,
    ) -> DropoutResult:
        """Run a single sensor dropout test.

        Simulates dropping `n_drop` consecutive frames starting at
        `drop_start`. The transformer autoregressively predicts embeddings
        for the dropped frames, and each prediction is matched against
        the FAISS map to get GPS coordinates.

        Args:
            grids: Full sequence of event grids, shape (T_total, C, H, W).
            gps_coords: Ground truth GPS, shape (T_total, 2).
            n_drop: Number of frames to drop.
            drop_start: Index at which to start dropping. If None, drops
                        from position `sequence_length` (after building context).

        Returns:
            DropoutResult with per-frame errors and aggregated metrics.
        """
        T_total = grids.shape[0]

        if drop_start is None:
            drop_start = self.sequence_length

        if drop_start + n_drop > T_total:
            raise ValueError(
                f"Cannot drop {n_drop} frames starting at {drop_start} "
                f"(total frames: {T_total})"
            )

        if drop_start < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} context frames "
                f"before dropout start (got drop_start={drop_start})"
            )

        # Step 1: Encode the context window (frames before dropout)
        context_grids = grids[drop_start - self.sequence_length : drop_start]
        context_embeddings = self._encode_grids(context_grids)  # (seq_len, D)
        context_seq = context_embeddings.unsqueeze(0)  # (1, seq_len, D)

        # Step 2: Autoregressively predict through the dropout window
        predicted_embeddings = self.transformer.predict_autoregressive(
            context_seq, n_steps=n_drop
        )  # List of n_drop tensors, each (1, D)

        # Step 3: Match each predicted embedding against the FAISS map
        gps_predicted = np.zeros((n_drop, 2), dtype=np.float64)
        for i, pred_emb in enumerate(predicted_embeddings):
            results = self.map_db.search(pred_emb.cpu().numpy(), top_k=1)
            if results[0]:
                gps_predicted[i] = results[0][0]["gps"]

        # Step 4: Compute localization errors
        gps_gt = gps_coords[drop_start : drop_start + n_drop]

        errors = np.zeros(n_drop)
        for i in range(n_drop):
            gt = torch.tensor(gps_gt[i], dtype=torch.float64).unsqueeze(0)
            pred = torch.tensor(gps_predicted[i], dtype=torch.float64).unsqueeze(0)
            errors[i] = haversine_distance(gt, pred).item()

        return DropoutResult(
            n_dropped=n_drop,
            drop_start_idx=drop_start,
            gps_predicted=gps_predicted,
            gps_ground_truth=gps_gt,
            localization_errors_m=errors,
            mean_error_m=float(errors.mean()),
            max_error_m=float(errors.max()),
            recall_at_25m=float((errors < 25.0).mean()),
        )

    def run_sweep(
        self,
        grids: torch.Tensor,
        gps_coords: np.ndarray,
        drop_counts: list[int] = None,
        n_trials: int = 5,
        seed: int = 42,
    ) -> dict[int, list[DropoutResult]]:
        """Run dropout tests across multiple dropout lengths.

        For each dropout count, runs `n_trials` tests at random positions
        and reports aggregate statistics.

        Args:
            grids: Full sequence of event grids, shape (T_total, C, H, W).
            gps_coords: Ground truth GPS, shape (T_total, 2).
            drop_counts: List of dropout lengths to test (default: [5,10,15,20]).
            n_trials: Number of random positions per dropout length.
            seed: RNG seed for reproducibility.

        Returns:
            Dict mapping n_drop → list of DropoutResults.
        """
        if drop_counts is None:
            drop_counts = [5, 10, 15, 20]

        T_total = grids.shape[0]
        rng = np.random.default_rng(seed)

        results: dict[int, list[DropoutResult]] = {}

        for n_drop in drop_counts:
            max_start = T_total - n_drop
            min_start = self.sequence_length
            if min_start >= max_start:
                logger.warning(
                    f"Skipping n_drop={n_drop}: not enough frames "
                    f"(need {n_drop + self.sequence_length}, have {T_total})"
                )
                continue

            trial_results = []
            positions = rng.integers(min_start, max_start, size=n_trials)

            for trial, pos in enumerate(positions):
                result = self.run_dropout_test(
                    grids, gps_coords, n_drop=n_drop, drop_start=int(pos)
                )
                trial_results.append(result)
                logger.debug(
                    f"n_drop={n_drop}, trial={trial}: "
                    f"mean_err={result.mean_error_m:.1f}m, "
                    f"R@25m={result.recall_at_25m:.1%}"
                )

            results[n_drop] = trial_results

        return results

    @staticmethod
    def summarize_sweep(
        sweep_results: dict[int, list[DropoutResult]]
    ) -> dict[int, dict]:
        """Aggregate sweep results into summary statistics.

        Args:
            sweep_results: Output from run_sweep().

        Returns:
            Dict mapping n_drop → {mean_error, std_error, recall@25m, ...}.
        """
        summary = {}
        for n_drop, trials in sweep_results.items():
            errors = np.array([t.mean_error_m for t in trials])
            max_errors = np.array([t.max_error_m for t in trials])
            recalls = np.array([t.recall_at_25m for t in trials])

            summary[n_drop] = {
                "n_trials": len(trials),
                "mean_error_m": float(errors.mean()),
                "std_error_m": float(errors.std()),
                "median_error_m": float(np.median(errors)),
                "max_error_m": float(max_errors.max()),
                "recall_at_25m": float(recalls.mean()),
                "recall_at_25m_std": float(recalls.std()),
            }

        return summary

    @staticmethod
    def print_sweep_summary(summary: dict[int, dict]):
        """Pretty-print sweep summary to console."""
        print("\n" + "=" * 70)
        print("  Sensor Dropout Robustness Test Results")
        print("=" * 70)
        print(f"  {'Dropped':>8s}  {'Mean Err (m)':>12s}  {'Std':>8s}  "
              f"{'Max (m)':>8s}  {'R@25m':>8s}")
        print("-" * 70)

        for n_drop in sorted(summary.keys()):
            s = summary[n_drop]
            print(
                f"  {n_drop:>8d}  {s['mean_error_m']:>12.1f}  "
                f"{s['std_error_m']:>8.1f}  {s['max_error_m']:>8.1f}  "
                f"{s['recall_at_25m']:>7.1%}"
            )

        print("=" * 70)


def compare_standard_vs_forecasting(
    encoder: SNNEncoder,
    transformer: ForecastingTransformer,
    map_db: MapDatabase,
    grids: torch.Tensor,
    gps_coords: np.ndarray,
    n_dropped: int = 10,
    drop_start: Optional[int] = None,
    sequence_length: int = 10,
    device: Optional[torch.device] = None,
) -> dict:
    """Compare Standard VPR vs Forecasting VPR during sensor dropout.

    Standard VPR:
      - Cannot localize during dropout (no sensor data).
      - Returns NaN for dropped frames.

    Forecasting VPR:
      - Uses autoregressive prediction to maintain localization.
      - Returns predicted GPS even during dropout.

    Args:
        encoder: Pre-trained encoder.
        transformer: Pre-trained transformer.
        map_db: Populated map database.
        grids: Full event grid sequence, shape (T, C, H, W).
        gps_coords: Ground truth GPS, shape (T, 2).
        n_dropped: Number of frames to drop.
        drop_start: Where to start the dropout.
        sequence_length: Transformer context window.
        device: Compute device.

    Returns:
        Dict with comparison results for both methods.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if drop_start is None:
        drop_start = sequence_length

    # --- Forecasting VPR ---
    dropout_test = SensorDropoutTest(
        encoder=encoder,
        transformer=transformer,
        map_db=map_db,
        device=device,
        sequence_length=sequence_length,
    )
    forecasting_result = dropout_test.run_dropout_test(
        grids, gps_coords, n_drop=n_dropped, drop_start=drop_start
    )

    # --- Standard VPR ---
    # Standard VPR has NO predictions during dropout (sensor offline)
    standard_errors = np.full(n_dropped, np.nan)

    # For frames WITH data (before/after dropout), standard VPR works
    # but during the dropout window it's completely blind
    standard_result = {
        "n_dropped": n_dropped,
        "mean_error_m": np.nan,  # No predictions possible
        "recall_at_25m": 0.0,     # 0% recall — no data, no predictions
        "status": "BLIND — no sensor data available",
    }

    comparison = {
        "n_dropped": n_dropped,
        "drop_start": drop_start,
        "forecasting_vpr": {
            "mean_error_m": forecasting_result.mean_error_m,
            "max_error_m": forecasting_result.max_error_m,
            "recall_at_25m": forecasting_result.recall_at_25m,
            "per_frame_errors_m": forecasting_result.localization_errors_m.tolist(),
        },
        "standard_vpr": standard_result,
        "improvement": (
            "Forecasting VPR maintains localization during dropout. "
            f"Mean error: {forecasting_result.mean_error_m:.1f}m, "
            f"Recall@25m: {forecasting_result.recall_at_25m:.1%}. "
            "Standard VPR: COMPLETELY BLIND (0% recall)."
        ),
    }

    return comparison
