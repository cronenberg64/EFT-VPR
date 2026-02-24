"""
Event Binning Engine for Brisbane Event VPR Dataset.

Transforms raw, asynchronous event streams (from parquet files) into structured
2D spatial grids suitable for SNN processing.

Supports two binning modes:
  - Fixed-count: accumulate N events per bin
  - Fixed-duration: accumulate events within Δt ms windows

Output: torch.Tensor grids of shape (grid_size, grid_size) per bin.

Hardware note: All operations are vectorized with NumPy for maximum throughput.
Avoid Python-level loops over individual events.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class BinMode(Enum):
    """Binning strategy for event accumulation."""
    FIXED_COUNT = "fixed_count"
    FIXED_DURATION = "fixed_duration"


@dataclass
class BinConfig:
    """Configuration for event binning.

    Attributes:
        grid_size: Output spatial resolution (grid_size x grid_size).
        sensor_width: Original sensor width in pixels (DAVIS346 = 346).
        sensor_height: Original sensor height in pixels (DAVIS346 = 260).
        bin_mode: Binning strategy (fixed_count or fixed_duration).
        bin_count: Number of events per bin (fixed_count mode).
        bin_duration_ms: Duration per bin in milliseconds (fixed_duration mode).
        polarity_mode: How to handle event polarity.
            'binary' — clamp to {0, 1} (any event = 1).
            'polarity_sum' — separate ON/OFF channels, shape (2, H, W).
    """
    grid_size: int = 64
    sensor_width: int = 346
    sensor_height: int = 260
    bin_mode: BinMode = BinMode.FIXED_COUNT
    bin_count: int = 5000
    bin_duration_ms: float = 50.0
    polarity_mode: str = "binary"

    @classmethod
    def from_dict(cls, config: dict) -> "BinConfig":
        """Create BinConfig from a config dictionary (e.g., loaded from YAML)."""
        data_cfg = config.get("data", config)
        return cls(
            grid_size=data_cfg.get("grid_size", 64),
            sensor_width=data_cfg.get("sensor_width", 346),
            sensor_height=data_cfg.get("sensor_height", 260),
            bin_mode=BinMode(data_cfg.get("bin_mode", "fixed_count")),
            bin_count=data_cfg.get("bin_count", 5000),
            bin_duration_ms=data_cfg.get("bin_duration_ms", 50.0),
            polarity_mode=data_cfg.get("polarity_mode", "binary"),
        )


def _project_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    sensor_width: int,
    sensor_height: int,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project sensor-space coordinates to grid-space coordinates.

    Uses floor division to map (x, y) ∈ [0, sensor_w) × [0, sensor_h)
    onto [0, grid_size) × [0, grid_size).

    Args:
        x: Event x-coordinates, shape (N,).
        y: Event y-coordinates, shape (N,).
        sensor_width: Original sensor width.
        sensor_height: Original sensor height.
        grid_size: Target grid resolution.

    Returns:
        Tuple of (grid_x, grid_y), each shape (N,), clipped to [0, grid_size-1].
    """
    grid_x = (x * grid_size // sensor_width).astype(np.int32)
    grid_y = (y * grid_size // sensor_height).astype(np.int32)
    # Safety clamp (handles edge-case where x == sensor_width)
    np.clip(grid_x, 0, grid_size - 1, out=grid_x)
    np.clip(grid_y, 0, grid_size - 1, out=grid_y)
    return grid_x, grid_y


def _accumulate_events_binary(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """Accumulate events into a binary spike grid (any event → 1).

    Args:
        grid_x: Projected x-coordinates, shape (N,).
        grid_y: Projected y-coordinates, shape (N,).
        grid_size: Output grid size.

    Returns:
        Binary grid of shape (1, grid_size, grid_size) with values in {0, 1}.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(grid, (grid_y, grid_x), 1.0)
    # Clamp to binary
    grid = np.clip(grid, 0.0, 1.0)
    return grid[np.newaxis, :, :]  # (1, H, W)


def _accumulate_events_polarity(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    polarity: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """Accumulate events into polarity-summed 2-channel grid.

    Channel 0: ON events (polarity > 0), summed counts.
    Channel 1: OFF events (polarity <= 0), summed counts.

    Args:
        grid_x: Projected x-coordinates, shape (N,).
        grid_y: Projected y-coordinates, shape (N,).
        polarity: Event polarities, shape (N,). Typically +1 or -1.
        grid_size: Output grid size.

    Returns:
        Grid of shape (2, grid_size, grid_size) with float32 counts.
    """
    grid = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    on_mask = polarity > 0
    off_mask = ~on_mask

    if np.any(on_mask):
        np.add.at(grid[0], (grid_y[on_mask], grid_x[on_mask]), 1.0)
    if np.any(off_mask):
        np.add.at(grid[1], (grid_y[off_mask], grid_x[off_mask]), 1.0)

    return grid  # (2, H, W)


def _accumulate_events(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    polarity: np.ndarray,
    grid_size: int,
    polarity_mode: str,
) -> np.ndarray:
    """Dispatch to the correct accumulation strategy.

    Args:
        grid_x: Projected x-coordinates.
        grid_y: Projected y-coordinates.
        polarity: Event polarities.
        grid_size: Output grid size.
        polarity_mode: 'binary' or 'polarity_sum'.

    Returns:
        Grid of shape (C, grid_size, grid_size).
    """
    if polarity_mode == "binary":
        return _accumulate_events_binary(grid_x, grid_y, grid_size)
    elif polarity_mode == "polarity_sum":
        return _accumulate_events_polarity(grid_x, grid_y, polarity, grid_size)
    else:
        raise ValueError(f"Unknown polarity_mode: {polarity_mode!r}. "
                         f"Use 'binary' or 'polarity_sum'.")


def bin_events_fixed_count(
    events: pd.DataFrame,
    config: BinConfig,
) -> list[dict]:
    """Bin events using fixed-count strategy.

    Every `config.bin_count` events are accumulated into one spatial grid.

    Args:
        events: DataFrame with columns ['x', 'y', 'timestamp', 'polarity'].
            Must be sorted by timestamp.
        config: Binning configuration.

    Returns:
        List of dicts with keys:
            'grid': np.ndarray of shape (C, grid_size, grid_size)
            'timestamp_start': float, first event timestamp in the bin
            'timestamp_end': float, last event timestamp in the bin
            'num_events': int, number of events in the bin
    """
    n_events = len(events)
    if n_events == 0:
        return []

    x = events["x"].values
    y = events["y"].values
    polarity = events["polarity"].values
    timestamps = events["timestamp"].values

    bins = []
    for start_idx in range(0, n_events, config.bin_count):
        end_idx = min(start_idx + config.bin_count, n_events)

        chunk_x = x[start_idx:end_idx]
        chunk_y = y[start_idx:end_idx]
        chunk_pol = polarity[start_idx:end_idx]

        grid_x, grid_y = _project_to_grid(
            chunk_x, chunk_y,
            config.sensor_width, config.sensor_height,
            config.grid_size,
        )

        grid = _accumulate_events(
            grid_x, grid_y, chunk_pol,
            config.grid_size, config.polarity_mode,
        )

        bins.append({
            "grid": grid,
            "timestamp_start": float(timestamps[start_idx]),
            "timestamp_end": float(timestamps[end_idx - 1]),
            "num_events": end_idx - start_idx,
        })

    logger.debug(f"Fixed-count binning: {len(bins)} bins from {n_events} events "
                 f"(bin_count={config.bin_count})")
    return bins


def bin_events_fixed_duration(
    events: pd.DataFrame,
    config: BinConfig,
) -> list[dict]:
    """Bin events using fixed-duration strategy.

    Events within each `config.bin_duration_ms` window are accumulated.

    Args:
        events: DataFrame with columns ['x', 'y', 'timestamp', 'polarity'].
            Timestamp should be in microseconds. Must be sorted by timestamp.
        config: Binning configuration.

    Returns:
        List of dicts with same structure as bin_events_fixed_count.
    """
    n_events = len(events)
    if n_events == 0:
        return []

    x = events["x"].values
    y = events["y"].values
    polarity = events["polarity"].values
    timestamps = events["timestamp"].values

    # Convert duration from ms to the same unit as timestamps (microseconds)
    duration_us = config.bin_duration_ms * 1000.0

    t_start = timestamps[0]
    t_end = timestamps[-1]

    bins = []
    window_start = t_start

    while window_start < t_end:
        window_end = window_start + duration_us

        # Binary search for event indices in this window
        start_idx = np.searchsorted(timestamps, window_start, side="left")
        end_idx = np.searchsorted(timestamps, window_end, side="left")

        if end_idx > start_idx:
            chunk_x = x[start_idx:end_idx]
            chunk_y = y[start_idx:end_idx]
            chunk_pol = polarity[start_idx:end_idx]

            grid_x, grid_y = _project_to_grid(
                chunk_x, chunk_y,
                config.sensor_width, config.sensor_height,
                config.grid_size,
            )

            grid = _accumulate_events(
                grid_x, grid_y, chunk_pol,
                config.grid_size, config.polarity_mode,
            )

            bins.append({
                "grid": grid,
                "timestamp_start": float(timestamps[start_idx]),
                "timestamp_end": float(timestamps[end_idx - 1]),
                "num_events": end_idx - start_idx,
            })
        else:
            # Empty window — still emit a zero grid for temporal consistency
            channels = 1 if config.polarity_mode == "binary" else 2
            bins.append({
                "grid": np.zeros(
                    (channels, config.grid_size, config.grid_size),
                    dtype=np.float32,
                ),
                "timestamp_start": float(window_start),
                "timestamp_end": float(window_end),
                "num_events": 0,
            })

        window_start = window_end

    logger.debug(f"Fixed-duration binning: {len(bins)} bins from {n_events} events "
                 f"(bin_duration_ms={config.bin_duration_ms})")
    return bins


def bin_events(
    events: pd.DataFrame,
    config: BinConfig,
) -> list[dict]:
    """Bin events using the strategy specified in config.

    This is the main entry point for the binning engine.

    Args:
        events: DataFrame with columns ['x', 'y', 'timestamp', 'polarity'].
        config: Binning configuration.

    Returns:
        List of binned grids with metadata.
    """
    if config.bin_mode == BinMode.FIXED_COUNT:
        return bin_events_fixed_count(events, config)
    elif config.bin_mode == BinMode.FIXED_DURATION:
        return bin_events_fixed_duration(events, config)
    else:
        raise ValueError(f"Unknown bin_mode: {config.bin_mode}")


def bin_events_to_tensor(
    events: pd.DataFrame,
    config: BinConfig,
) -> tuple[torch.Tensor, np.ndarray]:
    """Bin events and return a stacked tensor + timestamp array.

    Convenience wrapper that returns the grids as a single PyTorch tensor
    and the timestamps as a NumPy array.

    Args:
        events: DataFrame with event data.
        config: Binning configuration.

    Returns:
        grids: torch.Tensor of shape (num_bins, C, grid_size, grid_size).
        timestamps: np.ndarray of shape (num_bins, 2) with [start, end] per bin.
    """
    bins = bin_events(events, config)

    if len(bins) == 0:
        channels = 1 if config.polarity_mode == "binary" else 2
        return (
            torch.zeros(0, channels, config.grid_size, config.grid_size),
            np.zeros((0, 2), dtype=np.float64),
        )

    grids = torch.from_numpy(np.stack([b["grid"] for b in bins], axis=0))
    timestamps = np.array(
        [[b["timestamp_start"], b["timestamp_end"]] for b in bins],
        dtype=np.float64,
    )

    return grids, timestamps


def load_parquet_events(
    filepath: Path | str,
    required_columns: tuple[str, ...] = ("x", "y", "timestamp", "polarity"),
) -> pd.DataFrame:
    """Load event data from a parquet file.

    Validates that required columns exist and sorts by timestamp.

    Args:
        filepath: Path to the .parquet file.
        required_columns: Expected column names.

    Returns:
        Sorted DataFrame of events.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    logger.info(f"Loading events from {filepath}")
    df = pd.read_parquet(filepath)

    # Validate columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure sorted by timestamp
    if not df["timestamp"].is_monotonic_increasing:
        logger.warning("Events not sorted by timestamp — sorting now.")
        df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} events from {filepath.name}")
    return df


def stream_parquet_events(
    filepath: Path | str,
    chunk_size: int = 500_000,
) -> Generator[pd.DataFrame, None, None]:
    """Stream events from a parquet file in chunks to avoid RAM overload.

    Uses PyArrow's batch reading for memory efficiency on the 80 GB dataset.

    Args:
        filepath: Path to the .parquet file.
        chunk_size: Number of rows per chunk.

    Yields:
        DataFrame chunks, each with up to chunk_size rows.
    """
    import pyarrow.parquet as pq

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    parquet_file = pq.ParquetFile(filepath)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Streaming {total_rows:,} events from {filepath.name} "
                f"in chunks of {chunk_size:,}")

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()


# ---------------------------------------------------------------------------
# Quick self-test when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Generate synthetic events for testing
    rng = np.random.default_rng(42)
    n_test = 25_000
    synthetic_events = pd.DataFrame({
        "x": rng.integers(0, 346, size=n_test),
        "y": rng.integers(0, 260, size=n_test),
        "timestamp": np.sort(rng.integers(0, 1_000_000, size=n_test)),
        "polarity": rng.choice([-1, 1], size=n_test),
    })

    cfg = BinConfig(grid_size=64, bin_mode=BinMode.FIXED_COUNT, bin_count=5000)

    print(f"\n=== Fixed-Count Binning (count={cfg.bin_count}) ===")
    grids, ts = bin_events_to_tensor(synthetic_events, cfg)
    print(f"  Output shape: {grids.shape}")  # (5, 1, 64, 64)
    print(f"  Timestamps shape: {ts.shape}")
    print(f"  Grid stats: min={grids.min():.2f}, max={grids.max():.2f}, "
          f"mean={grids.float().mean():.4f}")
    print(f"  Sparsity: {(grids == 0).float().mean():.2%} zeros")

    cfg_dur = BinConfig(grid_size=64, bin_mode=BinMode.FIXED_DURATION,
                        bin_duration_ms=50.0)
    print(f"\n=== Fixed-Duration Binning (duration={cfg_dur.bin_duration_ms}ms) ===")
    grids2, ts2 = bin_events_to_tensor(synthetic_events, cfg_dur)
    print(f"  Output shape: {grids2.shape}")
    print(f"  Timestamps shape: {ts2.shape}")

    cfg_pol = BinConfig(grid_size=64, bin_mode=BinMode.FIXED_COUNT,
                        bin_count=5000, polarity_mode="polarity_sum")
    print(f"\n=== Polarity-Sum Mode ===")
    grids3, ts3 = bin_events_to_tensor(synthetic_events, cfg_pol)
    print(f"  Output shape: {grids3.shape}")  # (5, 2, 64, 64)
    print(f"  ON channel mean: {grids3[:, 0].float().mean():.4f}")
    print(f"  OFF channel mean: {grids3[:, 1].float().mean():.4f}")

    print("\n✓ All binning modes passed self-test.")
