"""
Preprocessing Pipeline for Brisbane Event VPR Dataset.

Reads raw parquet event files, bins them into spatial grids, pairs with GPS,
and saves as HDF5 files optimized for DataLoader throughput.

Usage:
    python scripts/preprocess.py --input data/raw/ --output data/processed/ \\
        --grid-size 64 --bin-mode fixed_count --bin-count 5000

Safety: Processes files one at a time with chunked parquet reading.
Never loads the entire 80 GB dataset into RAM.
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.event_binning import (
    BinConfig,
    BinMode,
    bin_events,
    stream_parquet_events,
    load_parquet_events,
)

logger = logging.getLogger(__name__)


def load_gps_data(gps_path: Path) -> pd.DataFrame | None:
    """Load GPS data from NMEA or CSV file.

    Attempts to parse common GPS formats. Returns a DataFrame with
    'timestamp', 'latitude', 'longitude' columns, or None if not found.

    Args:
        gps_path: Path to GPS file (.csv, .txt, or .nmea).

    Returns:
        DataFrame with GPS data or None.
    """
    if not gps_path.exists():
        logger.warning(f"GPS file not found: {gps_path}")
        return None

    suffix = gps_path.suffix.lower()

    if suffix in (".csv", ".txt"):
        try:
            df = pd.read_csv(gps_path)
            # Try common column naming conventions
            col_map = {}
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ("lat", "latitude"):
                    col_map[col] = "latitude"
                elif cl in ("lon", "lng", "longitude"):
                    col_map[col] = "longitude"
                elif cl in ("ts", "timestamp", "time", "t"):
                    col_map[col] = "timestamp"
            df = df.rename(columns=col_map)

            required = {"timestamp", "latitude", "longitude"}
            if not required.issubset(set(df.columns)):
                logger.warning(f"GPS file missing columns. Found: {list(df.columns)}")
                return None

            return df[["timestamp", "latitude", "longitude"]]
        except Exception as e:
            logger.warning(f"Failed to parse GPS file {gps_path}: {e}")
            return None

    logger.warning(f"Unsupported GPS file format: {suffix}")
    return None


def interpolate_gps(
    gps_df: pd.DataFrame,
    bin_timestamps: np.ndarray,
) -> np.ndarray:
    """Interpolate GPS coordinates to match bin timestamps.

    Uses linear interpolation between known GPS waypoints.

    Args:
        gps_df: DataFrame with 'timestamp', 'latitude', 'longitude'.
        bin_timestamps: Array of shape (N, 2) with [start, end] per bin.
            Uses the midpoint of each bin for interpolation.

    Returns:
        GPS coordinates of shape (N, 2) as [latitude, longitude].
    """
    bin_midpoints = (bin_timestamps[:, 0] + bin_timestamps[:, 1]) / 2.0

    gps_ts = gps_df["timestamp"].values.astype(np.float64)
    gps_lat = gps_df["latitude"].values.astype(np.float64)
    gps_lon = gps_df["longitude"].values.astype(np.float64)

    interp_lat = np.interp(bin_midpoints, gps_ts, gps_lat)
    interp_lon = np.interp(bin_midpoints, gps_ts, gps_lon)

    return np.stack([interp_lat, interp_lon], axis=1)


def preprocess_single_file(
    event_path: Path,
    output_path: Path,
    config: BinConfig,
    gps_path: Path | None = None,
    chunk_size: int = 500_000,
) -> dict:
    """Preprocess a single parquet event file into an HDF5 file.

    Reads events in chunks, bins them, pairs with GPS, and writes to HDF5.

    Args:
        event_path: Path to input .parquet file.
        output_path: Path for output .h5 file.
        config: Binning configuration.
        gps_path: Optional path to GPS data file.
        chunk_size: Number of events to read per chunk.

    Returns:
        Dict with processing statistics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load GPS if available
    gps_df = None
    if gps_path is not None:
        gps_df = load_gps_data(gps_path)

    # Process events — for small files, load all at once; for large, stream
    file_size_mb = event_path.stat().st_size / (1024 * 1024)

    if file_size_mb < 500:  # Under 500 MB: load entirely
        logger.info(f"Loading {event_path.name} ({file_size_mb:.0f} MB) into memory")
        events_df = load_parquet_events(event_path)
        all_bins = bin_events(events_df, config)
    else:
        logger.info(f"Streaming {event_path.name} ({file_size_mb:.0f} MB) in chunks")
        all_bins = []
        for chunk_df in stream_parquet_events(event_path, chunk_size=chunk_size):
            chunk_bins = bin_events(chunk_df, config)
            all_bins.extend(chunk_bins)

    if len(all_bins) == 0:
        logger.warning(f"No bins generated from {event_path.name}")
        return {"file": str(event_path), "bins": 0, "status": "empty"}

    # Stack grids and timestamps
    grids = np.stack([b["grid"] for b in all_bins], axis=0)  # (N, C, H, W)
    timestamps = np.array(
        [[b["timestamp_start"], b["timestamp_end"]] for b in all_bins],
        dtype=np.float64,
    )  # (N, 2)

    # GPS interpolation
    if gps_df is not None:
        gps_coords = interpolate_gps(gps_df, timestamps)
    else:
        # Placeholder zeros if no GPS available
        gps_coords = np.zeros((len(all_bins), 2), dtype=np.float64)

    # Write HDF5 with chunked storage for efficient slicing
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "bins", data=grids, dtype="float32",
            chunks=(min(64, grids.shape[0]),) + grids.shape[1:],
            compression="gzip", compression_opts=4,
        )
        f.create_dataset("timestamps", data=timestamps, dtype="float64")
        f.create_dataset("gps", data=gps_coords, dtype="float64")

        # Metadata
        f.attrs["grid_size"] = config.grid_size
        f.attrs["bin_mode"] = config.bin_mode.value
        f.attrs["bin_count"] = config.bin_count
        f.attrs["bin_duration_ms"] = config.bin_duration_ms
        f.attrs["polarity_mode"] = config.polarity_mode
        f.attrs["source_file"] = event_path.name
        f.attrs["num_bins"] = len(all_bins)

    stats = {
        "file": event_path.name,
        "output": output_path.name,
        "bins": len(all_bins),
        "grid_shape": list(grids.shape),
        "has_gps": gps_df is not None,
        "size_mb": output_path.stat().st_size / (1024 * 1024),
    }
    logger.info(f"Saved {stats['bins']} bins to {output_path.name} "
                f"({stats['size_mb']:.1f} MB)")
    return stats


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    config: BinConfig,
    chunk_size: int = 500_000,
) -> list[dict]:
    """Preprocess all parquet files in a directory.

    Searches for .parquet files and matching GPS files (same stem + .csv/.txt).

    Args:
        input_dir: Directory containing raw .parquet files.
        output_dir: Directory for output .h5 files.
        config: Binning configuration.
        chunk_size: Events per chunk for streaming.

    Returns:
        List of processing statistics dicts.
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No .parquet files found in {input_dir}")
        return []

    logger.info(f"Found {len(parquet_files)} parquet files in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for pf in tqdm(parquet_files, desc="Preprocessing"):
        out_name = pf.stem + ".h5"
        out_path = output_dir / out_name

        # Look for matching GPS file
        gps_path = None
        for ext in (".csv", ".txt", ".nmea"):
            candidate = pf.parent / (pf.stem + "_gps" + ext)
            if candidate.exists():
                gps_path = candidate
                break
            candidate = pf.parent / (pf.stem + ext)
            if candidate.exists():
                gps_path = candidate
                break

        stats = preprocess_single_file(
            event_path=pf,
            output_path=out_path,
            config=config,
            gps_path=gps_path,
            chunk_size=chunk_size,
        )
        all_stats.append(stats)

    total_bins = sum(s["bins"] for s in all_stats)
    logger.info(f"Preprocessing complete: {total_bins} total bins from "
                f"{len(parquet_files)} files")
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Brisbane Event VPR dataset into HDF5 bins."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing .parquet event files",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for .h5 preprocessed files",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--bin-mode", type=str, default="fixed_count",
                        choices=["fixed_count", "fixed_duration"])
    parser.add_argument("--bin-count", type=int, default=5000)
    parser.add_argument("--bin-duration-ms", type=float, default=50.0)
    parser.add_argument("--polarity", type=str, default="binary",
                        choices=["binary", "polarity_sum"])
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config from YAML or CLI args
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        config = BinConfig.from_dict(yaml_cfg)
    else:
        config = BinConfig(
            grid_size=args.grid_size,
            bin_mode=BinMode(args.bin_mode),
            bin_count=args.bin_count,
            bin_duration_ms=args.bin_duration_ms,
            polarity_mode=args.polarity,
        )

    logger.info(f"Config: grid={config.grid_size}, mode={config.bin_mode.value}, "
                f"count={config.bin_count}, polarity={config.polarity_mode}")

    stats = preprocess_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        config=config,
        chunk_size=args.chunk_size,
    )

    # Print summary
    print("\n=== Preprocessing Summary ===")
    for s in stats:
        gps_str = "✓ GPS" if s.get("has_gps") else "✗ no GPS"
        print(f"  {s['file']:40s} → {s['bins']:6d} bins  ({gps_str})")
    print(f"\n  Total: {sum(s['bins'] for s in stats)} bins from {len(stats)} files")


if __name__ == "__main__":
    main()
