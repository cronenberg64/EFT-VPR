"""
Brisbane Event VPR Dataset Download Utility.

The dataset is hosted by QUT Centre for Robotics and contains:
  - Parquet files with event data (x, y, timestamp, polarity)
  - GPS/NMEA files with geolocation
  - RGB frame archives (optional, not used for EFT-VPR)

Dataset page: https://research.qut.edu.au/qcr/datasets/brisbane-event-vpr-dataset/

Note: The full dataset is ~80 GB. This script provides structured download
with resume support and integrity checking.
"""

import hashlib
import logging
import sys
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# Dataset manifest
# =============================================================================
# These URLs should be updated once the actual download links are confirmed.
# The Brisbane Event VPR dataset is available from QCR's data portal.
# Placeholder structure based on documented dataset format:
#   - 6 traversals of the route
#   - Each traversal has event data (parquet) and GPS (csv)
# =============================================================================

DATASET_INFO = {
    "name": "Brisbane Event VPR Dataset",
    "url": "https://research.qut.edu.au/qcr/datasets/brisbane-event-vpr-dataset/",
    "description": (
        "48km of repeated traversals captured with DAVIS346 event camera "
        "(346x260 resolution). 6 traversals of an 8km route under varying "
        "lighting and weather conditions."
    ),
    "traversals": [
        "sunset1",
        "sunset2",
        "daytime",
        "morning",
        "sunrise",
        "night",
    ],
}


def download_file(
    url: str,
    output_path: Path,
    chunk_size: int = 8192,
    resume: bool = True,
) -> bool:
    """Download a file with progress bar and resume support.

    Args:
        url: URL to download from.
        output_path: Local path to save the file.
        chunk_size: Download chunk size in bytes.
        resume: Whether to resume partial downloads.

    Returns:
        True if download succeeded, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    initial_size = 0

    if resume and output_path.exists():
        initial_size = output_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        logger.info(f"Resuming download from {initial_size:,} bytes")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 416:
            logger.info(f"File already fully downloaded: {output_path.name}")
            return True

        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        if initial_size > 0:
            total_size += initial_size

        mode = "ab" if initial_size > 0 else "wb"

        with (
            open(output_path, mode) as f,
            tqdm(
                total=total_size,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Downloaded: {output_path.name} ({output_path.stat().st_size:,} bytes)")
        return True

    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        return False


def verify_file_integrity(filepath: Path, expected_md5: str | None = None) -> bool:
    """Verify file integrity via MD5 checksum.

    Args:
        filepath: Path to the file.
        expected_md5: Expected MD5 hash string.

    Returns:
        True if file exists and hash matches (or no hash provided).
    """
    if not filepath.exists():
        return False

    if expected_md5 is None:
        return True

    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    actual = md5.hexdigest()
    if actual != expected_md5:
        logger.warning(f"MD5 mismatch for {filepath.name}: "
                       f"expected {expected_md5}, got {actual}")
        return False

    return True


def print_dataset_info():
    """Print information about the Brisbane Event VPR dataset."""
    info = DATASET_INFO
    print(f"\n{'='*60}")
    print(f"  {info['name']}")
    print(f"{'='*60}")
    print(f"\n  {info['description']}")
    print(f"\n  Download page: {info['url']}")
    print(f"\n  Traversals ({len(info['traversals'])}):")
    for t in info["traversals"]:
        print(f"    - {t}")
    print(f"\n  Expected format:")
    print(f"    - Event data: .parquet files (columns: x, y, timestamp, polarity)")
    print(f"    - GPS data: .csv files (columns: timestamp, latitude, longitude)")
    print(f"    - Sensor: DAVIS346 (346×260 pixels)")
    print(f"\n  Total size: ~80 GB")
    print(f"\n  Place downloaded files in: data/raw/")
    print(f"{'='*60}\n")


def setup_data_directory(data_dir: Path):
    """Create the expected data directory structure.

    Args:
        data_dir: Root data directory (e.g., data/raw/).
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    for traversal in DATASET_INFO["traversals"]:
        (data_dir / traversal).mkdir(exist_ok=True)

    logger.info(f"Data directory structure created at {data_dir}")


def create_sample_data(output_dir: Path, num_events: int = 50_000):
    """Create a small synthetic dataset for development and testing.

    Generates fake event data with realistic statistics, allowing the full
    pipeline to be tested without downloading the 80 GB dataset.

    Args:
        output_dir: Directory to save the sample data.
        num_events: Number of synthetic events to generate.
    """
    import numpy as np
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Simulate a vehicle driving a route (smooth GPS trajectory)
    n_gps_points = num_events // 100
    t_gps = np.linspace(0, 1_000_000, n_gps_points)  # 1 second in microseconds

    # Simulate a path in Brisbane coordinates
    base_lat, base_lon = -27.5, 152.9
    lat = base_lat + np.cumsum(rng.normal(0, 0.00001, n_gps_points))
    lon = base_lon + np.cumsum(rng.normal(0, 0.00001, n_gps_points))

    gps_df = pd.DataFrame({
        "timestamp": t_gps,
        "latitude": lat,
        "longitude": lon,
    })
    gps_path = output_dir / "sample_gps.csv"
    gps_df.to_csv(gps_path, index=False)

    # Generate synthetic events
    events_df = pd.DataFrame({
        "x": rng.integers(0, 346, size=num_events),
        "y": rng.integers(0, 260, size=num_events),
        "timestamp": np.sort(rng.integers(0, 1_000_000, size=num_events)),
        "polarity": rng.choice([-1, 1], size=num_events),
    })

    event_path = output_dir / "sample.parquet"
    events_df.to_parquet(event_path, index=False)

    print(f"✓ Created sample dataset:")
    print(f"  Events: {event_path} ({num_events:,} events)")
    print(f"  GPS:    {gps_path} ({n_gps_points} waypoints)")
    print(f"\n  Use with:")
    print(f"    python scripts/preprocess.py --input {output_dir} "
          f"--output data/processed --grid-size 64")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Brisbane Event VPR Dataset download & setup utility."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Info command
    subparsers.add_parser("info", help="Print dataset information")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Create data directory structure")
    setup_parser.add_argument("--data-dir", type=str, default="data/raw")

    # Sample command
    sample_parser = subparsers.add_parser(
        "sample", help="Create synthetic sample data for testing"
    )
    sample_parser.add_argument("--output", type=str, default="data/raw")
    sample_parser.add_argument("--num-events", type=int, default=50_000)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "info":
        print_dataset_info()
    elif args.command == "setup":
        setup_data_directory(Path(args.data_dir))
    elif args.command == "sample":
        create_sample_data(Path(args.output), args.num_events)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/download_dataset.py info")
        print("  python scripts/download_dataset.py setup --data-dir data/raw")
        print("  python scripts/download_dataset.py sample --output data/raw")


if __name__ == "__main__":
    main()
