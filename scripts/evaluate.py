"""
EFT-VPR Full Evaluation Script.

Runs the complete benchmark suite comparing Standard VPR vs Forecasting VPR
across all test traversals. Also runs sensor dropout sweep.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \\
        --data data/processed --checkpoint checkpoints/transformer_frozen_best.pt

Outputs:
    - Console summary table
    - JSON results file
    - Dropout sweep data for plotting
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.snn_encoder import SNNEncoder, SNNEncoderConfig
from src.models.forecasting_transformer import ForecastingTransformer, TransformerConfig
from src.vpr.map_database import MapDatabase, MapBuilder
from src.vpr.inference import EFTVPRPipeline, StandardVPRBaseline
from src.vpr.robustness import SensorDropoutTest
from src.evaluation.metrics import (
    evaluate_vpr,
    print_evaluation_table,
    generate_results_summary,
    compute_localization_errors,
)

logger = logging.getLogger(__name__)


def load_models(
    checkpoint_path: Path,
    config: dict,
    device: torch.device,
) -> tuple[SNNEncoder, ForecastingTransformer]:
    """Load encoder and transformer from checkpoint."""
    from src.training.train_transformer import TransformerTrainer

    encoder, transformer, ckpt = TransformerTrainer.load_checkpoint(
        checkpoint_path, device=device,
    )
    logger.info(f"Loaded models from {checkpoint_path} (epoch {ckpt['epoch'] + 1})")
    return encoder, transformer


def build_or_load_map(
    encoder: SNNEncoder,
    map_files: list[Path],
    map_cache: Path,
    device: torch.device,
    use_gpu_index: bool = False,
) -> MapDatabase:
    """Build map database or load from cache."""
    if map_cache.with_suffix(".faiss").exists():
        logger.info(f"Loading cached map from {map_cache}")
        return MapDatabase.load(map_cache, use_gpu=use_gpu_index)

    logger.info(f"Building map from {len(map_files)} files...")
    builder = MapBuilder(encoder=encoder, device=device)
    map_db = builder.build_from_h5(map_files, use_gpu_index=use_gpu_index)

    map_db.save(map_cache)
    logger.info(f"Map saved to {map_cache} ({map_db.size} entries)")
    return map_db


def evaluate_traversal(
    h5_path: Path,
    encoder: SNNEncoder,
    transformer: ForecastingTransformer,
    map_db: MapDatabase,
    device: torch.device,
    sequence_length: int = 10,
    top_k: int = 20,
    threshold_m: float = 25.0,
    batch_size: int = 128,
) -> tuple:
    """Evaluate both methods on one test traversal.

    Returns:
        Tuple of (standard_result, forecasting_result).
    """
    with h5py.File(h5_path, "r") as f:
        bins = torch.tensor(f["bins"][:], dtype=torch.float32)
        gps = f["gps"][:]

    n_bins = bins.shape[0]
    traversal_name = h5_path.stem

    # --- Standard VPR: encode each bin → search FAISS ---
    baseline = StandardVPRBaseline(encoder=encoder, map_db=map_db, device=device)

    all_standard_results = []
    for start in tqdm(range(0, n_bins, batch_size), desc="Standard VPR", leave=False):
        end = min(start + batch_size, n_bins)
        batch = bins[start:end]
        results = baseline.localize(batch, top_k=top_k)
        all_standard_results.extend(results)

    # Extract top-1 GPS and top-K GPS for recall computation
    standard_top1_gps = np.zeros((n_bins, 2), dtype=np.float64)
    standard_topk_gps = []
    for i, r in enumerate(all_standard_results):
        if r:
            standard_top1_gps[i] = r[0]["gps"]
            standard_topk_gps.append(
                np.array([match["gps"] for match in r])
            )
        else:
            standard_topk_gps.append(np.zeros((0, 2)))

    standard_result = evaluate_vpr(
        predicted_gps=standard_top1_gps,
        ground_truth_gps=gps,
        method="standard_vpr",
        traversal=traversal_name,
        threshold_m=threshold_m,
        all_retrieved_gps=standard_topk_gps,
    )

    # --- Forecasting VPR: use temporal sequences ---
    pipeline = EFTVPRPipeline(
        encoder=encoder, transformer=transformer,
        map_db=map_db, device=device,
        sequence_length=sequence_length,
    )

    # Evaluate using batch forecasting for bins with enough context
    forecast_top1_gps = np.zeros((n_bins, 2), dtype=np.float64)
    forecast_topk_gps = []

    # First `sequence_length` bins: fall back to standard
    for i in range(min(sequence_length, n_bins)):
        if all_standard_results[i]:
            forecast_top1_gps[i] = all_standard_results[i][0]["gps"]
            forecast_topk_gps.append(
                np.array([m["gps"] for m in all_standard_results[i]])
            )
        else:
            forecast_topk_gps.append(np.zeros((0, 2)))

    # Remaining bins: use transformer
    # Use smaller batch size so B * T <= standard batch size to avoid OOM
    seq_batch_size = max(1, batch_size // sequence_length)
    
    for start in tqdm(range(sequence_length, n_bins, seq_batch_size), desc="Forecasting VPR", leave=False):
        end = min(start + seq_batch_size, n_bins)
        batch_indices = list(range(start, end))

        batch_seqs = []
        for idx in batch_indices:
            batch_seqs.append(bins[idx - sequence_length : idx])
            
        batch_tensor = torch.stack(batch_seqs)  # (B, T, C, H, W)
        results_batch = pipeline.localize_batch_forecasting(batch_tensor, top_k=top_k)
        
        for i, idx in enumerate(batch_indices):
            r = results_batch[i]
            if r:
                forecast_top1_gps[idx] = r[0]["gps"]
                forecast_topk_gps.append(
                    np.array([match["gps"] for match in r])
                )
            else:
                forecast_topk_gps.append(np.zeros((0, 2)))

    forecasting_result = evaluate_vpr(
        predicted_gps=forecast_top1_gps,
        ground_truth_gps=gps,
        method="forecasting_vpr",
        traversal=traversal_name,
        threshold_m=threshold_m,
        all_retrieved_gps=forecast_topk_gps,
    )

    return standard_result, forecasting_result


def run_dropout_evaluation(
    encoder: SNNEncoder,
    transformer: ForecastingTransformer,
    map_db: MapDatabase,
    test_files: list[Path],
    device: torch.device,
    sequence_length: int = 10,
    drop_counts: list[int] = None,
    n_trials: int = 5,
) -> dict:
    """Run sensor dropout sweep across test traversals."""
    if drop_counts is None:
        drop_counts = [5, 10, 15, 20]

    dropout_tester = SensorDropoutTest(
        encoder=encoder, transformer=transformer,
        map_db=map_db, device=device,
        sequence_length=sequence_length,
    )

    all_sweep_results = {}
    for h5_path in tqdm(test_files, desc="Dropout sweep"):
        with h5py.File(h5_path, "r") as f:
            grids = torch.tensor(f["bins"][:], dtype=torch.float32)
            gps = f["gps"][:]

        sweep = dropout_tester.run_sweep(
            grids, gps,
            drop_counts=drop_counts,
            n_trials=n_trials,
        )

        for n_drop, trials in sweep.items():
            if n_drop not in all_sweep_results:
                all_sweep_results[n_drop] = []
            all_sweep_results[n_drop].extend(trials)

    summary = SensorDropoutTest.summarize_sweep(all_sweep_results)
    SensorDropoutTest.print_sweep_summary(summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description="EFT-VPR Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default="data/processed",
                        help="Directory with test .h5 files")
    parser.add_argument("--map-data", type=str, default=None,
                        help="Map reference data (default: same as --data)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to transformer checkpoint .pt")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=25.0,
                        help="Recall threshold in meters")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dropout-sweep", action="store_true",
                        help="Run sensor dropout sweep")
    parser.add_argument("--dropout-counts", type=int, nargs="+",
                        default=[5, 10, 15, 20])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # Load models
    encoder, transformer = load_models(Path(args.checkpoint), config, device)

    # Find test files
    data_dir = Path(args.data)
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        logger.error(f"No .h5 files found in {data_dir}")
        return

    # Build or load map (use map_data or first 80% of data)
    map_data_dir = Path(args.map_data) if args.map_data else data_dir
    map_files = sorted(map_data_dir.glob("*.h5"))
    split_idx = max(1, int(len(map_files) * 0.8))
    map_ref_files = map_files[:split_idx]
    test_files = map_files[split_idx:] if len(map_files) > 1 else map_files

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_cache = output_dir / "reference_map"
    map_db = build_or_load_map(
        encoder, map_ref_files, map_cache, device,
        use_gpu_index=(device.type == "cuda"),
    )

    data_cfg = config.get("data", {})
    seq_len = data_cfg.get("sequence_length", 10)

    # --- Evaluate all test traversals ---
    logger.info(f"Evaluating {len(test_files)} test traversals...")
    start_time = time.time()
    all_results = []

    for h5_path in tqdm(test_files, desc="Evaluating"):
        std_result, fcast_result = evaluate_traversal(
            h5_path, encoder, transformer, map_db, device,
            sequence_length=seq_len,
            top_k=args.top_k,
            threshold_m=args.threshold,
        )
        all_results.extend([std_result, fcast_result])

    eval_time = time.time() - start_time
    logger.info(f"Evaluation complete in {eval_time:.1f}s")

    # Print results
    print_evaluation_table(all_results)

    # Generate summary
    summary = generate_results_summary(all_results)
    summary["evaluation_time_s"] = eval_time
    summary["device"] = str(device)
    summary["checkpoint"] = str(args.checkpoint)
    summary["threshold_m"] = args.threshold

    # --- Dropout sweep (optional) ---
    dropout_summary = None
    if args.dropout_sweep:
        logger.info("Running sensor dropout sweep...")
        dropout_summary = run_dropout_evaluation(
            encoder, transformer, map_db, test_files, device,
            sequence_length=seq_len,
            drop_counts=args.dropout_counts,
        )
        summary["dropout_sweep"] = {
            str(k): v for k, v in dropout_summary.items()
        }

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    print(f"\n✓ Evaluation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
