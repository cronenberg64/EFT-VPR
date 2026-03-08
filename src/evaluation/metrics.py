"""
Evaluation Metrics for EFT-VPR.

Implements standard Visual Place Recognition metrics:
  - Recall@N: Fraction of queries where the correct place is within
    the top-N retrieved results (using GPS-distance threshold).
  - Mean Localization Error (MLE): Average GPS distance between
    the top-1 predicted location and the ground truth.

All distances use the Haversine formula on GPS coordinates.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.losses.triplet_loss import haversine_distance

logger = logging.getLogger(__name__)


def compute_localization_errors(
    predicted_gps: np.ndarray,
    ground_truth_gps: np.ndarray,
) -> np.ndarray:
    """Compute per-query localization error in meters.

    Args:
        predicted_gps: Shape (N, 2) as [lat, lon].
        ground_truth_gps: Shape (N, 2) as [lat, lon].

    Returns:
        Errors in meters, shape (N,).
    """
    pred = torch.tensor(predicted_gps, dtype=torch.float64)
    gt = torch.tensor(ground_truth_gps, dtype=torch.float64)
    return haversine_distance(pred, gt).numpy()


def recall_at_n(
    errors_m: np.ndarray,
    threshold_m: float = 25.0,
    n_values: list[int] = None,
    all_retrieved_errors: Optional[list[np.ndarray]] = None,
) -> dict[str, float]:
    """Compute Recall@N at a given GPS-distance threshold.

    Two modes:
      1. Simple mode (errors_m only): Uses the top-1 error per query.
         Recall@1 = fraction of queries where top-1 error < threshold.
      2. Full mode (all_retrieved_errors): Uses top-K errors per query.
         Recall@N = fraction where ANY of top-N has error < threshold.

    Args:
        errors_m: Top-1 localization errors in meters, shape (N,).
        threshold_m: Maximum GPS distance for a correct match (meters).
        n_values: List of N values to compute (default: [1, 5, 10, 20]).
        all_retrieved_errors: Optional list of N arrays, each shape (K,),
            containing the GPS errors for the top-K retrieved matches.
            If provided, enables Recall@5/10/20 via any-within-top-K.

    Returns:
        Dict with keys like 'R@1', 'R@5', etc. Values are fractions [0, 1].
    """
    if n_values is None:
        n_values = [1, 5, 10, 20]

    results = {}
    n_queries = len(errors_m)

    for n in n_values:
        if all_retrieved_errors is not None and n > 1:
            # Full mode: check if ANY of top-N is within threshold
            correct = 0
            for query_errors in all_retrieved_errors:
                top_n = query_errors[:n]
                if np.any(top_n < threshold_m):
                    correct += 1
            results[f"R@{n}"] = correct / max(n_queries, 1)
        else:
            # Simple mode: only check top-1
            if n == 1:
                results[f"R@{n}"] = float(np.mean(errors_m < threshold_m))
            else:
                # Can't compute R@N>1 without top-K data; skip
                results[f"R@{n}"] = float("nan")

    return results


def mean_localization_error(errors_m: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for localization errors.

    Args:
        errors_m: Per-query errors in meters, shape (N,).

    Returns:
        Dict with mean, median, std, min, max, p90, p95.
    """
    return {
        "mean_m": float(np.mean(errors_m)),
        "median_m": float(np.median(errors_m)),
        "std_m": float(np.std(errors_m)),
        "min_m": float(np.min(errors_m)),
        "max_m": float(np.max(errors_m)),
        "p90_m": float(np.percentile(errors_m, 90)),
        "p95_m": float(np.percentile(errors_m, 95)),
    }


@dataclass
class EvaluationResult:
    """Complete evaluation result for one method on one traversal.

    Attributes:
        method: 'standard_vpr' or 'forecasting_vpr'.
        traversal: Traversal name (e.g., 'sunset1').
        n_queries: Number of queries evaluated.
        recall: Dict with R@1, R@5, R@10, R@20.
        mle: Mean localization error statistics.
        errors_m: Raw per-query errors.
        threshold_m: GPS threshold used for recall computation.
    """
    method: str
    traversal: str
    n_queries: int
    recall: dict = field(default_factory=dict)
    mle: dict = field(default_factory=dict)
    errors_m: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold_m: float = 25.0


def evaluate_vpr(
    predicted_gps: np.ndarray,
    ground_truth_gps: np.ndarray,
    method: str = "standard_vpr",
    traversal: str = "unknown",
    threshold_m: float = 25.0,
    n_values: list[int] = None,
    all_retrieved_gps: Optional[list[np.ndarray]] = None,
    all_gt_expanded: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Run full VPR evaluation on a set of predictions.

    Args:
        predicted_gps: Top-1 predicted GPS, shape (N, 2).
        ground_truth_gps: Ground truth GPS, shape (N, 2).
        method: Method name for the result.
        traversal: Traversal name.
        threshold_m: Recall threshold in meters.
        n_values: Recall@N values to compute.
        all_retrieved_gps: Optional list of N arrays, shape (K, 2),
            the top-K retrieved GPS for each query.
        all_gt_expanded: Not used (for API consistency).

    Returns:
        EvaluationResult with recall and MLE.
    """
    errors = compute_localization_errors(predicted_gps, ground_truth_gps)

    # Compute top-K errors if full retrieval data available
    all_retrieved_errors = None
    if all_retrieved_gps is not None:
        all_retrieved_errors = []
        for i, retrieved in enumerate(all_retrieved_gps):
            gt_expanded = np.tile(ground_truth_gps[i], (len(retrieved), 1))
            query_errors = compute_localization_errors(retrieved, gt_expanded)
            all_retrieved_errors.append(query_errors)

    recall = recall_at_n(
        errors, threshold_m=threshold_m, n_values=n_values,
        all_retrieved_errors=all_retrieved_errors,
    )
    mle = mean_localization_error(errors)

    return EvaluationResult(
        method=method,
        traversal=traversal,
        n_queries=len(errors),
        recall=recall,
        mle=mle,
        errors_m=errors,
        threshold_m=threshold_m,
    )


def print_evaluation_table(results: list[EvaluationResult]):
    """Pretty-print evaluation results as a table.

    Args:
        results: List of EvaluationResult objects.
    """
    print("\n" + "=" * 90)
    print("  EFT-VPR Evaluation Results")
    print("=" * 90)
    print(f"  {'Method':<20s}  {'Traversal':<12s}  {'R@1':>6s}  {'R@5':>6s}  "
          f"{'R@10':>6s}  {'R@20':>6s}  {'MLE (m)':>8s}  {'Med (m)':>8s}")
    print("-" * 90)

    for r in results:
        r1 = r.recall.get("R@1", float("nan"))
        r5 = r.recall.get("R@5", float("nan"))
        r10 = r.recall.get("R@10", float("nan"))
        r20 = r.recall.get("R@20", float("nan"))
        mle = r.mle.get("mean_m", float("nan"))
        med = r.mle.get("median_m", float("nan"))

        r1_str = f"{r1:.1%}" if not np.isnan(r1) else "  N/A"
        r5_str = f"{r5:.1%}" if not np.isnan(r5) else "  N/A"
        r10_str = f"{r10:.1%}" if not np.isnan(r10) else "  N/A"
        r20_str = f"{r20:.1%}" if not np.isnan(r20) else "  N/A"

        print(f"  {r.method:<20s}  {r.traversal:<12s}  {r1_str:>6s}  {r5_str:>6s}  "
              f"{r10_str:>6s}  {r20_str:>6s}  {mle:>8.1f}  {med:>8.1f}")

    print("=" * 90)


def generate_results_summary(results: list[EvaluationResult]) -> dict:
    """Generate a JSON-serializable summary of all results.

    Args:
        results: List of EvaluationResult objects.

    Returns:
        Dict with per-method aggregated statistics.
    """
    methods = {}
    for r in results:
        if r.method not in methods:
            methods[r.method] = {
                "traversals": [],
                "all_errors": [],
                "all_recalls": [],
            }
        methods[r.method]["traversals"].append(r.traversal)
        methods[r.method]["all_errors"].extend(r.errors_m.tolist())
        methods[r.method]["all_recalls"].append(r.recall)

    summary = {}
    for method, data in methods.items():
        all_errors = np.array(data["all_errors"])
        avg_recall = {}
        for key in data["all_recalls"][0]:
            vals = [r[key] for r in data["all_recalls"]]
            avg_recall[key] = float(np.nanmean(vals))

        summary[method] = {
            "n_traversals": len(data["traversals"]),
            "n_total_queries": len(all_errors),
            "avg_recall": avg_recall,
            "avg_mle": mean_localization_error(all_errors),
        }

    return summary
