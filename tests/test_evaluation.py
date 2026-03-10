"""
Unit tests for Evaluation Metrics.

Tests Recall@N, MLE, EvaluationResult, and summary generation.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_localization_errors,
    recall_at_n,
    mean_localization_error,
    evaluate_vpr,
    generate_results_summary,
    EvaluationResult,
)


# =============================================================================
# Localization Error Tests
# =============================================================================


class TestLocalizationErrors:
    def test_same_point_zero_error(self):
        gps = np.array([[0.0, 0.0], [1.0, 1.0]])
        errors = compute_localization_errors(gps, gps)
        np.testing.assert_allclose(errors, 0.0, atol=1e-6)

    def test_known_distance(self):
        """Roughly verify Haversine: 1 degree lat ≈ 111km."""
        pred = np.array([[0.0, 0.0]])
        gt = np.array([[1.0, 0.0]])
        errors = compute_localization_errors(pred, gt)
        assert 110_000 < errors[0] < 112_000  # ~111km

    def test_output_shape(self):
        pred = np.random.randn(10, 2)
        gt = np.random.randn(10, 2)
        errors = compute_localization_errors(pred, gt)
        assert errors.shape == (10,)

    def test_non_negative(self):
        pred = np.random.randn(20, 2)
        gt = np.random.randn(20, 2)
        errors = compute_localization_errors(pred, gt)
        assert np.all(errors >= 0)


# =============================================================================
# Recall@N Tests
# =============================================================================


class TestRecallAtN:
    def test_perfect_recall(self):
        errors = np.array([1.0, 2.0, 3.0, 5.0, 10.0])  # All < 25m
        recall = recall_at_n(errors, threshold_m=25.0)
        assert recall["R@1"] == 1.0

    def test_zero_recall(self):
        errors = np.array([100.0, 200.0, 300.0])  # All > 25m
        recall = recall_at_n(errors, threshold_m=25.0)
        assert recall["R@1"] == 0.0

    def test_partial_recall(self):
        errors = np.array([10.0, 30.0, 5.0, 50.0])
        recall = recall_at_n(errors, threshold_m=25.0)
        assert recall["R@1"] == 0.5  # 2/4 within 25m

    def test_custom_threshold(self):
        errors = np.array([10.0, 30.0, 5.0, 50.0])
        recall = recall_at_n(errors, threshold_m=10.0)
        assert recall["R@1"] == 0.25  # Only 5m < 10m

    def test_custom_n_values(self):
        errors = np.array([10.0])
        recall = recall_at_n(errors, n_values=[1, 3])
        assert "R@1" in recall
        assert "R@3" in recall

    def test_with_topk_data(self):
        """Full mode with top-K retrieval data."""
        errors = np.array([50.0, 50.0])  # Top-1 is bad
        topk = [
            np.array([50.0, 10.0, 5.0]),   # But top-3 has good match
            np.array([50.0, 40.0, 30.0]),   # No good match in top-3
        ]
        recall = recall_at_n(
            errors, threshold_m=25.0, n_values=[1, 5],
            all_retrieved_errors=topk,
        )
        assert recall["R@1"] == 0.0    # Top-1 both > 25m
        assert recall["R@5"] == 0.5    # 1/2 has a match in top-5


# =============================================================================
# MLE Tests
# =============================================================================


class TestMeanLocalizationError:
    def test_basic_stats(self):
        errors = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mle = mean_localization_error(errors)
        assert mle["mean_m"] == pytest.approx(30.0)
        assert mle["median_m"] == pytest.approx(30.0)
        assert mle["min_m"] == pytest.approx(10.0)
        assert mle["max_m"] == pytest.approx(50.0)

    def test_contains_percentiles(self):
        errors = np.random.rand(100)
        mle = mean_localization_error(errors)
        assert "p90_m" in mle
        assert "p95_m" in mle
        assert mle["p90_m"] <= mle["p95_m"]


# =============================================================================
# Full Evaluation Tests
# =============================================================================


class TestEvaluateVPR:
    def test_basic_evaluation(self):
        pred = np.array([[0.0, 0.0], [0.0, 0.0]])
        gt = np.array([[0.0, 0.0], [0.001, 0.0]])
        result = evaluate_vpr(pred, gt, method="test", traversal="demo")
        assert isinstance(result, EvaluationResult)
        assert result.method == "test"
        assert result.traversal == "demo"
        assert result.n_queries == 2
        assert "R@1" in result.recall
        assert "mean_m" in result.mle

    def test_perfect_match(self):
        gps = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = evaluate_vpr(gps, gps)
        assert result.recall["R@1"] == 1.0
        assert result.mle["mean_m"] < 1.0  # Essentially 0


class TestGenerateResultsSummary:
    def test_summary_structure(self):
        results = [
            EvaluationResult(
                method="standard", traversal="t1", n_queries=10,
                recall={"R@1": 0.5, "R@5": 0.8},
                mle={"mean_m": 20.0},
                errors_m=np.random.rand(10) * 50,
            ),
            EvaluationResult(
                method="forecasting", traversal="t1", n_queries=10,
                recall={"R@1": 0.7, "R@5": 0.9},
                mle={"mean_m": 15.0},
                errors_m=np.random.rand(10) * 30,
            ),
        ]
        summary = generate_results_summary(results)
        assert "standard" in summary
        assert "forecasting" in summary
        assert summary["standard"]["n_total_queries"] == 10

    def test_multi_traversal_aggregation(self):
        results = [
            EvaluationResult(
                method="standard", traversal="t1", n_queries=5,
                recall={"R@1": 0.6},
                mle={"mean_m": 20.0},
                errors_m=np.random.rand(5),
            ),
            EvaluationResult(
                method="standard", traversal="t2", n_queries=5,
                recall={"R@1": 0.8},
                mle={"mean_m": 10.0},
                errors_m=np.random.rand(5),
            ),
        ]
        summary = generate_results_summary(results)
        assert summary["standard"]["n_traversals"] == 2
        assert summary["standard"]["n_total_queries"] == 10
        # Average R@1 should be 0.7
        assert summary["standard"]["avg_recall"]["R@1"] == pytest.approx(0.7)
