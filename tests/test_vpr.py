"""
Unit tests for Phase 4: VPR Integration.

Tests FAISS map database, inference pipeline, sensor dropout robustness,
and standard vs forecasting VPR comparison with synthetic data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.snn_encoder import SNNEncoder, SNNEncoderConfig
from src.models.forecasting_transformer import ForecastingTransformer, TransformerConfig
from src.vpr.map_database import MapDatabase, MapEntry
from src.vpr.inference import EFTVPRPipeline, StandardVPRBaseline
from src.vpr.robustness import SensorDropoutTest, compare_standard_vs_forecasting


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def embedding_dim():
    return 256


@pytest.fixture
def encoder():
    config = SNNEncoderConfig(
        in_channels=1, channels=[32, 64, 128],
        embedding_dim=256, beta_init=0.9, grid_size=64,
    )
    return SNNEncoder(config)


@pytest.fixture
def transformer():
    config = TransformerConfig(
        d_model=256, nhead=8, num_layers=4,
        dim_feedforward=1024, dropout=0.0,
        max_seq_len=50, embedding_dim=256,
    )
    return ForecastingTransformer(config)


@pytest.fixture
def sample_embeddings(embedding_dim):
    """50 random embeddings for map building."""
    return np.random.randn(50, embedding_dim).astype(np.float32)


@pytest.fixture
def sample_gps():
    """50 GPS coords along a route in Brisbane."""
    rng = np.random.default_rng(42)
    lat = -27.5 + np.cumsum(rng.normal(0, 0.0001, 50))
    lon = 152.9 + np.cumsum(rng.normal(0, 0.0001, 50))
    return np.column_stack([lat, lon])


@pytest.fixture
def populated_map(embedding_dim, sample_embeddings, sample_gps):
    """MapDatabase with 50 entries."""
    db = MapDatabase(embedding_dim=embedding_dim, use_gpu=False)
    db.add(sample_embeddings, sample_gps, traversal="test_route")
    return db


@pytest.fixture
def sample_grids():
    """30 synthetic event grids for pipeline testing."""
    return torch.randn(30, 1, 64, 64)


# =============================================================================
# Map Database Tests
# =============================================================================


class TestMapDatabase:
    def test_empty_database(self, embedding_dim):
        db = MapDatabase(embedding_dim=embedding_dim)
        assert db.size == 0

    def test_add_embeddings(self, embedding_dim, sample_embeddings, sample_gps):
        db = MapDatabase(embedding_dim=embedding_dim)
        n_added = db.add(sample_embeddings, sample_gps, traversal="test")
        assert n_added == 50
        assert db.size == 50

    def test_add_torch_tensors(self, embedding_dim):
        db = MapDatabase(embedding_dim=embedding_dim)
        emb = torch.randn(10, embedding_dim)
        gps = torch.randn(10, 2).double()
        n = db.add(emb, gps)
        assert n == 10

    def test_search_returns_results(self, populated_map, sample_embeddings):
        query = sample_embeddings[:3]
        results = populated_map.search(query, top_k=5)
        assert len(results) == 3
        for r in results:
            assert len(r) <= 5
            assert len(r) > 0

    def test_search_top1_is_self(self, populated_map, sample_embeddings):
        """Searching for existing embeddings should ideally return themselves."""
        query = sample_embeddings[:5]
        results = populated_map.search(query, top_k=1)
        for i, r in enumerate(results):
            assert len(r) == 1
            # Top-1 should have high similarity
            assert r[0]["similarity"] > 0.9

    def test_search_result_fields(self, populated_map, sample_embeddings):
        results = populated_map.search(sample_embeddings[:1], top_k=1)
        r = results[0][0]
        assert "rank" in r
        assert "similarity" in r
        assert "gps" in r
        assert "traversal" in r
        assert r["traversal"] == "test_route"

    def test_get_top1_gps(self, populated_map, sample_embeddings):
        gps = populated_map.get_top1_gps(sample_embeddings[:5])
        assert gps.shape == (5, 2)

    def test_search_empty_database(self, embedding_dim):
        db = MapDatabase(embedding_dim=embedding_dim)
        query = np.random.randn(3, embedding_dim).astype(np.float32)
        results = db.search(query, top_k=5)
        assert len(results) == 3
        for r in results:
            assert len(r) == 0

    def test_save_and_load(self, populated_map, tmp_path, sample_embeddings):
        save_path = tmp_path / "test_map"
        populated_map.save(save_path)

        assert (save_path.with_suffix(".faiss")).exists()
        assert (save_path.with_suffix(".meta")).exists()

        loaded = MapDatabase.load(save_path, use_gpu=False)
        assert loaded.size == populated_map.size
        assert loaded.embedding_dim == populated_map.embedding_dim

        # Verify search still works
        results = loaded.search(sample_embeddings[:3], top_k=1)
        assert len(results) == 3

    def test_remove_traversal(self, embedding_dim, sample_embeddings, sample_gps):
        db = MapDatabase(embedding_dim=embedding_dim)
        db.add(sample_embeddings[:25], sample_gps[:25], traversal="route_a")
        db.add(sample_embeddings[25:], sample_gps[25:], traversal="route_b")
        assert db.size == 50

        db.remove_traversal("route_a")
        assert db.size == 25

    def test_get_stats(self, populated_map):
        stats = populated_map.get_stats()
        assert stats["size"] == 50
        assert "test_route" in stats["traversals"]

    def test_multiple_traversals(self, embedding_dim):
        db = MapDatabase(embedding_dim=embedding_dim)
        rng = np.random.default_rng(42)

        for name in ["morning", "sunset", "night"]:
            emb = rng.standard_normal((10, embedding_dim)).astype(np.float32)
            gps = rng.standard_normal((10, 2))
            db.add(emb, gps, traversal=name)

        assert db.size == 30
        stats = db.get_stats()
        assert len(stats["traversals"]) == 3


# =============================================================================
# Inference Pipeline Tests
# =============================================================================


class TestEFTVPRPipeline:
    def test_standard_localization(self, encoder, transformer, populated_map):
        pipeline = EFTVPRPipeline(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=5,
        )
        grid = torch.randn(1, 64, 64)
        results = pipeline.localize_standard(grid, top_k=3)
        assert isinstance(results, list)
        # Map has entries, so should get results
        assert len(results) > 0

    def test_forecasting_fallback(self, encoder, transformer, populated_map):
        """Forecasting should fall back to standard when buffer is short."""
        pipeline = EFTVPRPipeline(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=5,
        )
        grid = torch.randn(1, 64, 64)
        # First call — buffer too short, should still produce results
        results = pipeline.localize_forecasting(grid, top_k=3)
        assert len(results) > 0

    def test_forecasting_with_full_context(self, encoder, transformer, populated_map):
        """After enough frames, forecasting should use the transformer."""
        pipeline = EFTVPRPipeline(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=3,
        )
        # Feed enough frames to fill the buffer
        for _ in range(5):
            grid = torch.randn(1, 64, 64)
            results = pipeline.localize_forecasting(grid, top_k=3)

        assert len(results) > 0

    def test_reset_clears_buffer(self, encoder, transformer, populated_map):
        pipeline = EFTVPRPipeline(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=3,
        )
        for _ in range(5):
            pipeline.localize_forecasting(torch.randn(1, 64, 64))

        assert len(pipeline._embedding_buffer) > 0
        pipeline.reset()
        assert len(pipeline._embedding_buffer) == 0

    def test_batch_standard(self, encoder, transformer, populated_map):
        pipeline = EFTVPRPipeline(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=5,
        )
        grids = torch.randn(4, 1, 64, 64)
        results = pipeline.localize_batch_standard(grids, top_k=3)
        assert len(results) == 4

    def test_batch_forecasting(self, encoder, transformer, populated_map):
        pipeline = EFTVPRPipeline(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=5,
        )
        sequences = torch.randn(2, 5, 1, 64, 64)
        results = pipeline.localize_batch_forecasting(sequences, top_k=3)
        assert len(results) == 2

    def test_pipeline_info(self, encoder, transformer, populated_map):
        pipeline = EFTVPRPipeline(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=10,
        )
        info = pipeline.get_pipeline_info()
        assert info["map_size"] == 50
        assert info["sequence_length"] == 10


class TestStandardVPRBaseline:
    def test_localize(self, encoder, populated_map):
        baseline = StandardVPRBaseline(encoder=encoder, map_db=populated_map)
        grids = torch.randn(4, 1, 64, 64)
        results = baseline.localize(grids, top_k=3)
        assert len(results) == 4

    def test_localize_from_embeddings(self, populated_map, sample_embeddings):
        # StandardVPRBaseline doesn't need a real encoder for pre-computed embs
        baseline = StandardVPRBaseline(
            encoder=SNNEncoder(SNNEncoderConfig()),
            map_db=populated_map,
        )
        results = baseline.localize_from_embeddings(sample_embeddings[:5], top_k=1)
        assert len(results) == 5


# =============================================================================
# Sensor Dropout Tests
# =============================================================================


class TestSensorDropoutTest:
    def test_basic_dropout(self, encoder, transformer, populated_map):
        dropout_test = SensorDropoutTest(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=5,
        )
        grids = torch.randn(20, 1, 64, 64)
        gps = np.random.randn(20, 2)

        result = dropout_test.run_dropout_test(
            grids, gps, n_drop=5, drop_start=5
        )
        assert result.n_dropped == 5
        assert result.gps_predicted.shape == (5, 2)
        assert result.localization_errors_m.shape == (5,)
        assert result.mean_error_m >= 0

    def test_dropout_result_fields(self, encoder, transformer, populated_map):
        dropout_test = SensorDropoutTest(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=5,
        )
        grids = torch.randn(20, 1, 64, 64)
        gps = np.random.randn(20, 2)

        result = dropout_test.run_dropout_test(grids, gps, n_drop=3, drop_start=5)
        assert hasattr(result, "recall_at_25m")
        assert 0.0 <= result.recall_at_25m <= 1.0

    def test_dropout_too_many_frames_raises(self, encoder, transformer, populated_map):
        dropout_test = SensorDropoutTest(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=5,
        )
        grids = torch.randn(10, 1, 64, 64)
        gps = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Cannot drop"):
            dropout_test.run_dropout_test(grids, gps, n_drop=20, drop_start=5)

    def test_dropout_insufficient_context_raises(self, encoder, transformer, populated_map):
        dropout_test = SensorDropoutTest(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=10,
        )
        grids = torch.randn(20, 1, 64, 64)
        gps = np.random.randn(20, 2)

        with pytest.raises(ValueError, match="Need at least"):
            dropout_test.run_dropout_test(grids, gps, n_drop=5, drop_start=3)

    def test_sweep(self, encoder, transformer, populated_map):
        dropout_test = SensorDropoutTest(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=3,
        )
        grids = torch.randn(30, 1, 64, 64)
        gps = np.random.randn(30, 2)

        sweep_results = dropout_test.run_sweep(
            grids, gps, drop_counts=[3, 5], n_trials=2
        )
        assert 3 in sweep_results
        assert 5 in sweep_results
        assert len(sweep_results[3]) == 2

    def test_summarize_sweep(self, encoder, transformer, populated_map):
        dropout_test = SensorDropoutTest(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, sequence_length=3,
        )
        grids = torch.randn(30, 1, 64, 64)
        gps = np.random.randn(30, 2)

        sweep_results = dropout_test.run_sweep(
            grids, gps, drop_counts=[3], n_trials=2
        )
        summary = SensorDropoutTest.summarize_sweep(sweep_results)
        assert 3 in summary
        assert "mean_error_m" in summary[3]
        assert "recall_at_25m" in summary[3]


class TestCompareStandardVsForecasting:
    def test_comparison(self, encoder, transformer, populated_map):
        grids = torch.randn(20, 1, 64, 64)
        gps = np.random.randn(20, 2)

        result = compare_standard_vs_forecasting(
            encoder=encoder, transformer=transformer,
            map_db=populated_map, grids=grids, gps_coords=gps,
            n_dropped=5, drop_start=5, sequence_length=5,
        )
        assert "forecasting_vpr" in result
        assert "standard_vpr" in result
        assert result["standard_vpr"]["recall_at_25m"] == 0.0
        assert "improvement" in result
