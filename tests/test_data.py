"""
Unit tests for the data pipeline (Phase 1).

Tests event binning, transforms, sequence dataset, and preprocessing.
Uses synthetic data — no real dataset required.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.event_binning import (
    BinConfig,
    BinMode,
    bin_events,
    bin_events_to_tensor,
    load_parquet_events,
    _project_to_grid,
    _accumulate_events_binary,
    _accumulate_events_polarity,
)
from src.data.transforms import (
    BinarySpikeTransform,
    NormalizeTransform,
    RandomFlipTransform,
    ComposeTransforms,
    get_default_transforms,
)
from src.data.sequence_dataset import EventSequenceDataset


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_events():
    """Create a synthetic event DataFrame."""
    rng = np.random.default_rng(42)
    n = 25_000
    return pd.DataFrame({
        "x": rng.integers(0, 346, size=n),
        "y": rng.integers(0, 260, size=n),
        "timestamp": np.sort(rng.integers(0, 1_000_000, size=n)),
        "polarity": rng.choice([-1, 1], size=n),
    })


@pytest.fixture
def default_config():
    """Default binning config."""
    return BinConfig(
        grid_size=64,
        bin_mode=BinMode.FIXED_COUNT,
        bin_count=5000,
        polarity_mode="binary",
    )


@pytest.fixture
def sample_h5_file(synthetic_events, default_config, tmp_path):
    """Create a sample HDF5 file for dataset testing."""
    grids, timestamps = bin_events_to_tensor(synthetic_events, default_config)
    n_bins = grids.shape[0]

    # Create fake GPS data
    rng = np.random.default_rng(42)
    gps = np.column_stack([
        -27.5 + np.cumsum(rng.normal(0, 0.0001, n_bins)),
        152.9 + np.cumsum(rng.normal(0, 0.0001, n_bins)),
    ])

    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("bins", data=grids.numpy(), dtype="float32")
        f.create_dataset("timestamps", data=timestamps, dtype="float64")
        f.create_dataset("gps", data=gps, dtype="float64")

    return h5_path


# =============================================================================
# Event Binning Tests
# =============================================================================


class TestProjectToGrid:
    def test_basic_projection(self):
        x = np.array([0, 173, 345])
        y = np.array([0, 130, 259])
        gx, gy = _project_to_grid(x, y, 346, 260, 64)
        assert gx.shape == (3,)
        assert gy.shape == (3,)
        assert gx[0] == 0
        assert gx[-1] <= 63
        assert gy[0] == 0
        assert gy[-1] <= 63

    def test_clipping(self):
        """Edge case: coordinates at sensor boundary."""
        x = np.array([346])  # Out of range
        y = np.array([260])
        gx, gy = _project_to_grid(x, y, 346, 260, 64)
        assert gx[0] <= 63
        assert gy[0] <= 63


class TestBinEventsFixedCount:
    def test_output_shape(self, synthetic_events, default_config):
        bins = bin_events(synthetic_events, default_config)
        assert len(bins) == 5  # 25000 / 5000

        for b in bins:
            assert b["grid"].shape == (1, 64, 64)
            assert b["grid"].dtype == np.float32

    def test_binary_values(self, synthetic_events, default_config):
        bins = bin_events(synthetic_events, default_config)
        for b in bins:
            unique_vals = np.unique(b["grid"])
            assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_timestamps_ordered(self, synthetic_events, default_config):
        bins = bin_events(synthetic_events, default_config)
        for b in bins:
            assert b["timestamp_start"] <= b["timestamp_end"]
        for i in range(1, len(bins)):
            assert bins[i]["timestamp_start"] >= bins[i - 1]["timestamp_start"]


class TestBinEventsFixedDuration:
    def test_produces_bins(self, synthetic_events):
        config = BinConfig(
            grid_size=64,
            bin_mode=BinMode.FIXED_DURATION,
            bin_duration_ms=50.0,
        )
        bins = bin_events(synthetic_events, config)
        assert len(bins) > 0

    def test_empty_windows_produce_zeros(self):
        """Events with big gaps should produce zero grids for empty windows."""
        events = pd.DataFrame({
            "x": [10, 20],
            "y": [10, 20],
            "timestamp": [0, 1_000_000],  # 1 second gap
            "polarity": [1, 1],
        })
        config = BinConfig(
            grid_size=32,
            bin_mode=BinMode.FIXED_DURATION,
            bin_duration_ms=50.0,
        )
        bins = bin_events(events, config)
        # Should have many bins, most with zero events
        zero_bins = [b for b in bins if b["num_events"] == 0]
        assert len(zero_bins) > 0


class TestPolaritySumMode:
    def test_two_channels(self, synthetic_events):
        config = BinConfig(
            grid_size=64,
            bin_mode=BinMode.FIXED_COUNT,
            bin_count=5000,
            polarity_mode="polarity_sum",
        )
        bins = bin_events(synthetic_events, config)
        for b in bins:
            assert b["grid"].shape == (2, 64, 64)

    def test_channels_non_negative(self, synthetic_events):
        config = BinConfig(
            grid_size=64,
            bin_mode=BinMode.FIXED_COUNT,
            bin_count=5000,
            polarity_mode="polarity_sum",
        )
        bins = bin_events(synthetic_events, config)
        for b in bins:
            assert b["grid"].min() >= 0


class TestBinEventsToTensor:
    def test_returns_tensor(self, synthetic_events, default_config):
        grids, ts = bin_events_to_tensor(synthetic_events, default_config)
        assert isinstance(grids, torch.Tensor)
        assert grids.shape == (5, 1, 64, 64)
        assert ts.shape == (5, 2)

    def test_empty_events(self, default_config):
        empty_df = pd.DataFrame(columns=["x", "y", "timestamp", "polarity"])
        grids, ts = bin_events_to_tensor(empty_df, default_config)
        assert grids.shape[0] == 0
        assert ts.shape[0] == 0


# =============================================================================
# Transform Tests
# =============================================================================


class TestBinarySpikeTransform:
    def test_numpy(self):
        grid = np.array([[[0, 3, 0], [1, 0, 5]]], dtype=np.float32)
        result = BinarySpikeTransform()(grid)
        expected = np.array([[[0, 1, 0], [1, 0, 1]]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_torch(self):
        grid = torch.tensor([[[0, 3, 0], [1, 0, 5]]], dtype=torch.float32)
        result = BinarySpikeTransform()(grid)
        assert result.dtype == torch.float32
        assert result.sum() == 3


class TestNormalizeTransform:
    def test_minmax_range(self):
        grid = np.random.rand(1, 32, 32).astype(np.float32) * 10
        result = NormalizeTransform("minmax")(grid)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_zscore_mean(self):
        grid = np.random.rand(1, 32, 32).astype(np.float32) * 10
        result = NormalizeTransform("zscore")(grid)
        assert abs(result.mean()) < 1e-5


class TestComposeTransforms:
    def test_compose(self):
        t = ComposeTransforms([BinarySpikeTransform(), NormalizeTransform("minmax")])
        grid = np.random.rand(1, 32, 32).astype(np.float32) * 10
        result = t(grid)
        assert result.min() >= 0.0


# =============================================================================
# Sequence Dataset Tests
# =============================================================================


class TestEventSequenceDataset:
    def test_length(self, sample_h5_file):
        ds = EventSequenceDataset([sample_h5_file], sequence_length=3, stride=1)
        # 5 bins, seq_len=3, need 3+1=4 bins per sample => 2 valid windows
        assert len(ds) == 2

    def test_getitem_shapes(self, sample_h5_file):
        ds = EventSequenceDataset([sample_h5_file], sequence_length=3, stride=1)
        sample = ds[0]

        assert sample["input"].shape == (3, 1, 64, 64)
        assert sample["target"].shape == (1, 64, 64)
        assert sample["target_gps"].shape == (2,)
        assert sample["input_gps"].shape == (3, 2)
        assert sample["timestamps"].shape == (4, 2)

    def test_getitem_dtypes(self, sample_h5_file):
        ds = EventSequenceDataset([sample_h5_file], sequence_length=3, stride=1)
        sample = ds[0]

        assert sample["input"].dtype == torch.float32
        assert sample["target"].dtype == torch.float32
        assert sample["target_gps"].dtype == torch.float64

    def test_with_transforms(self, sample_h5_file):
        transform = get_default_transforms(normalize="minmax", augment=False)
        ds = EventSequenceDataset(
            [sample_h5_file], sequence_length=3, stride=1, transform=transform
        )
        sample = ds[0]
        assert sample["input"].min() >= 0.0
        assert sample["input"].max() <= 1.0 + 1e-6

    def test_stride(self, sample_h5_file):
        ds1 = EventSequenceDataset([sample_h5_file], sequence_length=2, stride=1)
        ds2 = EventSequenceDataset([sample_h5_file], sequence_length=2, stride=2)
        assert len(ds1) >= len(ds2)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EventSequenceDataset([tmp_path / "nonexistent.h5"])


# =============================================================================
# Parquet Loading Tests
# =============================================================================


class TestLoadParquet:
    def test_load_valid_parquet(self, tmp_path):
        events = pd.DataFrame({
            "x": [10, 20, 30],
            "y": [10, 20, 30],
            "timestamp": [100, 200, 300],
            "polarity": [1, -1, 1],
        })
        path = tmp_path / "test.parquet"
        events.to_parquet(path, index=False)

        loaded = load_parquet_events(path)
        assert len(loaded) == 3
        assert loaded["timestamp"].is_monotonic_increasing

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_parquet_events(tmp_path / "missing.parquet")

    def test_missing_columns_raises(self, tmp_path):
        bad_df = pd.DataFrame({"a": [1], "b": [2]})
        path = tmp_path / "bad.parquet"
        bad_df.to_parquet(path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_parquet_events(path)
