"""
Unit tests for Phase 2: SNN Encoder and Triplet Loss.

Tests model architecture, forward/backward pass, embedding properties,
GPS triplet mining, and checkpoint save/load.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.snn_encoder import SNNEncoder, SNNEncoderConfig, SpikingConvBlock
from src.losses.triplet_loss import (
    GPSTripletLoss,
    BatchAllTripletLoss,
    haversine_distance,
    compute_pairwise_gps_distances,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def encoder_config():
    return SNNEncoderConfig(
        in_channels=1,
        channels=[32, 64, 128],
        embedding_dim=256,
        beta_init=0.9,
        grid_size=64,
    )


@pytest.fixture
def encoder(encoder_config):
    return SNNEncoder(encoder_config)


@pytest.fixture
def sample_input():
    """Batch of sequences: (batch=4, timesteps=10, C=1, H=64, W=64)."""
    return torch.randn(4, 10, 1, 64, 64)


@pytest.fixture
def sample_gps():
    """4 GPS coords in Brisbane with varying distances."""
    return torch.tensor([
        [-27.5000, 152.9000],   # Location A
        [-27.5001, 152.9001],   # ~14m from A (positive pair)
        [-27.5050, 152.9050],   # ~700m from A (negative)
        [-27.5100, 152.9100],   # ~1400m from A (negative)
    ], dtype=torch.float64)


# =============================================================================
# SNN Encoder Tests
# =============================================================================


class TestSNNEncoderArchitecture:
    def test_has_three_blocks(self, encoder):
        assert len(encoder.blocks) == 3

    def test_block_types(self, encoder):
        for block in encoder.blocks:
            assert isinstance(block, SpikingConvBlock)

    def test_output_fc_dim(self, encoder, encoder_config):
        assert encoder.fc.out_features == encoder_config.embedding_dim

    def test_parameter_count(self, encoder):
        params = encoder.get_num_parameters()
        assert params["total"] > 0
        assert params["trainable"] == params["total"]


class TestSNNEncoderForward:
    def test_output_shape(self, encoder, sample_input):
        output = encoder(sample_input)
        assert output.shape == (4, 256)

    def test_output_is_continuous(self, encoder, sample_input):
        """Embedding should be continuous (membrane potential), not binary spikes."""
        output = encoder(sample_input)
        unique_vals = torch.unique(output)
        # Continuous values should have many unique values, not just 0/1
        assert len(unique_vals) > 2

    def test_different_inputs_different_outputs(self, encoder):
        x1 = torch.randn(2, 5, 1, 64, 64)
        x2 = torch.randn(2, 5, 1, 64, 64)
        e1 = encoder(x1)
        e2 = encoder(x2)
        # Outputs should differ (extremely unlikely to be equal for random inputs)
        assert not torch.allclose(e1, e2)

    def test_spike_counts_returned(self, encoder, sample_input):
        output, stats = encoder(sample_input, return_spike_counts=True)
        assert "block_0" in stats
        assert "block_1" in stats
        assert "block_2" in stats
        assert "output" in stats
        assert output.shape == (4, 256)

    def test_deterministic_eval_mode(self, encoder, sample_input):
        encoder.eval()
        with torch.no_grad():
            e1 = encoder(sample_input)
            e2 = encoder(sample_input)
        # Should be deterministic in eval mode (no dropout)
        assert torch.allclose(e1, e2)


class TestSNNEncoderGradients:
    def test_gradients_flow(self, encoder, sample_input):
        output = encoder(sample_input)
        loss = output.sum()
        loss.backward()

        has_grad = False
        for p in encoder.parameters():
            if p.grad is not None and p.grad.norm() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through the model"

    def test_beta_has_gradient(self, encoder, sample_input):
        """Learnable beta should receive gradients."""
        output = encoder(sample_input)
        loss = output.sum()
        loss.backward()
        assert encoder.blocks[0].lif.beta.grad is not None

    def test_no_nan_gradients(self, encoder, sample_input):
        output = encoder(sample_input)
        loss = output.sum()
        loss.backward()
        for name, p in encoder.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


class TestSNNEncoderSingleEncode:
    def test_single_grid(self, encoder):
        grid = torch.randn(1, 64, 64)
        emb = encoder.encode_single(grid)
        assert emb.shape == (1, 256)

    def test_batch_grids(self, encoder):
        grids = torch.randn(8, 1, 64, 64)
        emb = encoder.encode_single(grids)
        assert emb.shape == (8, 256)


class TestSNNEncoderConfig:
    def test_from_dict(self):
        cfg_dict = {
            "encoder": {
                "in_channels": 2,
                "channels": [16, 32, 64],
                "embedding_dim": 128,
                "beta": 0.85,
            },
            "data": {"grid_size": 32},
        }
        config = SNNEncoderConfig.from_dict(cfg_dict)
        assert config.in_channels == 2
        assert config.channels == [16, 32, 64]
        assert config.embedding_dim == 128
        assert config.beta_init == 0.85
        assert config.grid_size == 32

    def test_default_channels(self):
        config = SNNEncoderConfig()
        assert config.channels == [32, 64, 128]


class TestMembraneReset:
    def test_different_sequences_independent(self, encoder):
        """Membrane states should reset between forward calls."""
        x1 = torch.randn(2, 5, 1, 64, 64)
        x2 = torch.randn(2, 5, 1, 64, 64)

        e1_first = encoder(x1)

        # Process x2

        # Process x1 again — should give same result since states reset
        e1_second = encoder(x1)

        assert torch.allclose(e1_first, e1_second, atol=1e-5)


# =============================================================================
# Triplet Loss Tests
# =============================================================================


class TestHaversineDistance:
    def test_zero_distance(self):
        gps = torch.tensor([[-27.5, 152.9]])
        dist = haversine_distance(gps, gps)
        assert abs(dist.item()) < 1.0  # Should be ~0

    def test_known_distance(self):
        """Brisbane CBD to Brookfield is roughly 10–12 km."""
        cbd = torch.tensor([[-27.4698, 153.0251]])
        brookfield = torch.tensor([[-27.4972, 152.9073]])
        dist = haversine_distance(cbd, brookfield)
        assert 10_000 < dist.item() < 15_000

    def test_symmetry(self):
        a = torch.tensor([[-27.5, 152.9]])
        b = torch.tensor([[-27.6, 153.0]])
        assert torch.allclose(haversine_distance(a, b), haversine_distance(b, a))


class TestPairwiseGPSDistances:
    def test_shape(self, sample_gps):
        dists = compute_pairwise_gps_distances(sample_gps)
        assert dists.shape == (4, 4)

    def test_diagonal_zero(self, sample_gps):
        dists = compute_pairwise_gps_distances(sample_gps)
        for i in range(4):
            assert dists[i, i].item() < 1.0

    def test_symmetric(self, sample_gps):
        dists = compute_pairwise_gps_distances(sample_gps)
        assert torch.allclose(dists, dists.T, atol=1.0)


class TestGPSTripletLoss:
    def test_loss_is_scalar(self, encoder, sample_input, sample_gps):
        embeddings = encoder(sample_input)
        loss_fn = GPSTripletLoss(margin=0.3, positive_threshold_m=25.0, negative_threshold_m=100.0)
        loss, stats = loss_fn(embeddings, sample_gps)
        assert loss.ndim == 0  # Scalar

    def test_loss_non_negative(self, encoder, sample_input, sample_gps):
        embeddings = encoder(sample_input)
        loss_fn = GPSTripletLoss()
        loss, stats = loss_fn(embeddings, sample_gps)
        assert loss.item() >= 0

    def test_stats_keys(self, encoder, sample_input, sample_gps):
        embeddings = encoder(sample_input)
        loss_fn = GPSTripletLoss()
        _, stats = loss_fn(embeddings, sample_gps)
        assert "n_triplets" in stats
        assert "n_active" in stats
        assert "active_ratio" in stats

    def test_gradient_flows_through_loss(self, encoder, sample_input, sample_gps):
        embeddings = encoder(sample_input)
        loss_fn = GPSTripletLoss()
        loss, _ = loss_fn(embeddings, sample_gps)
        if loss.item() > 0:
            loss.backward()
            has_grad = any(
                p.grad is not None and p.grad.norm() > 0
                for p in encoder.parameters()
            )
            assert has_grad


class TestBatchAllTripletLoss:
    def test_loss_non_negative(self, encoder, sample_input, sample_gps):
        embeddings = encoder(sample_input)
        loss_fn = BatchAllTripletLoss()
        loss, stats = loss_fn(embeddings, sample_gps)
        assert loss.item() >= 0

    def test_all_same_location_zero_triplets(self, encoder, sample_input):
        """If all samples are at the same GPS, no negatives should be found."""
        same_gps = torch.tensor([[-27.5, 152.9]] * 4, dtype=torch.float64)
        embeddings = encoder(sample_input)
        loss_fn = BatchAllTripletLoss(negative_threshold_m=100.0)
        loss, stats = loss_fn(embeddings, same_gps)
        # With all at same location: all distances < 100m, so no negatives
        assert stats["n_triplets"] == 0
