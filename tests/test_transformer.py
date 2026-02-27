"""
Unit tests for Phase 3: Forecasting Transformer and Temporal Contrastive Loss.

Tests architecture, causal masking, autoregressive prediction,
InfoNCE loss properties, and integration with the SNN encoder.
"""

import pytest
import torch
import torch.nn as nn

from src.models.forecasting_transformer import (
    ForecastingTransformer,
    TransformerConfig,
    LearnablePositionalEncoding,
    generate_causal_mask,
)
from src.losses.temporal_contrastive import (
    TemporalContrastiveLoss,
    TemporalContrastiveLossAllPositions,
)
from src.models.snn_encoder import SNNEncoder, SNNEncoderConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def transformer_config():
    return TransformerConfig(
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=50,
        embedding_dim=256,
    )


@pytest.fixture
def transformer(transformer_config):
    return ForecastingTransformer(transformer_config)


@pytest.fixture
def sample_embeddings():
    """Batch of embedding sequences: (batch=4, seq_len=10, D=256)."""
    return torch.randn(4, 10, 256)


@pytest.fixture
def encoder():
    config = SNNEncoderConfig(
        in_channels=1, channels=[32, 64, 128],
        embedding_dim=256, beta_init=0.9, grid_size=64,
    )
    return SNNEncoder(config)


# =============================================================================
# Causal Mask Tests
# =============================================================================


class TestCausalMask:
    def test_shape(self):
        mask = generate_causal_mask(5, torch.device("cpu"))
        assert mask.shape == (5, 5)

    def test_diagonal_allowed(self):
        mask = generate_causal_mask(5, torch.device("cpu"))
        for i in range(5):
            assert mask[i, i] == 0.0  # Position can attend to itself

    def test_future_blocked(self):
        mask = generate_causal_mask(5, torch.device("cpu"))
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[i, j] == float('-inf')  # Can't see future

    def test_past_allowed(self):
        mask = generate_causal_mask(5, torch.device("cpu"))
        for i in range(5):
            for j in range(i):
                assert mask[i, j] == 0.0  # Can see past

    def test_single_position(self):
        mask = generate_causal_mask(1, torch.device("cpu"))
        assert mask.shape == (1, 1)
        assert mask[0, 0] == 0.0


# =============================================================================
# Positional Encoding Tests
# =============================================================================


class TestLearnablePositionalEncoding:
    def test_output_shape(self):
        pe = LearnablePositionalEncoding(d_model=256, max_len=50)
        x = torch.randn(4, 10, 256)
        out = pe(x)
        assert out.shape == x.shape

    def test_positions_are_learnable(self):
        pe = LearnablePositionalEncoding(d_model=256, max_len=50)
        assert pe.position_embeddings.weight.requires_grad

    def test_different_positions_different_encodings(self):
        pe = LearnablePositionalEncoding(d_model=256, max_len=50, dropout=0.0)
        x = torch.zeros(1, 5, 256)
        out = pe(x)
        # All positions should be different (positions have different encodings)
        for i in range(4):
            assert not torch.allclose(out[0, i], out[0, i + 1])


# =============================================================================
# Forecasting Transformer Tests
# =============================================================================


class TestForecastingTransformerArchitecture:
    def test_num_layers(self, transformer):
        assert transformer.transformer_encoder.num_layers == 4

    def test_parameter_count(self, transformer):
        params = transformer.get_num_parameters()
        assert params["total"] > 0
        assert params["trainable"] == params["total"]

    def test_config_from_dict(self):
        cfg = {
            "transformer": {
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
            },
            "encoder": {"embedding_dim": 128},
        }
        config = TransformerConfig.from_dict(cfg)
        assert config.d_model == 128
        assert config.nhead == 4
        assert config.num_layers == 2


class TestForecastingTransformerForward:
    def test_output_shape_last_position(self, transformer, sample_embeddings):
        output = transformer(sample_embeddings)
        assert output.shape == (4, 256)

    def test_output_shape_all_positions(self, transformer, sample_embeddings):
        output = transformer(sample_embeddings, return_all_positions=True)
        assert output.shape == (4, 10, 256)

    def test_different_inputs_different_outputs(self, transformer):
        x1 = torch.randn(2, 5, 256)
        x2 = torch.randn(2, 5, 256)
        o1 = transformer(x1)
        o2 = transformer(x2)
        assert not torch.allclose(o1, o2)

    def test_deterministic_eval(self, transformer, sample_embeddings):
        transformer.eval()
        with torch.no_grad():
            o1 = transformer(sample_embeddings)
            o2 = transformer(sample_embeddings)
        assert torch.allclose(o1, o2)

    def test_variable_sequence_length(self, transformer):
        """Should handle different sequence lengths up to max_seq_len."""
        for seq_len in [3, 5, 10, 20]:
            x = torch.randn(2, seq_len, 256)
            out = transformer(x)
            assert out.shape == (2, 256)


class TestForecastingTransformerGradients:
    def test_gradients_flow(self, transformer, sample_embeddings):
        output = transformer(sample_embeddings)
        loss = output.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.norm() > 0
            for p in transformer.parameters()
        )
        assert has_grad

    def test_no_nan_gradients(self, transformer, sample_embeddings):
        output = transformer(sample_embeddings)
        loss = output.sum()
        loss.backward()
        for name, p in transformer.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    def test_positional_encoding_gradient(self, transformer, sample_embeddings):
        output = transformer(sample_embeddings)
        loss = output.sum()
        loss.backward()
        assert transformer.pos_encoding.position_embeddings.weight.grad is not None


class TestAutoregressive:
    def test_prediction_count(self, transformer, sample_embeddings):
        preds = transformer.predict_autoregressive(sample_embeddings, n_steps=5)
        assert len(preds) == 5

    def test_prediction_shapes(self, transformer, sample_embeddings):
        preds = transformer.predict_autoregressive(sample_embeddings, n_steps=3)
        for p in preds:
            assert p.shape == (4, 256)

    def test_predictions_differ(self, transformer, sample_embeddings):
        """Each autoregressive step should produce different predictions."""
        transformer.eval()
        preds = transformer.predict_autoregressive(sample_embeddings, n_steps=3)
        assert not torch.allclose(preds[0], preds[1])


class TestCausalMaskIntegrity:
    def test_future_information_leak(self, transformer):
        """Verify that modifying future positions doesn't change current prediction.

        If causal masking works correctly, position t's output shouldn't
        change when we modify positions > t.
        """
        transformer.eval()
        x = torch.randn(1, 10, 256)

        with torch.no_grad():
            # Get output at position 5
            out1 = transformer(x, return_all_positions=True)[:, 5, :]

            # Modify positions 6-9 (future)
            x_modified = x.clone()
            x_modified[:, 6:, :] = torch.randn(1, 4, 256)
            out2 = transformer(x_modified, return_all_positions=True)[:, 5, :]

        # Position 5's output should be identical
        assert torch.allclose(out1, out2, atol=1e-5), \
            "Future leak detected: position 5 output changed when future was modified"


# =============================================================================
# Temporal Contrastive Loss Tests
# =============================================================================


class TestTemporalContrastiveLoss:
    def test_loss_is_scalar(self):
        tcl = TemporalContrastiveLoss(temperature=0.07)
        pred = torch.randn(8, 256)
        actual = torch.randn(8, 256)
        loss, stats = tcl(pred, actual)
        assert loss.ndim == 0

    def test_loss_positive(self):
        tcl = TemporalContrastiveLoss(temperature=0.07)
        loss, _ = tcl(torch.randn(8, 256), torch.randn(8, 256))
        assert loss.item() > 0

    def test_identical_gives_low_loss(self):
        """When predictions exactly match targets, loss should be near zero."""
        tcl = TemporalContrastiveLoss(temperature=0.07)
        actual = torch.randn(8, 256)
        loss, stats = tcl(actual, actual)
        assert stats["accuracy"] == 1.0
        # Loss should be very low (but not exactly 0 due to negatives)
        assert loss.item() < 1.0

    def test_accuracy_metric(self):
        tcl = TemporalContrastiveLoss(temperature=0.07)
        actual = torch.randn(8, 256)
        _, stats = tcl(actual, actual)
        assert "accuracy" in stats
        assert 0.0 <= stats["accuracy"] <= 1.0

    def test_temperature_affects_loss(self):
        pred = torch.randn(8, 256)
        actual = torch.randn(8, 256)
        loss_cold, _ = TemporalContrastiveLoss(temperature=0.01)(pred, actual)
        loss_warm, _ = TemporalContrastiveLoss(temperature=1.0)(pred, actual)
        # Colder temperature should give higher loss (sharper distribution)
        assert loss_cold.item() != loss_warm.item()

    def test_gradient_flows(self):
        tcl = TemporalContrastiveLoss()
        pred = torch.randn(8, 256, requires_grad=True)
        actual = torch.randn(8, 256)
        loss, _ = tcl(pred, actual)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.norm() > 0

    def test_batch_size_one(self):
        """Edge case: batch of 1 (no negatives, only positive)."""
        tcl = TemporalContrastiveLoss()
        loss, stats = tcl(torch.randn(1, 256), torch.randn(1, 256))
        assert loss.ndim == 0  # Should still work

    def test_stats_similarity_values(self):
        tcl = TemporalContrastiveLoss()
        actual = torch.randn(8, 256)
        _, stats = tcl(actual, actual)
        assert stats["mean_positive_sim"] > stats["mean_negative_sim"]


class TestTemporalContrastiveLossAllPositions:
    def test_loss_shape(self):
        tcl = TemporalContrastiveLossAllPositions()
        pred_seq = torch.randn(4, 5, 256)
        actual_seq = torch.randn(4, 6, 256)  # T+1
        loss, stats = tcl(pred_seq, actual_seq)
        assert loss.ndim == 0

    def test_num_positions(self):
        tcl = TemporalContrastiveLossAllPositions()
        pred_seq = torch.randn(4, 5, 256)
        actual_seq = torch.randn(4, 6, 256)
        _, stats = tcl(pred_seq, actual_seq)
        assert stats["num_positions"] == 5


# =============================================================================
# Integration: Encoder + Transformer
# =============================================================================


class TestEncoderTransformerIntegration:
    def test_end_to_end_shapes(self, encoder, transformer):
        """Full pipeline: event grids → encoder → transformer → prediction."""
        # Simulate event bins
        bins = torch.randn(2, 11, 1, 64, 64)  # 10 input + 1 target

        # Encode all bins
        B, T, C, H, W = bins.shape
        all_bins = bins.reshape(B * T, C, H, W)
        with torch.no_grad():
            all_emb = encoder.encode_single(all_bins)  # (B*T, D)

        all_emb = all_emb.reshape(B, T, -1)
        input_emb = all_emb[:, :-1, :]   # (2, 10, 256)
        target_emb = all_emb[:, -1, :]    # (2, 256)

        # Predict
        predicted = transformer(input_emb)  # (2, 256)
        assert predicted.shape == (2, 256)

        # Compute loss
        tcl = TemporalContrastiveLoss()
        loss, stats = tcl(predicted, target_emb)
        assert loss.ndim == 0
        assert loss.item() > 0
