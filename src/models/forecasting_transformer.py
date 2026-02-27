"""
Forecasting Transformer for EFT-VPR.

Predicts the next place embedding from a temporal sequence of embeddings
produced by the SNN Encoder. Uses causal (autoregressive) masking to
ensure position t only attends to positions <= t.

Architecture:
    Learnable Positional Encoding → TransformerEncoder (4 layers, 8 heads)
    → Prediction Head (Linear) → ẑ_{t+1}

The prediction is taken from the last position's hidden state (the most
informed position under causal masking).
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for the Forecasting Transformer.

    Attributes:
        d_model: Model dimension (must match encoder embedding_dim).
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoder layers.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length for positional encoding.
        embedding_dim: Output prediction dimension (usually == d_model).
    """
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 50
    embedding_dim: int = 256  # Output dim, matches encoder

    @classmethod
    def from_dict(cls, config: dict) -> "TransformerConfig":
        """Create config from YAML dictionary."""
        t = config.get("transformer", config)
        return cls(
            d_model=t.get("d_model", 256),
            nhead=t.get("nhead", 8),
            num_layers=t.get("num_layers", 4),
            dim_feedforward=t.get("dim_feedforward", 1024),
            dropout=t.get("dropout", 0.1),
            max_seq_len=t.get("max_seq_len", 50),
            embedding_dim=config.get("encoder", {}).get("embedding_dim", 256),
        )


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence positions.

    Each position gets its own learned embedding vector, unlike sinusoidal
    encodings which are fixed. More flexible for shorter sequences typical
    in VPR trajectories.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout rate applied after adding position encodings.
    """

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # Initialize with small values for stable training
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Positionally-encoded tensor, same shape.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
        pos_emb = self.position_embeddings(positions)         # (seq_len, d_model)
        return self.dropout(x + pos_emb.unsqueeze(0))


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate a causal (look-ahead) mask for autoregressive attention.

    Position i can attend to positions j where j <= i.
    Uses float('-inf') for masked positions (PyTorch convention).

    Args:
        seq_len: Sequence length.
        device: Target device.

    Returns:
        Mask of shape (seq_len, seq_len) with 0.0 for allowed positions
        and float('-inf') for blocked positions.
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1,
    )
    return mask


class ForecastingTransformer(nn.Module):
    """Predicts the next place embedding from a temporal sequence.

    Takes a sequence of D-dimensional embeddings from the SNN Encoder and
    predicts ẑ_{t+1} using multi-head self-attention with causal masking.

    The prediction is extracted from the last position's hidden state,
    which under causal masking has attended to all previous positions.

    Args:
        config: TransformerConfig with architecture parameters.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()

        if config is None:
            config = TransformerConfig()
        self.config = config

        # Input projection (in case d_model != embedding_dim)
        self.input_proj = nn.Identity()
        if config.embedding_dim != config.d_model:
            self.input_proj = nn.Linear(config.embedding_dim, config.d_model)

        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,               # (batch, seq, d_model) convention
            norm_first=True,                 # Pre-norm for stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,      # Required for causal mask
        )

        # Layer norm before prediction head
        self.norm = nn.LayerNorm(config.d_model)

        # Prediction head: maps last hidden state → predicted embedding
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.embedding_dim),
        )

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ForecastingTransformer: {config.num_layers} layers, "
            f"{config.nhead} heads, d_model={config.d_model}, "
            f"FFN={config.dim_feedforward}, params={n_params:,}"
        )

    def _init_weights(self):
        """Initialize weights with Xavier uniform for stability."""
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        emb_seq: torch.Tensor,
        return_all_positions: bool = False,
    ) -> torch.Tensor:
        """Predict the next embedding from an input sequence.

        Args:
            emb_seq: Sequence of embeddings, shape (batch, seq_len, D).
            return_all_positions: If True, return predictions for all
                positions (useful for teacher-forced training). Otherwise,
                return only the prediction from the last position.

        Returns:
            If return_all_positions:
                Predictions shape (batch, seq_len, D).
            Else:
                Prediction shape (batch, D) — the predicted ẑ_{t+1}.
        """
        batch_size, seq_len, _ = emb_seq.shape
        device = emb_seq.device

        # Project input if needed
        x = self.input_proj(emb_seq)  # (batch, seq_len, d_model)

        # Add learnable positional encodings
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)

        # Generate causal mask
        causal_mask = generate_causal_mask(seq_len, device)

        # Transformer encoder with causal masking
        x = self.transformer_encoder(x, mask=causal_mask)  # (batch, seq_len, d_model)

        # Layer norm
        x = self.norm(x)

        if return_all_positions:
            # Predict at every position (for teacher-forced training)
            predictions = self.prediction_head(x)  # (batch, seq_len, D)
            return predictions
        else:
            # Take the last position's hidden state
            last_hidden = x[:, -1, :]  # (batch, d_model)
            prediction = self.prediction_head(last_hidden)  # (batch, D)
            return prediction

    def predict_autoregressive(
        self,
        initial_seq: torch.Tensor,
        n_steps: int,
        encoder: Optional[nn.Module] = None,
    ) -> list[torch.Tensor]:
        """Autoregressively predict future embeddings.

        Used during inference and robustness testing (sensor dropout).
        Feeds each predicted embedding back as input for the next step.

        Args:
            initial_seq: Starting sequence, shape (batch, seq_len, D).
            n_steps: Number of future steps to predict.
            encoder: Optional encoder (unused here, but allows interface
                     consistency for end-to-end autoregressive prediction).

        Returns:
            List of n_steps predicted embeddings, each shape (batch, D).
        """
        self.eval()
        predictions = []

        current_seq = initial_seq.clone()

        with torch.no_grad():
            for step in range(n_steps):
                # Predict next embedding from current sequence
                next_emb = self.forward(current_seq)  # (batch, D)
                predictions.append(next_emb)

                # Slide window: drop oldest, append prediction
                next_emb_expanded = next_emb.unsqueeze(1)  # (batch, 1, D)
                current_seq = torch.cat(
                    [current_seq[:, 1:, :], next_emb_expanded], dim=1
                )

        return predictions

    def get_num_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = TransformerConfig(
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=50,
        embedding_dim=256,
    )
    model = ForecastingTransformer(config)
    print(f"\nModel parameters: {model.get_num_parameters()}")

    # Test forward pass
    batch_size, seq_len = 4, 10
    emb_seq = torch.randn(batch_size, seq_len, 256)

    print(f"\nInput shape: {emb_seq.shape}")
    prediction = model(emb_seq)
    print(f"Prediction shape (last pos): {prediction.shape}")

    all_predictions = model(emb_seq, return_all_positions=True)
    print(f"All positions shape: {all_predictions.shape}")

    # Test autoregressive prediction
    predictions = model.predict_autoregressive(emb_seq, n_steps=5)
    print(f"\nAutoregressive: {len(predictions)} steps, each {predictions[0].shape}")

    # Verify causal mask
    mask = generate_causal_mask(5, torch.device("cpu"))
    print(f"\nCausal mask (5x5):\n{mask}")

    # Test gradient flow
    loss = prediction.sum()
    loss.backward()
    grad_norms = {name: p.grad.norm().item()
                  for name, p in model.named_parameters()
                  if p.grad is not None}
    print(f"\nGradient norms (first 5):")
    for name, norm in list(grad_norms.items())[:5]:
        print(f"  {name}: {norm:.6f}")

    print("\n✓ Forecasting Transformer self-test passed.")
