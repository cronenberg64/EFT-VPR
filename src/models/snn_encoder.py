"""
Spiking Neural Network Encoder for EFT-VPR.

3-layer convolutional SNN using snnTorch that compresses event grids
into continuous place embeddings via membrane potential readout.

Architecture per layer:
    Conv2d → BatchNorm → LIF (Leaky) → MaxPool2d

Output: Continuous 256-dim embedding from the membrane potential of
the final fully connected layer (not spikes).

Hardware target: RTX 4070 (Compute Capability 8.9).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SNNEncoderConfig:
    """Configuration for the SNN Encoder.

    Attributes:
        in_channels: Input channels (1 for binary, 2 for polarity_sum).
        channels: Conv2d output channels per layer.
        kernel_size: Convolution kernel size.
        embedding_dim: Output embedding dimension D.
        beta_init: Initial LIF decay rate (learnable).
        spike_grad_slope: Slope for fast_sigmoid surrogate gradient.
        grid_size: Spatial resolution of input grids.
    """
    in_channels: int = 1
    channels: list[int] = None  # Default set in __post_init__
    kernel_size: int = 3
    embedding_dim: int = 256
    beta_init: float = 0.9
    spike_grad_slope: int = 25
    grid_size: int = 64

    def __post_init__(self):
        if self.channels is None:
            self.channels = [32, 64, 128]

    @classmethod
    def from_dict(cls, config: dict) -> "SNNEncoderConfig":
        """Create config from YAML dictionary."""
        enc = config.get("encoder", config)
        data = config.get("data", {})
        return cls(
            in_channels=enc.get("in_channels", 1),
            channels=enc.get("channels", [32, 64, 128]),
            kernel_size=enc.get("kernel_size", 3),
            embedding_dim=enc.get("embedding_dim", 256),
            beta_init=enc.get("beta", 0.9),
            spike_grad_slope=enc.get("slope", 25),
            grid_size=data.get("grid_size", 64),
        )


class SpikingConvBlock(nn.Module):
    """Single spiking convolutional block: Conv2d → BN → LIF → MaxPool.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        kernel_size: Conv kernel size.
        beta_init: Initial LIF decay rate.
        spike_grad: Surrogate gradient function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        beta_init: float = 0.9,
        spike_grad=None,
    ):
        super().__init__()
        padding = kernel_size // 2  # Same padding

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = snn.Leaky(
            beta=beta_init,
            learn_beta=True,
            spike_grad=spike_grad,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single timestep.

        Args:
            x: Input tensor of shape (batch, C_in, H, W).
            mem: Previous membrane potential, shape (batch, C_out, H/2, W/2).

        Returns:
            spk: Output spikes, shape (batch, C_out, H/2, W/2).
            mem: Updated membrane potential, same shape.
        """
        cur = self.pool(self.bn(self.conv(x)))
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SNNEncoder(nn.Module):
    """3-layer Spiking Neural Network encoder for place embedding extraction.

    Processes a sequence of event grids through 3 spiking convolutional blocks,
    then a fully connected layer. Returns the membrane potential of the FC layer
    as a continuous embedding vector.

    The encoder processes the full temporal sequence, accumulating spike
    statistics over timesteps. The final embedding comes from the membrane
    potential at the last timestep.

    Args:
        config: SNNEncoderConfig with architecture parameters.
    """

    def __init__(self, config: Optional[SNNEncoderConfig] = None):
        super().__init__()

        if config is None:
            config = SNNEncoderConfig()
        self.config = config

        # Surrogate gradient for backprop through spikes
        spike_grad = surrogate.fast_sigmoid(slope=config.spike_grad_slope)

        # Build 3 spiking conv blocks
        channels = [config.in_channels] + config.channels
        self.blocks = nn.ModuleList()
        for i in range(len(config.channels)):
            self.blocks.append(SpikingConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=config.kernel_size,
                beta_init=config.beta_init,
                spike_grad=spike_grad,
            ))

        # Compute spatial size after 3 MaxPool2d(2) operations
        # 64 → 32 → 16 → 8
        spatial_size = config.grid_size // (2 ** len(config.channels))
        fc_input_dim = config.channels[-1] * spatial_size * spatial_size

        # Fully connected embedding layer with LIF readout
        self.fc = nn.Linear(fc_input_dim, config.embedding_dim)
        self.lif_out = snn.Leaky(
            beta=config.beta_init,
            learn_beta=True,
            spike_grad=spike_grad,
            output=True,  # This is the readout layer
        )

        self._fc_input_dim = fc_input_dim
        self._spatial_size = spatial_size

        logger.info(
            f"SNNEncoder: {len(self.blocks)} blocks, "
            f"spatial {config.grid_size}→{spatial_size}, "
            f"FC {fc_input_dim}→{config.embedding_dim}, "
            f"beta_init={config.beta_init}"
        )

    def _init_membrane_states(
        self, batch_size: int, device: torch.device
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Initialize membrane potentials to zero for all layers.

        Args:
            batch_size: Current batch size.
            device: Target device (cuda/cpu).

        Returns:
            block_mems: List of membrane potentials for conv blocks.
            fc_mem: Membrane potential for the output FC layer.
        """
        block_mems = []
        spatial = self.config.grid_size
        for i, block in enumerate(self.blocks):
            spatial = spatial // 2  # After MaxPool
            mem = torch.zeros(
                batch_size, self.config.channels[i], spatial, spatial,
                device=device,
            )
            block_mems.append(mem)

        fc_mem = torch.zeros(
            batch_size, self.config.embedding_dim,
            device=device,
        )
        return block_mems, fc_mem

    def forward(
        self,
        x: torch.Tensor,
        return_spike_counts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass through the full temporal sequence.

        Processes each timestep through the spiking layers sequentially,
        accumulating membrane potentials. Returns the final FC membrane
        potential as the continuous place embedding.

        Args:
            x: Input tensor of shape (batch, timesteps, C, H, W).
            return_spike_counts: If True, also return spike statistics.

        Returns:
            embedding: Continuous embedding, shape (batch, embedding_dim).
            stats: (Optional) Dict with spike counts per layer.
        """
        batch_size, timesteps, C, H, W = x.shape
        device = x.device

        # Initialize membrane potentials
        block_mems, fc_mem = self._init_membrane_states(batch_size, device)

        # Track spike counts for energy analysis
        spike_counts = {f"block_{i}": 0 for i in range(len(self.blocks))}
        spike_counts["output"] = 0

        # Process each timestep
        for t in range(timesteps):
            x_t = x[:, t]  # (batch, C, H, W)

            # Propagate through spiking conv blocks
            for i, block in enumerate(self.blocks):
                x_t, block_mems[i] = block(x_t, block_mems[i])
                if return_spike_counts:
                    spike_counts[f"block_{i}"] += x_t.sum().item()

            # Flatten and pass through FC
            x_flat = x_t.reshape(batch_size, -1)  # (batch, fc_input_dim)
            fc_cur = self.fc(x_flat)
            spk_out, fc_mem = self.lif_out(fc_cur, fc_mem)

            if return_spike_counts:
                spike_counts["output"] += spk_out.sum().item()

        # The embedding is the final membrane potential (continuous value)
        embedding = fc_mem  # (batch, embedding_dim)

        if return_spike_counts:
            # Normalize by batch_size * timesteps for per-sample-per-step rates
            total = batch_size * timesteps
            spike_rates = {k: v / total for k, v in spike_counts.items()}
            return embedding, spike_rates

        return embedding

    def encode_single(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a single event grid (no temporal sequence).

        Convenience method for encoding individual bins, e.g., for building
        the reference map database.

        Args:
            grid: Single grid of shape (batch, C, H, W) or (C, H, W).

        Returns:
            embedding: Shape (batch, embedding_dim).
        """
        if grid.ndim == 3:
            grid = grid.unsqueeze(0)  # Add batch dim
        # Treat single grid as sequence of length 1
        grid = grid.unsqueeze(1)  # (batch, 1, C, H, W)
        return self.forward(grid)

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

    config = SNNEncoderConfig(
        in_channels=1,
        channels=[32, 64, 128],
        embedding_dim=256,
        beta_init=0.9,
        grid_size=64,
    )
    model = SNNEncoder(config)
    print(f"\nModel parameters: {model.get_num_parameters()}")

    # Test with a batch of sequences
    batch_size, seq_len = 4, 10
    x = torch.randn(batch_size, seq_len, 1, 64, 64)

    print(f"\nInput shape: {x.shape}")
    embedding, spike_rates = model(x, return_spike_counts=True)
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding stats: mean={embedding.mean():.4f}, std={embedding.std():.4f}")
    print(f"Spike rates: {spike_rates}")

    # Test single encoding
    single_grid = torch.randn(2, 1, 64, 64)
    single_emb = model.encode_single(single_grid)
    print(f"\nSingle encode: {single_grid.shape} → {single_emb.shape}")

    # Test gradient flow
    loss = embedding.sum()
    loss.backward()
    grad_norms = {name: p.grad.norm().item()
                  for name, p in model.named_parameters()
                  if p.grad is not None}
    print(f"\nGradient norms (showing first 5):")
    for name, norm in list(grad_norms.items())[:5]:
        print(f"  {name}: {norm:.6f}")

    # Check learnable beta values
    for i, block in enumerate(model.blocks):
        print(f"  Block {i} beta: {block.lif.beta.item():.4f}")
    print(f"  Output beta: {model.lif_out.beta.item():.4f}")

    print("\n✓ SNN Encoder self-test passed.")
