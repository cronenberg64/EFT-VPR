"""
Transforms for event bin grids.

Provides composable transforms for normalization and augmentation of
binned event grids. All transforms operate on numpy arrays or torch tensors
of shape (C, H, W).
"""

from typing import Optional

import numpy as np
import torch


class BinarySpikeTransform:
    """Clamp grid values to binary {0, 1}.

    Any cell with at least one event becomes 1, otherwise 0.
    Works on both numpy arrays and torch tensors.
    """

    def __call__(self, grid: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(grid, torch.Tensor):
            return (grid > 0).float()
        return (grid > 0).astype(np.float32)


class NormalizeTransform:
    """Per-channel normalization of event grids.

    Supports 'minmax' (scale to [0, 1]) and 'zscore' (zero mean, unit variance).
    Applied per-sample, not across the dataset.
    """

    def __init__(self, mode: str = "minmax", eps: float = 1e-8):
        """
        Args:
            mode: 'minmax' or 'zscore'.
            eps: Small constant to avoid division by zero.
        """
        if mode not in ("minmax", "zscore"):
            raise ValueError(f"Unknown normalization mode: {mode!r}")
        self.mode = mode
        self.eps = eps

    def __call__(self, grid: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(grid, torch.Tensor):
            return self._normalize_tensor(grid)
        return self._normalize_numpy(grid)

    def _normalize_numpy(self, grid: np.ndarray) -> np.ndarray:
        grid = grid.astype(np.float32)
        if self.mode == "minmax":
            g_min = grid.min()
            g_max = grid.max()
            denom = g_max - g_min + self.eps
            return (grid - g_min) / denom
        else:  # zscore
            mean = grid.mean()
            std = grid.std() + self.eps
            return (grid - mean) / std

    def _normalize_tensor(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.float()
        if self.mode == "minmax":
            g_min = grid.min()
            g_max = grid.max()
            denom = g_max - g_min + self.eps
            return (grid - g_min) / denom
        else:  # zscore
            mean = grid.mean()
            std = grid.std() + self.eps
            return (grid - mean) / std


class RandomFlipTransform:
    """Random horizontal and/or vertical flip for data augmentation.

    Applied independently per sample during training.
    """

    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.0):
        """
        Args:
            p_horizontal: Probability of horizontal flip.
            p_vertical: Probability of vertical flip.
        """
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, grid: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(grid, torch.Tensor):
            if torch.rand(1).item() < self.p_horizontal:
                grid = torch.flip(grid, dims=[-1])  # Flip width dimension
            if torch.rand(1).item() < self.p_vertical:
                grid = torch.flip(grid, dims=[-2])  # Flip height dimension
        else:
            if np.random.random() < self.p_horizontal:
                grid = np.flip(grid, axis=-1).copy()
            if np.random.random() < self.p_vertical:
                grid = np.flip(grid, axis=-2).copy()
        return grid


class RandomNoiseTransform:
    """Add small Gaussian noise to event grids for regularization.

    Only applied during training. Noise is clipped to ensure non-negativity.
    """

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, grid: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(grid, torch.Tensor):
            noise = torch.randn_like(grid) * self.std
            return torch.clamp(grid + noise, min=0.0)
        noise = np.random.randn(*grid.shape).astype(np.float32) * self.std
        return np.clip(grid + noise, 0.0, None)


class ComposeTransforms:
    """Compose multiple transforms sequentially (like torchvision.Compose)."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, grid: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        for t in self.transforms:
            grid = t(grid)
        return grid

    def __repr__(self) -> str:
        names = [t.__class__.__name__ for t in self.transforms]
        return f"ComposeTransforms({', '.join(names)})"


def get_default_transforms(
    normalize: str = "minmax",
    augment: bool = False,
) -> ComposeTransforms:
    """Get a default transform pipeline.

    Args:
        normalize: Normalization mode ('minmax', 'zscore', or None for no norm).
        augment: Whether to include data augmentation (for training).

    Returns:
        ComposeTransforms instance.
    """
    transforms = []

    if augment:
        transforms.append(RandomFlipTransform(p_horizontal=0.5))
        transforms.append(RandomNoiseTransform(std=0.01))

    if normalize:
        transforms.append(NormalizeTransform(mode=normalize))

    return ComposeTransforms(transforms)
