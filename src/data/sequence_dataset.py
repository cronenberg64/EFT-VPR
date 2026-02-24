"""
Sliding Window Sequence Dataset for EFT-VPR.

Loads preprocessed event bins from HDF5 files and serves temporal sequences
for training the SNN Encoder and Forecasting Transformer.

Memory safety: Uses h5py dataset slicing — never loads the full file into RAM.
GPU optimization: pin_memory=True compatible for RTX 4070 throughput.
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import ComposeTransforms, get_default_transforms

logger = logging.getLogger(__name__)


class EventSequenceDataset(Dataset):
    """Sliding window dataset over preprocessed event bin HDF5 files.

    Each HDF5 file is expected to contain:
        - 'bins': dataset of shape (N, C, H, W) — binned event grids
        - 'timestamps': dataset of shape (N, 2) — [start_us, end_us] per bin
        - 'gps': dataset of shape (N, 2) — [latitude, longitude] per bin

    The dataset serves windows of consecutive bins:
        Input:  X = bins[i : i + seq_len]       shape (seq_len, C, H, W)
        Target: Y = bins[i + seq_len]            shape (C, H, W)
        GPS:    gps[i + seq_len]                 shape (2,)

    Args:
        h5_paths: List of paths to preprocessed HDF5 files.
        sequence_length: Number of past bins in each input sequence.
        stride: Step size between consecutive windows (default=1).
        transform: Optional transform to apply to each bin grid.
    """

    def __init__(
        self,
        h5_paths: list[Path | str],
        sequence_length: int = 10,
        stride: int = 1,
        transform: Optional[ComposeTransforms] = None,
    ):
        self.h5_paths = [Path(p) for p in h5_paths]
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform

        # Build an index mapping: (file_idx, start_bin_idx) for each sample
        self._index_map: list[tuple[int, int]] = []
        self._file_lengths: list[int] = []

        for file_idx, fpath in enumerate(self.h5_paths):
            if not fpath.exists():
                raise FileNotFoundError(f"HDF5 file not found: {fpath}")
            with h5py.File(fpath, "r") as f:
                n_bins = f["bins"].shape[0]
                self._file_lengths.append(n_bins)

            # Valid windows: need seq_len bins for input + 1 for target
            n_windows = max(0, (n_bins - sequence_length - 1) // stride + 1)
            for w in range(n_windows):
                start_idx = w * stride
                self._index_map.append((file_idx, start_idx))

        logger.info(
            f"EventSequenceDataset: {len(self._index_map)} windows from "
            f"{len(self.h5_paths)} files (seq_len={sequence_length}, stride={stride})"
        )

        # Cache open file handles for performance (lazy open)
        self._file_handles: dict[int, h5py.File] = {}

    def _get_file(self, file_idx: int) -> h5py.File:
        """Get (or lazily open) an HDF5 file handle."""
        if file_idx not in self._file_handles:
            fpath = self.h5_paths[file_idx]
            self._file_handles[file_idx] = h5py.File(fpath, "r")
        return self._file_handles[file_idx]

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sequence sample.

        Returns:
            Dict with keys:
                'input': torch.Tensor of shape (seq_len, C, H, W)
                'target': torch.Tensor of shape (C, H, W)
                'target_gps': torch.Tensor of shape (2,)
                'input_gps': torch.Tensor of shape (seq_len, 2)
                'timestamps': torch.Tensor of shape (seq_len + 1, 2)
        """
        file_idx, start_bin = self._index_map[idx]
        f = self._get_file(file_idx)

        end_input = start_bin + self.sequence_length
        target_idx = end_input  # The very next bin after the input sequence

        # Slice directly from HDF5 — only reads requested bins from disk
        input_bins = f["bins"][start_bin:end_input]      # (seq_len, C, H, W)
        target_bin = f["bins"][target_idx]                # (C, H, W)

        # GPS coordinates
        input_gps = f["gps"][start_bin:end_input]         # (seq_len, 2)
        target_gps = f["gps"][target_idx]                 # (2,)

        # Timestamps
        timestamps = f["timestamps"][start_bin:target_idx + 1]  # (seq_len+1, 2)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_bins.astype(np.float32))
        target_tensor = torch.from_numpy(target_bin.astype(np.float32))

        # Apply transforms to each bin individually
        if self.transform is not None:
            transformed = []
            for t_step in range(input_tensor.shape[0]):
                transformed.append(self.transform(input_tensor[t_step]))
            input_tensor = torch.stack(transformed, dim=0)
            target_tensor = self.transform(target_tensor)

        return {
            "input": input_tensor,                                        # (seq_len, C, H, W)
            "target": target_tensor,                                      # (C, H, W)
            "target_gps": torch.from_numpy(target_gps.astype(np.float64)),  # (2,)
            "input_gps": torch.from_numpy(input_gps.astype(np.float64)),    # (seq_len, 2)
            "timestamps": torch.from_numpy(timestamps.astype(np.float64)), # (seq_len+1, 2)
        }

    def close(self):
        """Close all open HDF5 file handles."""
        for f in self._file_handles.values():
            f.close()
        self._file_handles.clear()

    def __del__(self):
        self.close()


def create_dataloader(
    h5_paths: list[Path | str],
    sequence_length: int = 10,
    stride: int = 1,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = False,
    normalize: str = "minmax",
) -> DataLoader:
    """Create a DataLoader for event sequence training.

    Wraps EventSequenceDataset with optimal settings for RTX 4070.

    Args:
        h5_paths: Paths to preprocessed HDF5 files.
        sequence_length: Number of past bins per sample.
        stride: Window stride.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        augment: Whether to apply data augmentation.
        normalize: Normalization mode.

    Returns:
        Configured DataLoader.
    """
    transform = get_default_transforms(normalize=normalize, augment=augment)

    dataset = EventSequenceDataset(
        h5_paths=h5_paths,
        sequence_length=sequence_length,
        stride=stride,
        transform=transform,
    )

    # Note: num_workers > 0 requires h5py files to be opened per-worker.
    # With num_workers=0, we reuse cached handles (faster for small datasets).
    effective_workers = num_workers if len(h5_paths) > 1 else 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Ensure consistent batch sizes for training
        persistent_workers=effective_workers > 0,
    )

    logger.info(
        f"DataLoader: {len(dataset)} samples, batch_size={batch_size}, "
        f"workers={effective_workers}, pin_memory={pin_memory}"
    )
    return loader
