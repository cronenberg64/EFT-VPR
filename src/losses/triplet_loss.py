"""
GPS-Distance Triplet Loss for SNN Encoder Training.

Uses geographical distance between event bins to define positive/negative
pairs for metric learning. Embeddings from nearby locations (< 25m) should
be close; embeddings from distant locations (> 100m) should be far apart.

Implements semi-hard negative mining within each batch for stable training.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def haversine_distance(
    gps_a: torch.Tensor,
    gps_b: torch.Tensor,
) -> torch.Tensor:
    """Compute haversine distance in meters between two sets of GPS coordinates.

    Args:
        gps_a: Coordinates of shape (..., 2) as [latitude, longitude] in degrees.
        gps_b: Coordinates of shape (..., 2) as [latitude, longitude] in degrees.

    Returns:
        Distances in meters, same leading dimensions as input.
    """
    R = 6371000.0  # Earth radius in meters

    lat1 = torch.deg2rad(gps_a[..., 0])
    lat2 = torch.deg2rad(gps_b[..., 0])
    dlat = torch.deg2rad(gps_b[..., 0] - gps_a[..., 0])
    dlon = torch.deg2rad(gps_b[..., 1] - gps_a[..., 1])

    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c


def compute_pairwise_gps_distances(gps_coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise GPS distances for a batch.

    Args:
        gps_coords: Shape (batch_size, 2) as [lat, lon].

    Returns:
        Distance matrix of shape (batch_size, batch_size) in meters.
    """
    n = gps_coords.shape[0]
    # Expand for pairwise computation
    gps_a = gps_coords.unsqueeze(1).expand(n, n, 2)  # (N, N, 2)
    gps_b = gps_coords.unsqueeze(0).expand(n, n, 2)  # (N, N, 2)
    return haversine_distance(gps_a, gps_b)


class GPSTripletLoss(nn.Module):
    """Triplet loss with GPS-distance-based pair mining.

    For each anchor embedding, finds:
      - Positive: embedding with GPS distance < positive_threshold_m
      - Negative: embedding with GPS distance > negative_threshold_m

    Uses semi-hard negative mining: selects negatives that are farther
    than the positive but still within the margin boundary.

    Args:
        margin: Triplet margin (default 0.3).
        positive_threshold_m: Max GPS distance for positive pairs (meters).
        negative_threshold_m: Min GPS distance for negative pairs (meters).
        distance_metric: 'euclidean' or 'cosine'.
    """

    def __init__(
        self,
        margin: float = 0.3,
        positive_threshold_m: float = 25.0,
        negative_threshold_m: float = 100.0,
        distance_metric: str = "euclidean",
    ):
        super().__init__()
        self.margin = margin
        self.positive_threshold_m = positive_threshold_m
        self.negative_threshold_m = negative_threshold_m
        self.distance_metric = distance_metric

    def _pairwise_embedding_distances(
        self, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distances in embedding space.

        Args:
            embeddings: Shape (batch_size, embedding_dim).

        Returns:
            Distance matrix of shape (batch_size, batch_size).
        """
        if self.distance_metric == "euclidean":
            # Efficient pairwise euclidean distance
            diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
            return torch.norm(diff, p=2, dim=-1)
        elif self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            normed = F.normalize(embeddings, p=2, dim=-1)
            sim = torch.mm(normed, normed.t())
            return 1.0 - sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def forward(
        self,
        embeddings: torch.Tensor,
        gps_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute triplet loss with GPS-based mining.

        Args:
            embeddings: Shape (batch_size, embedding_dim).
            gps_coords: Shape (batch_size, 2) as [lat, lon].

        Returns:
            loss: Scalar triplet loss.
            stats: Dict with mining statistics.
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Compute GPS distance matrix
        gps_dists = compute_pairwise_gps_distances(gps_coords)  # (B, B)

        # Compute embedding distance matrix
        emb_dists = self._pairwise_embedding_distances(embeddings)  # (B, B)

        # Define positive and negative masks
        positive_mask = (gps_dists < self.positive_threshold_m) & \
                        (~torch.eye(batch_size, dtype=torch.bool, device=device))
        negative_mask = gps_dists > self.negative_threshold_m

        # For each anchor, find valid triplets
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_triplets = 0
        n_active = 0  # Triplets with non-zero loss

        for i in range(batch_size):
            # Find positives for this anchor
            pos_indices = torch.where(positive_mask[i])[0]
            neg_indices = torch.where(negative_mask[i])[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Get hardest positive (farthest in embedding space)
            pos_dists = emb_dists[i, pos_indices]
            hardest_pos_idx = pos_indices[pos_dists.argmax()]
            d_ap = emb_dists[i, hardest_pos_idx]

            # Semi-hard negative mining:
            # Select negatives where d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
            neg_dists = emb_dists[i, neg_indices]
            semi_hard_mask = (neg_dists > d_ap) & (neg_dists < d_ap + self.margin)

            if semi_hard_mask.any():
                # Pick the closest semi-hard negative
                semi_hard_neg_dists = neg_dists[semi_hard_mask]
                hardest_neg_idx = neg_indices[semi_hard_mask][semi_hard_neg_dists.argmin()]
            else:
                # Fallback: hardest negative (closest to anchor)
                hardest_neg_idx = neg_indices[neg_dists.argmin()]

            d_an = emb_dists[i, hardest_neg_idx]

            # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            triplet_loss = F.relu(d_ap - d_an + self.margin)
            total_loss = total_loss + triplet_loss
            n_triplets += 1
            if triplet_loss.item() > 0:
                n_active += 1

        # Average over valid triplets
        if n_triplets > 0:
            total_loss = total_loss / n_triplets

        stats = {
            "n_triplets": n_triplets,
            "n_active": n_active,
            "active_ratio": n_active / max(n_triplets, 1),
            "loss": total_loss.item(),
            "mean_pos_dist": float(emb_dists[positive_mask].mean()) if positive_mask.any() else 0.0,
            "mean_neg_dist": float(emb_dists[negative_mask].mean()) if negative_mask.any() else 0.0,
        }

        return total_loss, stats


class BatchAllTripletLoss(nn.Module):
    """Batch-all triplet loss variant.

    Considers ALL valid triplets in the batch rather than just the hardest.
    More stable gradients but slower convergence than semi-hard mining.

    Only used as a fallback/comparison — GPS semi-hard mining is preferred.
    """

    def __init__(
        self,
        margin: float = 0.3,
        positive_threshold_m: float = 25.0,
        negative_threshold_m: float = 100.0,
    ):
        super().__init__()
        self.margin = margin
        self.positive_threshold_m = positive_threshold_m
        self.negative_threshold_m = negative_threshold_m

    def forward(
        self,
        embeddings: torch.Tensor,
        gps_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Pairwise distances
        gps_dists = compute_pairwise_gps_distances(gps_coords)
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
        emb_dists = torch.norm(diff, p=2, dim=-1)

        # Masks
        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = (gps_dists < self.positive_threshold_m) & ~eye
        neg_mask = gps_dists > self.negative_threshold_m

        # All valid triplets: (i, j, k) where j is positive, k is negative
        # Shape: (B, B, B) — too expensive for large batches, so we vectorize
        d_ap = emb_dists.unsqueeze(2)  # (B, B, 1)
        d_an = emb_dists.unsqueeze(1)  # (B, 1, B)

        triplet_loss = F.relu(d_ap - d_an + self.margin)  # (B, B, B)

        # Mask valid triplets
        valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)  # (B, B, B)
        triplet_loss = triplet_loss * valid.float()

        n_valid = valid.sum().item()
        n_active = (triplet_loss > 0).sum().item()

        if n_valid > 0:
            loss = triplet_loss.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        stats = {
            "n_triplets": int(n_valid),
            "n_active": int(n_active),
            "active_ratio": n_active / max(n_valid, 1),
            "loss": loss.item(),
        }

        return loss, stats
