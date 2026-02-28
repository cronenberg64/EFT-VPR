"""
FAISS-Based Map Database for Visual Place Recognition.

Stores reference place embeddings with associated GPS coordinates.
Uses IndexFlatIP (inner product / cosine similarity on L2-normalized vectors)
for exact nearest-neighbor search. Supports GPU acceleration via GpuIndexFlatIP
on RTX 4070.

Map lifecycle:
    1. Build: Encode reference traversal → add embeddings to index
    2. Query: Encode query → search top-K → return GPS coordinates
    3. Save/Load: Persist index + metadata to disk
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MapEntry:
    """Metadata for a single map entry.

    Attributes:
        gps: GPS coordinates [latitude, longitude].
        timestamp: Timestamp of the event bin (midpoint).
        traversal: Source traversal name (e.g., 'sunset1').
        bin_index: Index within the traversal's HDF5 file.
    """
    gps: np.ndarray
    timestamp: float = 0.0
    traversal: str = ""
    bin_index: int = 0


class MapDatabase:
    """FAISS-backed reference map for Visual Place Recognition.

    All embeddings are L2-normalized before indexing, so inner product
    search (IndexFlatIP) is equivalent to cosine similarity search.

    Supports:
      - CPU and GPU (RTX 4070) indexing
      - Incremental add/remove
      - Save/load with metadata
      - Top-K retrieval with distance filtering

    Args:
        embedding_dim: Dimension of place embeddings (default 256).
        use_gpu: Whether to use GPU-accelerated FAISS index.
        gpu_id: GPU device ID (default 0).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        use_gpu: bool = False,
        gpu_id: int = 0,
    ):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        # Create flat inner-product index (cosine sim on normalized vectors)
        self._cpu_index = faiss.IndexFlatIP(embedding_dim)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, gpu_id, self._cpu_index)
                self._gpu_resources = res
                logger.info(f"FAISS GPU index created on device {gpu_id}")
            except Exception as e:
                logger.warning(f"GPU FAISS failed, falling back to CPU: {e}")
                self._index = self._cpu_index
                self.use_gpu = False
        else:
            self._index = self._cpu_index

        # Metadata storage (parallel to index)
        self._entries: list[MapEntry] = []

        logger.info(
            f"MapDatabase: dim={embedding_dim}, "
            f"device={'GPU' if self.use_gpu else 'CPU'}"
        )

    @property
    def size(self) -> int:
        """Number of entries in the map."""
        return self._index.ntotal

    def add(
        self,
        embeddings: np.ndarray | torch.Tensor,
        gps_coords: np.ndarray | torch.Tensor,
        timestamps: Optional[np.ndarray] = None,
        traversal: str = "",
        start_index: int = 0,
    ) -> int:
        """Add embeddings to the map database.

        Args:
            embeddings: Place embeddings, shape (N, D).
            gps_coords: GPS coordinates, shape (N, 2) as [lat, lon].
            timestamps: Optional timestamps, shape (N,).
            traversal: Source traversal name.
            start_index: Starting bin index for metadata.

        Returns:
            Number of entries added.
        """
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(gps_coords, torch.Tensor):
            gps_coords = gps_coords.detach().cpu().numpy()

        embeddings = embeddings.astype(np.float32)
        n = embeddings.shape[0]

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_normalized = embeddings / norms

        # Add to FAISS index
        self._index.add(embeddings_normalized)

        # Store metadata
        if timestamps is None:
            timestamps = np.zeros(n)

        for i in range(n):
            entry = MapEntry(
                gps=gps_coords[i].copy(),
                timestamp=float(timestamps[i]) if timestamps is not None else 0.0,
                traversal=traversal,
                bin_index=start_index + i,
            )
            self._entries.append(entry)

        logger.debug(f"Added {n} entries from '{traversal}' (total: {self.size})")
        return n

    def search(
        self,
        query_embeddings: np.ndarray | torch.Tensor,
        top_k: int = 5,
        min_similarity: float = -1.0,
    ) -> list[list[dict]]:
        """Search for nearest places in the map.

        Args:
            query_embeddings: Query embeddings, shape (N, D).
            top_k: Number of nearest neighbors to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of N lists, each containing top_k dicts with:
                'rank', 'similarity', 'gps', 'timestamp', 'traversal',
                'bin_index', 'map_index'.
        """
        if self.size == 0:
            return [[] for _ in range(query_embeddings.shape[0])]

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()

        query_embeddings = query_embeddings.astype(np.float32)

        # L2-normalize queries
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        query_normalized = query_embeddings / norms

        # FAISS search
        top_k_actual = min(top_k, self.size)
        similarities, indices = self._index.search(query_normalized, top_k_actual)

        # Build results
        results = []
        for i in range(query_normalized.shape[0]):
            query_results = []
            for rank in range(top_k_actual):
                idx = int(indices[i, rank])
                sim = float(similarities[i, rank])

                if idx < 0 or sim < min_similarity:
                    continue

                entry = self._entries[idx]
                query_results.append({
                    "rank": rank,
                    "similarity": sim,
                    "gps": entry.gps.copy(),
                    "timestamp": entry.timestamp,
                    "traversal": entry.traversal,
                    "bin_index": entry.bin_index,
                    "map_index": idx,
                })
            results.append(query_results)

        return results

    def get_top1_gps(
        self, query_embeddings: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """Quick top-1 GPS lookup for a batch of queries.

        Args:
            query_embeddings: Shape (N, D).

        Returns:
            GPS coordinates, shape (N, 2).
        """
        results = self.search(query_embeddings, top_k=1)
        gps_out = np.zeros((len(results), 2), dtype=np.float64)
        for i, r in enumerate(results):
            if r:
                gps_out[i] = r[0]["gps"]
        return gps_out

    def remove_traversal(self, traversal: str):
        """Remove all entries from a specific traversal.

        Note: FAISS IndexFlatIP doesn't support direct removal,
        so we rebuild the index without the specified traversal.

        Args:
            traversal: Traversal name to remove.
        """
        keep_indices = [
            i for i, e in enumerate(self._entries)
            if e.traversal != traversal
        ]

        if len(keep_indices) == len(self._entries):
            logger.warning(f"Traversal '{traversal}' not found in map")
            return

        # Reconstruct embeddings from index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self._index)
        else:
            cpu_index = self._index

        all_embeddings = faiss.rev_swig_ptr(
            cpu_index.get_xb(), self.size * self.embedding_dim
        ).reshape(self.size, self.embedding_dim).copy()

        # Filter
        kept_embeddings = all_embeddings[keep_indices]
        kept_entries = [self._entries[i] for i in keep_indices]

        # Reset and re-add
        self._index.reset()
        if len(kept_embeddings) > 0:
            self._index.add(kept_embeddings)
        self._entries = kept_entries

        removed = len(all_embeddings) - len(kept_embeddings)
        logger.info(f"Removed {removed} entries from '{traversal}' "
                    f"(remaining: {self.size})")

    def save(self, path: str | Path):
        """Save the map database to disk.

        Saves FAISS index and metadata separately.

        Args:
            path: Base path (without extension). Creates:
                  {path}.faiss - FAISS index
                  {path}.meta - Metadata pickle
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index (convert GPU→CPU if needed)
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self._index)
        else:
            cpu_index = self._index

        faiss.write_index(cpu_index, str(path.with_suffix(".faiss")))

        # Save metadata
        meta = {
            "embedding_dim": self.embedding_dim,
            "entries": self._entries,
            "size": self.size,
        }
        with open(path.with_suffix(".meta"), "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"Map saved: {self.size} entries → {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        use_gpu: bool = False,
        gpu_id: int = 0,
    ) -> "MapDatabase":
        """Load a map database from disk.

        Args:
            path: Base path (without extension).
            use_gpu: Whether to load onto GPU.
            gpu_id: GPU device ID.

        Returns:
            Loaded MapDatabase instance.
        """
        path = Path(path)

        # Load metadata
        with open(path.with_suffix(".meta"), "rb") as f:
            meta = pickle.load(f)

        db = cls(
            embedding_dim=meta["embedding_dim"],
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )

        # Load FAISS index
        cpu_index = faiss.read_index(str(path.with_suffix(".faiss")))

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                db._index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                db._gpu_resources = res
            except Exception:
                db._index = cpu_index
                db.use_gpu = False
        else:
            db._index = cpu_index

        db._entries = meta["entries"]

        logger.info(f"Map loaded: {db.size} entries from {path}")
        return db

    def get_stats(self) -> dict:
        """Get map statistics."""
        if self.size == 0:
            return {"size": 0}

        traversals = {}
        for e in self._entries:
            traversals[e.traversal] = traversals.get(e.traversal, 0) + 1

        gps_all = np.array([e.gps for e in self._entries])
        return {
            "size": self.size,
            "embedding_dim": self.embedding_dim,
            "device": "GPU" if self.use_gpu else "CPU",
            "traversals": traversals,
            "gps_bounds": {
                "lat_min": float(gps_all[:, 0].min()),
                "lat_max": float(gps_all[:, 0].max()),
                "lon_min": float(gps_all[:, 1].min()),
                "lon_max": float(gps_all[:, 1].max()),
            },
        }


class MapBuilder:
    """Utility to build a MapDatabase from HDF5 preprocessed files.

    Encodes all bins in each HDF5 file through the SNN encoder and
    adds them to the map with their GPS coordinates.

    Args:
        encoder: Pre-trained SNNEncoder (will be set to eval mode).
        device: Compute device.
        batch_size: Encoding batch size.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 128,
    ):
        self.encoder = encoder
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = self.encoder.to(self.device).eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def build_from_h5(
        self,
        h5_paths: list[Path | str],
        use_gpu_index: bool = False,
        embedding_dim: int = 256,
    ) -> MapDatabase:
        """Build a map database from preprocessed HDF5 files.

        Args:
            h5_paths: Paths to .h5 files.
            use_gpu_index: Whether to use GPU FAISS index.
            embedding_dim: Embedding dimension.

        Returns:
            Populated MapDatabase.
        """
        import h5py
        from tqdm import tqdm

        db = MapDatabase(
            embedding_dim=embedding_dim,
            use_gpu=use_gpu_index,
        )

        for h5_path in tqdm(h5_paths, desc="Building map"):
            h5_path = Path(h5_path)
            with h5py.File(h5_path, "r") as f:
                bins = f["bins"]           # (N, C, H, W)
                gps = f["gps"][:]          # (N, 2)
                ts = f["timestamps"][:, 0]  # Use start timestamp

                n_bins = bins.shape[0]
                all_embeddings = []

                for start in range(0, n_bins, self.batch_size):
                    end = min(start + self.batch_size, n_bins)
                    batch = torch.tensor(
                        bins[start:end], dtype=torch.float32
                    ).to(self.device)

                    emb = self.encoder.encode_single(batch)
                    all_embeddings.append(emb.cpu())

                embeddings = torch.cat(all_embeddings, dim=0).numpy()

            db.add(
                embeddings=embeddings,
                gps_coords=gps,
                timestamps=ts,
                traversal=h5_path.stem,
            )

        logger.info(f"Map built: {db.size} total entries from "
                    f"{len(h5_paths)} files")
        return db
