# EFT-VPR: Event Forecasting Transformer for Visual Place Recognition

A predictive localization system that forecasts spatial embeddings from temporal sequences of neuromorphic events, enabling "blind localization" during sensor dropout or total darkness.

## Architecture

```
Event Stream → Binning (64×64) → SNN Encoder → Temporal Transformer → FAISS Query → GPS
```

1. **Neuromorphic Data Engine** — Bins raw event streams into spatial grids
2. **SNN Encoder** — Spiking neural network extracts place embeddings (snnTorch)
3. **Forecasting Transformer** — Predicts next embedding from temporal sequence
4. **VPR Pipeline** — FAISS-indexed reference map for place retrieval

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (RTX 4070)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Dataset

Uses the [Brisbane Event VPR Dataset](https://research.qut.edu.au/qcr/datasets/brisbane-event-vpr-dataset/) (~80 GB).

```bash
# Download and preprocess
python scripts/preprocess.py --download --output data/processed --grid-size 64
```

## Training

```bash
# Phase 1: Train SNN Encoder (triplet loss)
python scripts/train.py --phase encoder --config configs/default.yaml

# Phase 2: Train Forecasting Transformer
python scripts/train.py --phase transformer --config configs/default.yaml

# Phase 3: End-to-end fine-tuning
python scripts/train.py --phase finetune --config configs/default.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml
```

## Current Hardware Specifications

- NVIDIA RTX 4070 (Compute Capability 8.9)
- Optimized h5py loading with pin_memory for GPU throughput

## License

Apache 2.0