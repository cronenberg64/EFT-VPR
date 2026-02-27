"""
Temporal Contrastive Loss (TCL) for the Forecasting Transformer.

InfoNCE-based loss that trains the transformer to predict the next
place embedding. Uses cosine similarity with temperature scaling.

For each predicted embedding ẑ_{t+1}:
  - Positive: the actual encoder output z_{t+1}
  - Negatives: actual encoder outputs from OTHER sequences in the batch

Loss = -log( exp(sim(ẑ, z⁺) / τ) / [exp(sim(ẑ, z⁺) / τ) + Σ exp(sim(ẑ, zⁿ) / τ)] )

This is equivalent to a cross-entropy loss over a softmax of similarities,
where the positive pair should have the highest similarity.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalContrastiveLoss(nn.Module):
    """InfoNCE-based Temporal Contrastive Loss.

    Maximizes similarity between predicted ẑ_{t+1} and actual z_{t+1}
    while minimizing similarity with negatives from other batch items.

    The loss uses cosine similarity with a learnable or fixed temperature
    parameter τ. Gradients flow through both the prediction and the
    target (to allow end-to-end fine-tuning when enabled).

    Args:
        temperature: Temperature scaling factor τ (default 0.07).
        learnable_temperature: If True, τ is a learnable parameter.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ):
        super().__init__()

        if learnable_temperature:
            # Log-parameterization for numerical stability
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(temperature).log(),
            )

        self.learnable_temperature = learnable_temperature

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value."""
        return self.log_temperature.exp()

    def forward(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        return_accuracy: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Compute InfoNCE contrastive loss.

        Each predicted embedding is compared against ALL actual embeddings
        in the batch. The diagonal (i, i) entries are the positive pairs;
        all off-diagonal entries are negatives.

        Args:
            predicted: Predicted next embeddings ẑ_{t+1}, shape (B, D).
            actual: Actual encoder outputs z_{t+1}, shape (B, D).
            return_accuracy: If True, compute top-1 retrieval accuracy.

        Returns:
            loss: Scalar InfoNCE loss.
            stats: Dict with loss value, temperature, accuracy, similarities.
        """
        batch_size = predicted.shape[0]
        device = predicted.device

        # L2-normalize for cosine similarity
        predicted_norm = F.normalize(predicted, p=2, dim=-1)  # (B, D)
        actual_norm = F.normalize(actual, p=2, dim=-1)        # (B, D)

        # Similarity matrix: each predicted vs. all actuals
        # sim[i, j] = cosine_sim(predicted_i, actual_j)
        similarity_matrix = torch.mm(
            predicted_norm, actual_norm.t()
        )  # (B, B)

        # Scale by temperature
        logits = similarity_matrix / self.temperature  # (B, B)

        # Labels: the positive pair for predicted[i] is actual[i] (diagonal)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss (equivalent to InfoNCE)
        loss = F.cross_entropy(logits, labels)

        # Statistics
        stats = {
            "loss": loss.item(),
            "temperature": self.temperature.item(),
            "mean_positive_sim": similarity_matrix.diag().mean().item(),
            "mean_negative_sim": (
                similarity_matrix.sum() - similarity_matrix.diag().sum()
            ).item() / max(batch_size * (batch_size - 1), 1),
        }

        if return_accuracy:
            # Top-1 retrieval accuracy: does the highest similarity
            # match the correct positive?
            top1_indices = logits.argmax(dim=-1)
            accuracy = (top1_indices == labels).float().mean().item()
            stats["accuracy"] = accuracy

        return loss, stats


class TemporalContrastiveLossAllPositions(nn.Module):
    """TCL variant for teacher-forced training at all sequence positions.

    When the transformer predicts at every position (return_all_positions=True),
    this loss computes InfoNCE at each timestep and averages.

    For position t, the predicted ẑ_{t+1} is compared against the actual
    z_{t+1} from the target sequence.

    Args:
        temperature: Temperature scaling factor.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.single_step_loss = TemporalContrastiveLoss(temperature=temperature)

    def forward(
        self,
        predicted_seq: torch.Tensor,
        actual_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute TCL averaged across all sequence positions.

        Args:
            predicted_seq: Predicted embeddings, shape (B, T, D).
                predicted_seq[:, t, :] predicts actual_seq[:, t+1, :]
            actual_seq: Actual embeddings, shape (B, T+1, D).
                actual_seq[:, 1:, :] are the targets.

        Returns:
            loss: Averaged scalar loss.
            stats: Dict with per-position and aggregate statistics.
        """
        # predicted_seq[:, t] predicts actual_seq[:, t+1]
        # So we align: predictions[0..T-1] → targets[1..T]
        T = predicted_seq.shape[1]

        total_loss = 0.0
        total_accuracy = 0.0

        for t in range(T):
            pred = predicted_seq[:, t, :]         # (B, D)
            target = actual_seq[:, t + 1, :]      # (B, D)

            step_loss, step_stats = self.single_step_loss(pred, target)
            total_loss = total_loss + step_loss
            total_accuracy += step_stats.get("accuracy", 0.0)

        avg_loss = total_loss / T
        avg_accuracy = total_accuracy / T

        stats = {
            "loss": avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss,
            "accuracy": avg_accuracy,
            "num_positions": T,
            "temperature": self.single_step_loss.temperature.item(),
        }

        return avg_loss, stats


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    batch_size = 8
    embedding_dim = 256

    # Simulate predicted and actual embeddings
    predicted = torch.randn(batch_size, embedding_dim)
    actual = torch.randn(batch_size, embedding_dim)

    # Test basic TCL
    tcl = TemporalContrastiveLoss(temperature=0.07)
    loss, stats = tcl(predicted, actual)
    print(f"=== Basic TCL ===")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Stats: {stats}")

    # Test with identical predictions (should have low loss)
    loss_identical, stats_identical = tcl(actual, actual)
    print(f"\n=== Identical pred/actual ===")
    print(f"  Loss: {loss_identical.item():.4f}")
    print(f"  Accuracy: {stats_identical['accuracy']:.1%}")
    assert stats_identical["accuracy"] == 1.0, "Perfect match should give 100% accuracy"

    # Test gradient flow
    predicted.requires_grad_(True)
    loss2, _ = tcl(predicted, actual)
    loss2.backward()
    assert predicted.grad is not None, "Gradients should flow to predictions"
    print(f"\n  Gradient norm: {predicted.grad.norm().item():.4f}")

    # Test all-positions variant
    T = 5
    pred_seq = torch.randn(batch_size, T, embedding_dim)
    actual_seq = torch.randn(batch_size, T + 1, embedding_dim)
    tcl_all = TemporalContrastiveLossAllPositions(temperature=0.07)
    loss_all, stats_all = tcl_all(pred_seq, actual_seq)
    print(f"\n=== All-Positions TCL ===")
    print(f"  Loss: {loss_all.item():.4f}")
    print(f"  Stats: {stats_all}")

    print("\n✓ Temporal Contrastive Loss self-test passed.")
