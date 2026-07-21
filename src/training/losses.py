"""
Loss functions for Knowledge Graph link prediction.
"""

import torch
import torch.nn as nn


def hinge_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Margin-based hinge loss for KG embeddings.

    L = max(0, margin + pos_score - neg_score)

    Args:
        pos_scores: Scores for positive triples (lower = better). Shape (B,).
        neg_scores: Scores for negative triples (lower = better). Shape (B * N,).
        margin: Hinge loss margin.

    Returns:
        Scalar loss.
    """
    # Reshape negative scores to (batch_size, num_neg)
    num_pos = pos_scores.shape[0]
    num_neg_per_pos = neg_scores.shape[0] // num_pos
    neg_scores = neg_scores.view(num_pos, num_neg_per_pos)

    # pos_scores: (B, 1), neg_scores: (B, N)
    pos_scores = pos_scores.unsqueeze(1)

    # max(0, margin + pos - neg)
    loss = torch.clamp(margin + pos_scores - neg_scores, min=0.0)

    return loss.mean()


def bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss for KG embeddings.

    Treats positive triples as label 1, negatives as label 0.
    Uses sigmoid on negative scores: -log(sigmoid(-score)).

    Args:
        pos_scores: Scores for positive triples (lower = better). Shape (B,).
        neg_scores: Scores for negative triples (lower = better). Shape (B * N,).

    Returns:
        Scalar loss.
    """
    # For BCE with distance scores: we want small distance for positives,
    # large distance for negatives.
    # Loss = -log(sigmoid(-pos_score)) - log(sigmoid(neg_score))
    pos_loss = -torch.log(torch.sigmoid(-pos_scores) + 1e-10).mean()
    neg_loss = -torch.log(torch.sigmoid(neg_scores) + 1e-10).mean()

    return pos_loss + neg_loss
