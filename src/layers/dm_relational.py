"""
DMRelationalLayer: Relation-specific Dvoretzky-Milman projection layer.

For each relation type r, we learn a projection matrix P_r ∈ R^{k × d} that
maps high-dimensional tangent-space vectors into a low-dimensional, relation-specific
Euclidean slice. This is the core innovation: different relations get different
"views" of the hyperbolic space, optimized for their specific geometry.
"""

import torch
import torch.nn as nn


class DMRelationalLayer(nn.Module):
    """
    Dvoretzky-Milman Relational Projection Layer.

    Maps d-dimensional tangent-space entity vectors into k-dimensional
    relation-specific Euclidean slices using learned or fixed projection matrices.

    Args:
        num_relations: Number of distinct relation types in the KG.
        high_dim: Dimension of the hyperbolic / tangent space (d).
        low_dim: Dimension of the projected Euclidean slice (k, k << d).
        learnable: If True, projection matrices are learned. If False, they are
                   fixed random Gaussian projections (JL-lemma style).
    """

    def __init__(
        self,
        num_relations: int,
        high_dim: int = 256,
        low_dim: int = 32,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.learnable = learnable

        # Relation-specific projection matrices: P_r ∈ R^{k × d}
        # Initialized as random Gaussian with variance 1/k (for JL preservation)
        init_std = 1.0 / (low_dim ** 0.5)
        projections = torch.randn(num_relations, low_dim, high_dim) * init_std
        if learnable:
            self.relation_projections = nn.Parameter(projections)
        else:
            self.register_buffer("relation_projections", projections)

        # Relation embeddings in the projected space (for TransE-style scoring)
        self.relation_embeddings = nn.Parameter(
            torch.randn(num_relations, low_dim) * 0.01
        )

        # Optional: learnable bias in the projected space
        self.bias = nn.Parameter(torch.zeros(num_relations, low_dim))

    def project_to_slice(
        self, tangent_nodes: torch.Tensor, relation_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Project tangent-space vectors into relation-specific Euclidean slices.

        Args:
            tangent_nodes: Tensor of shape (batch_size, high_dim), entity vectors
                           already mapped to tangent space via log_map_zero.
            relation_ids:  Tensor of shape (batch_size,), integer relation indices.

        Returns:
            Tensor of shape (batch_size, low_dim), projected vectors in the
            relation-specific Euclidean slice.
        """
        # Fetch the projection matrices for each relation in the batch
        # P_r: (batch_size, low_dim, high_dim)
        P_r = self.relation_projections[relation_ids]

        # Batched matrix multiply: (batch, low, high) × (batch, high, 1) -> (batch, low, 1)
        x_proj = torch.bmm(P_r, tangent_nodes.unsqueeze(-1)).squeeze(-1)

        return x_proj

    def get_relation_embedding(
        self, relation_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the relation embedding in the projected space.

        Args:
            relation_ids: Tensor of shape (batch_size,), integer relation indices.

        Returns:
            Tensor of shape (batch_size, low_dim).
        """
        return self.relation_embeddings[relation_ids]

    def get_projection_matrix(
        self, relation_id: int
    ) -> torch.Tensor:
        """
        Get the projection matrix for a single relation (useful for analysis).

        Args:
            relation_id: Integer relation index.

        Returns:
            Tensor of shape (low_dim, high_dim).
        """
        return self.relation_projections[relation_id]
