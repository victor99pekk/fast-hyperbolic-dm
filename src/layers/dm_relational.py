"""
DMRelationalLayer: Relation-specific Dvoretzky-Milman projection layer.

For each relation type r, we learn a projection matrix P_r ∈ R^{k × d} that
maps high-dimensional tangent-space vectors into a low-dimensional, relation-specific
Euclidean slice. This is the core innovation: different relations get different
"views" of the hyperbolic space, optimized for their specific geometry.
"""

import torch
import torch.nn as nn


def _random_orthogonal(shape: tuple, device=None) -> torch.Tensor:
    """Generate random semi-orthogonal matrices via QR decomposition.

    For each matrix of shape (k, d) with k < d, generates a random matrix
    of shape (d, k), computes QR, and takes Q[:, :k].T to get (k, d).
    """
    R, k, d = shape  # (num_relations, low_dim, high_dim)
    matrices = []
    for _ in range(R):
        # QR of tall (d, k) matrix → Q is (d, d), take first k columns
        A = torch.randn(d, k, device=device)
        q, _ = torch.linalg.qr(A)  # q: (d, d)
        matrices.append(q[:, :k].T)  # (k, d)
    return torch.stack(matrices)  # (R, k, d)


class DMRelationalLayer(nn.Module):
    """
    Dvoretzky-Milman Relational Projection Layer.

    Maps d-dimensional tangent-space entity vectors into k-dimensional
    relation-specific Euclidean slices using learned or fixed projection matrices.

    P_r matrices are initialized as random semi-orthogonal to preserve
    pairwise distances (Johnson-Lindenstrauss property).

    Args:
        num_relations: Number of distinct relation types in the KG.
        high_dim: Dimension of the hyperbolic / tangent space (d).
        low_dim: Dimension of the projected Euclidean slice (k, k << d).
        learnable: If True, projection matrices are learned. If False, they are
                   fixed random orthogonal projections.
        orthogonal: If True, constrain P_r to be semi-orthogonal (P_r P_r^T ≈ I).
    """

    def __init__(
        self,
        num_relations: int,
        high_dim: int = 256,
        low_dim: int = 32,
        learnable: bool = True,
        orthogonal: bool = True,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.learnable = learnable
        self.orthogonal = orthogonal

        # Initialize P_r as random semi-orthogonal (JL-preserving, no rank collapse)
        if orthogonal:
            with torch.no_grad():
                projections = _random_orthogonal(
                    (num_relations, low_dim, high_dim)
                )
        else:
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

    def project_to_slice(
        self, tangent_nodes: torch.Tensor, relation_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Project tangent-space vectors into relation-specific Euclidean slices.

        Groups entities by relation and uses 2D matmul instead of bmm for
        significant speedup on CPU and GPU.

        Args:
            tangent_nodes: (batch_size, high_dim) — tangent-space vectors.
            relation_ids:  (batch_size,) — integer relation indices.

        Returns:
            (batch_size, low_dim) — projected vectors.
        """
        B, d = tangent_nodes.shape
        k = self.low_dim
        output = torch.empty(B, k, device=tangent_nodes.device, dtype=tangent_nodes.dtype)

        # Group entities by relation ID for efficient 2D matmul
        unique_rels = torch.unique(relation_ids)
        for rel in unique_rels:
            mask = (relation_ids == rel)
            indices = mask.nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue
            # P_r: (k, d), x[mask]: (n, d) → output: (n, k)
            output[indices] = torch.matmul(
                tangent_nodes[indices],       # (n, d)
                self.relation_projections[rel].t(),  # (d, k)
            )

        return output

    def orthogonalize_(self) -> None:
        """
        Enforce semi-orthogonality on P_r matrices (call periodically during training).
        Uses QR decomposition: for P_r of shape (k, d) with k < d,
        QR of P_r^T (d, k) → Q of (d, d) → keep first k columns → transpose to (k, d).
        """
        if not self.learnable:
            return
        with torch.no_grad():
            for r in range(self.num_relations):
                P = self.relation_projections[r]  # (k, d)
                q, _ = torch.linalg.qr(P.T)       # QR of (d, k) → q is (d, d)
                self.relation_projections[r] = q[:, :self.low_dim].T  # (k, d)

    def get_relation_embedding(
        self, relation_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get relation embedding in projected space. (batch_size, low_dim)."""
        return self.relation_embeddings[relation_ids]

    def get_projection_matrix(self, relation_id: int) -> torch.Tensor:
        """Get P_r for a single relation. (low_dim, high_dim)."""
        return self.relation_projections[relation_id]

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
