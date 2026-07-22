"""
RotL: Rotation-based Lorentz KG Embedding Model.

Entities live on the Lorentz (hyperboloid) model of hyperbolic geometry.
Relations are modeled as spatial rotations — a subgroup of Lorentz transformations
that preserve the time coordinate.

Key insight: The Lorentz model uses simpler operations (matrix multiplication for
rotations, arcosh-based distance) compared to the Poincaré ball (Möbius addition,
fractional linear transforms). This gives near-hyperbolic expressiveness at a
fraction of the computational cost of Möbius-based hyperbolic models.

The Lorentz hyperboloid: H^{d,K} = {x ∈ R^{d+1} : ⟨x,x⟩_L = -1/K, x_0 > 0}
Lorentz inner product: ⟨x,y⟩_L = -x_0*y_0 + Σ_{i=1}^d x_i*y_i

Scoring: ||R_r(x_s) - x_o||₂ (Euclidean distance in ambient R^{d+1})
where R_r is a spatial rotation applied to the spatial coordinates of the subject.

This is the nearest competitor to DM-KG: both claim "near-hyperbolic accuracy
at near-Euclidean speed", but via different mechanisms — RotL simplifies the
hyperbolic operations algebraically (Lorentz → matrix ops), while DM-KG
projects to Euclidean slices and scores there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotL(nn.Module):
    """
    Rotation-based Lorentz KG Embedding Model.

    Entities are parameterized by their spatial coordinates x ∈ R^d.
    The time coordinate x_0 is computed on the fly to keep points on the hyperboloid:
        x_0 = sqrt(1/K + ||x_spatial||^2)

    Relations are learnable spatial rotation matrices R_r ∈ R^{d×d}.

    Args:
        num_entities: Number of entities.
        num_relations: Number of relations.
        dim: Spatial dimension d (total embedding dim = d + 1).
        curvature: Curvature constant K (default 1.0).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 256,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.curvature = curvature

        # Entity embeddings: spatial coordinates in R^d
        # Initialized near origin (small norm → x_0 ≈ 1/sqrt(K))
        bound = 1e-3
        self.entity_embeddings = nn.Parameter(
            torch.empty(num_entities, dim).uniform_(-bound, bound)
        )

        # Relation rotation matrices: R_r ∈ R^{d×d}
        # Initialized near identity: R_r ≈ I + small perturbation
        # This corresponds to a small rotation, which is a natural starting point
        eye = torch.eye(dim).unsqueeze(0).expand(num_relations, -1, -1).clone()
        noise = torch.randn(num_relations, dim, dim) * 0.01
        self.relation_rotations = nn.Parameter(eye + noise)

    def _to_hyperboloid(self, spatial: torch.Tensor) -> torch.Tensor:
        """
        Lift spatial coordinates to the Lorentz hyperboloid.

        Given spatial part x ∈ R^d, compute the full hyperboloid point:
            X = (sqrt(1/K + ||x||²), x) ∈ R^{d+1}

        Args:
            spatial: (..., d) spatial coordinates.

        Returns:
            hyperboloid_point: (..., d+1) with X_0 = sqrt(1/K + ||x_spatial||²).
        """
        sq_norm = (spatial ** 2).sum(dim=-1, keepdim=True)  # (..., 1)
        x0 = torch.sqrt(1.0 / self.curvature + sq_norm)  # (..., 1)
        return torch.cat([x0, spatial], dim=-1)  # (..., d+1)

    def _apply_spatial_rotation(
        self, spatial: torch.Tensor, rotation: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply relation-specific spatial rotation.

        Args:
            spatial: (batch, d) spatial coordinates.
            rotation: (batch, d, d) rotation matrices.

        Returns:
            rotated: (batch, d) rotated spatial coordinates.
        """
        # (batch, d) → (batch, 1, d) → bmm with (batch, d, d) → (batch, 1, d) → (batch, d)
        return torch.bmm(rotation, spatial.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples using RotL scoring.

        1. Get subject and object spatial coordinates
        2. Apply relation-specific rotation to subject
        3. Lift both to hyperboloid
        4. Score: Euclidean distance in ambient R^{d+1}

        Args:
            subject_ids:  (batch_size,)
            relation_ids: (batch_size,)
            object_ids:   (batch_size,)

        Returns:
            scores: (batch_size,) — distance, lower = more plausible.
        """
        sub_spatial = self.entity_embeddings[subject_ids]  # (B, d)
        obj_spatial = self.entity_embeddings[object_ids]    # (B, d)

        # Get relation rotation matrices
        R_r = self.relation_rotations[relation_ids]  # (B, d, d)

        # Rotate subject in spatial coordinates
        rotated_sub_spatial = self._apply_spatial_rotation(sub_spatial, R_r)  # (B, d)

        # Lift to hyperboloid
        sub_hyp = self._to_hyperboloid(rotated_sub_spatial)  # (B, d+1)
        obj_hyp = self._to_hyperboloid(obj_spatial)          # (B, d+1)

        # Score: Euclidean distance in ambient R^{d+1}
        # (For points on the hyperboloid, this approximates hyperbolic distance
        #  while being computationally simpler)
        scores = torch.norm(sub_hyp - obj_hyp, p=2, dim=-1)  # (B,)

        return scores

    def score_triples(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(subject_ids, relation_ids, object_ids)

    def lorentzian_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute proper Lorentzian (hyperbolic) distance between points.

        d_L(x,y) = sqrt(1/K) * arcosh(-K * ⟨x,y⟩_L)

        Args:
            x, y: (..., d+1) points on the hyperboloid.

        Returns:
            distance: (...)
        """
        # Lorentz inner product: ⟨x,y⟩_L = -x_0*y_0 + Σ x_i*y_i
        x0, x_spatial = x[..., 0:1], x[..., 1:]
        y0, y_spatial = y[..., 0:1], y[..., 1:]
        inner = -x0 * y0 + (x_spatial * y_spatial).sum(dim=-1, keepdim=True)
        inner = torch.clamp(inner, max=-1.0 / self.curvature - 1e-8)
        arg = -self.curvature * inner
        arg = torch.clamp(arg, min=1.0 + 1e-8)
        return torch.sqrt(1.0 / self.curvature) * torch.arccosh(arg).squeeze(-1)
