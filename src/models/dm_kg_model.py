"""
DM_KG_Model: Dvoretzky-Milman Knowledge Graph Embedding Model.

Stores entity embeddings in high-dimensional hyperbolic space (Poincaré ball),
maps them to relation-specific Euclidean slices via DM projection,
and scores triples using TransE-style scoring in the projected space.

Architecture:
    1. Entity embeddings live in Poincaré ball D^d
    2. For scoring a triple (s, r, o):
       a. Clip embeddings to stay in ball
       b. Map to tangent space via log_map_zero
       c. Project to relation-specific slice via DMRelationalLayer
       d. Score: ||s_proj + r_proj - o_proj||_2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import DMRelationalLayer
from ..utils.hyperbolic import log_map_zero, exp_map_zero, poincare_clip


class DM_KG_Model(nn.Module):
    """
    Dvoretzky-Milman Knowledge Graph Embedding Model.

    Args:
        num_entities: Number of entities in the KG.
        num_relations: Number of relation types.
        high_dim: Hyperbolic embedding dimension (d).
        low_dim: Projected slice dimension (k).
        curvature: Poincaré ball curvature constant c (default 1.0).
        learnable_projections: If True, DM projections are learned.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        high_dim: int = 256,
        low_dim: int = 32,
        curvature: float = 1.0,
        learnable_projections: bool = True,
        orthogonal: bool = True,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.curvature = curvature

        # Entity embeddings in high-dimensional hyperbolic space
        # Initialized near the origin with small variance
        self.entity_embeddings = nn.Parameter(
            torch.randn(num_entities, high_dim) * 1e-3
        )

        # DM relational projection layer
        self.dm_layer = DMRelationalLayer(
            num_relations=num_relations,
            high_dim=high_dim,
            low_dim=low_dim,
            learnable=learnable_projections,
            orthogonal=orthogonal,
        )

    def _get_entity(self, ids: torch.Tensor) -> torch.Tensor:
        """Fetch entity embeddings and clip to stay inside the Poincaré ball."""
        emb = self.entity_embeddings[ids]
        return poincare_clip(emb, c=self.curvature)

    def forward(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples (subject, relation, object) using TransE scoring
        in the relation-specific DM projected space.

        Args:
            subject_ids:  (batch_size,)
            relation_ids: (batch_size,)
            object_ids:   (batch_size,)

        Returns:
            scores: (batch_size,) — lower = more likely (distance-based).
        """
        # 1. Fetch entity embeddings (clipped to Poincaré ball)
        sub_emb = self._get_entity(subject_ids)
        obj_emb = self._get_entity(object_ids)

        # 2. Map from hyperbolic to tangent (Euclidean) space
        sub_tan = log_map_zero(sub_emb, c=self.curvature)
        obj_tan = log_map_zero(obj_emb, c=self.curvature)

        # 3. DM projection to relation-specific Euclidean slice
        sub_proj = self.dm_layer.project_to_slice(sub_tan, relation_ids)
        obj_proj = self.dm_layer.project_to_slice(obj_tan, relation_ids)

        # 4. Fetch relation embeddings in the projected space
        rel_proj = self.dm_layer.get_relation_embedding(relation_ids)

        # 5. TransE-style scoring: ||s_proj + r_proj - o_proj||_2
        scores = torch.norm(sub_proj + rel_proj - obj_proj, p=2, dim=-1)

        return scores

    def score_triples(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples — same as forward, provided for explicit API clarity.
        Returns distances (lower = more plausible).
        """
        return self.forward(subject_ids, relation_ids, object_ids)

    def get_all_entity_embeddings(self) -> torch.Tensor:
        """Return all entity embeddings (in hyperbolic space, clipped)."""
        return poincare_clip(self.entity_embeddings, c=self.curvature)
