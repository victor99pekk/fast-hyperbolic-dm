"""
HyperbolicTransE: Hyperbolic TransE baseline for KG link prediction.

Entity embeddings live in the Poincaré ball. Relations are modeled as
translations in the tangent space, which are then mapped to hyperbolic
space via the exponential map.

Scoring: hyperbolic_distance(mobius_add(s, exp_map_zero(r)), o)

This serves as the "accuracy" baseline — captures hierarchical structure
well but is slow due to Möbius operations.
"""

import torch
import torch.nn as nn

from ..utils.hyperbolic import (
    mobius_add,
    hyperbolic_distance,
    exp_map_zero,
    poincare_clip,
)


class HyperbolicTransE(nn.Module):
    """
    Hyperbolic TransE model.

    Entity embeddings live in the Poincaré ball.
    Relation embeddings are tangent vectors at the origin.

    Args:
        num_entities: Number of entities.
        num_relations: Number of relations.
        dim: Embedding dimension.
        curvature: Poincaré ball curvature constant c (default 1.0).
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

        # Entity embeddings in the Poincaré ball (initialized near origin)
        self.entity_embeddings = nn.Parameter(
            torch.randn(num_entities, dim) * 1e-3
        )

        # Relation embeddings as tangent vectors at the origin
        self.relation_embeddings = nn.Parameter(
            torch.randn(num_relations, dim) * 1e-3
        )

    def _get_entity(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.entity_embeddings[ids]
        return poincare_clip(emb, c=self.curvature)

    def forward(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples using hyperbolic TransE scoring.

        Score = d_H(s ⊕_c exp_0(r), o)
        where d_H is hyperbolic distance and ⊕_c is Möbius addition.

        Args:
            subject_ids:  (batch_size,)
            relation_ids: (batch_size,)
            object_ids:   (batch_size,)

        Returns:
            scores: (batch_size,) — hyperbolic distance, lower = more plausible.
        """
        sub_emb = self._get_entity(subject_ids)
        rel_emb = self.relation_embeddings[relation_ids]
        obj_emb = self._get_entity(object_ids)

        # Map relation from tangent space to hyperbolic space
        rel_hyperbolic = exp_map_zero(rel_emb, c=self.curvature)

        # Translate subject by relation using Möbius addition
        translated = mobius_add(sub_emb, rel_hyperbolic, c=self.curvature)

        # Score: hyperbolic distance between translated subject and object
        scores = hyperbolic_distance(translated, obj_emb, c=self.curvature)

        return scores

    def score_triples(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(subject_ids, relation_ids, object_ids)
