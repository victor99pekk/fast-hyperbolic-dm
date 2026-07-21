"""
EuclideanTransE: Standard Euclidean TransE baseline for KG link prediction.

Scores triples (s, r, o) using: ||s + r - o||_2
All embeddings live in standard Euclidean space R^d.

This serves as the "speed" baseline — fast but poor at capturing hierarchical
relations that naturally live in hyperbolic space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanTransE(nn.Module):
    """
    Standard TransE model in Euclidean space.

    Args:
        num_entities: Number of entities.
        num_relations: Number of relations.
        dim: Embedding dimension.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 256,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Entity embeddings in Euclidean space
        bound = 6.0 / (dim ** 0.5)
        self.entity_embeddings = nn.Parameter(
            torch.empty(num_entities, dim).uniform_(-bound, bound)
        )

        # Relation embeddings
        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, dim).uniform_(-bound, bound)
        )

        # Normalize relation embeddings (common TransE practice)
        with torch.no_grad():
            self.relation_embeddings.data = F.normalize(
                self.relation_embeddings.data, p=2, dim=-1
            )

    def forward(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples using TransE scoring.

        Args:
            subject_ids:  (batch_size,)
            relation_ids: (batch_size,)
            object_ids:   (batch_size,)

        Returns:
            scores: (batch_size,) — L2 distance, lower = more plausible.
        """
        sub_emb = self.entity_embeddings[subject_ids]
        rel_emb = self.relation_embeddings[relation_ids]
        obj_emb = self.entity_embeddings[object_ids]

        # ||s + r - o||_2
        scores = torch.norm(sub_emb + rel_emb - obj_emb, p=2, dim=-1)
        return scores

    def score_triples(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(subject_ids, relation_ids, object_ids)
