"""
RotatE: Rotation-based Knowledge Graph Embedding (Sun et al., 2019, ICLR).

Entities and relations are embedded in complex vector space C^k.
Each relation is a rotation (element-wise phase multiplication) in complex space:
    h ∘ r ≈ t   where h, r, t ∈ C^k and |r_i| = 1

Scoring: ||h ∘ r - t||  (L2 distance in complex space)

RotatE can model symmetry (r = ±1), antisymmetry, inversion (r^{-1} = conj(r)),
and composition (r1 ∘ r2 = r3) — patterns that TransE cannot capture.

This is the standard non-hyperbolic SOTA baseline for KG link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by rotation in complex space.

    Entity embeddings: e ∈ C^k (stored as k pairs of (real, imag))
    Relation embeddings: θ_r ∈ R^k (phase angles, converted to e^{iθ_r})

    Scoring: ||e_s ∘ r - e_o||₂ where ∘ is element-wise complex multiplication.

    Args:
        num_entities: Number of entities.
        num_relations: Number of relations.
        dim: Embedding dimension (must be even; effective complex dim = dim // 2).
        gamma: Margin parameter for the scoring (fixed to 12.0 in original paper
               for distance-based scoring; we use 1.0 for hinge loss compatibility).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 256,
        gamma: float = 12.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "RotatE embedding dim must be even (complex pairs)"
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.gamma = gamma
        self.complex_dim = dim // 2  # k — number of complex dimensions

        # Entity embeddings in complex space: stored as 2*k real values
        # Uniform initialization in [-bound, bound] for both real and imag parts
        bound = 6.0 / (dim ** 0.5)
        self.entity_embeddings = nn.Parameter(
            torch.empty(num_entities, dim).uniform_(-bound, bound)
        )

        # Relation phase embeddings: θ_r ∈ R^k
        # Each component is a phase angle in [0, 2π)
        # Initialized uniformly in [0, 2π)
        self.relation_phases = nn.Parameter(
            torch.empty(num_relations, self.complex_dim).uniform_(0, 2 * torch.pi)
        )

    def _get_entity_complex(self, ids: torch.Tensor):
        """Get entity embeddings reshaped as complex values.

        Returns:
            re: (batch, k) real parts
            im: (batch, k) imaginary parts
        """
        emb = self.entity_embeddings[ids]  # (batch, 2*k)
        re, im = emb.chunk(2, dim=-1)  # each (batch, k)
        return re, im

    def _get_relation_complex(self, ids: torch.Tensor):
        """Get relation embeddings as complex rotation vectors.

        Returns:
            re: (batch, k) cos(θ)
            im: (batch, k) sin(θ)
        """
        phases = self.relation_phases[ids]  # (batch, k)
        return torch.cos(phases), torch.sin(phases)

    def forward(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples using RotatE scoring: ||h ∘ r - t||

        Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i

        Args:
            subject_ids:  (batch_size,)
            relation_ids: (batch_size,)
            object_ids:   (batch_size,)

        Returns:
            scores: (batch_size,) — distance, lower = more plausible.
        """
        h_re, h_im = self._get_entity_complex(subject_ids)  # (B, k)
        r_re, r_im = self._get_relation_complex(relation_ids)  # (B, k)
        t_re, t_im = self._get_entity_complex(object_ids)  # (B, k)

        # h ∘ r: complex Hadamard product
        # (h_re + i*h_im) * (r_re + i*r_im) = (h_re*r_re - h_im*r_im) + i*(h_re*r_im + h_im*r_re)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # ||h ∘ r - t||
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        scores = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-12).sum(dim=-1)

        return scores

    def score_triples(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(subject_ids, relation_ids, object_ids)
