"""
Negative sampling utilities for Knowledge Graph link prediction.

Generates corrupted triples by replacing either the head or tail entity
with a randomly sampled entity. Supports both uniform and filtered settings.

In the filtered setting, we ensure the corrupted triple does not appear in
the set of known true triples (train/valid/test).
"""

import torch
import numpy as np
from typing import Set, Tuple, Optional


class NegativeSampler:
    """
    Negative sampler for KG link prediction.

    For each positive triple (s, r, o), generates num_neg corrupted
    triples by replacing the head or tail with a random entity.

    Args:
        num_entities: Total number of entities.
        all_true_triples: Set of all known true triples as (s, r, o) tuples,
                          used for filtered evaluation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_entities: int,
        all_true_triples: Optional[Set[Tuple[int, int, int]]] = None,
        seed: int = 42,
    ):
        self.num_entities = num_entities
        self.all_true_triples = all_true_triples or set()
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        subjects: torch.LongTensor,
        relations: torch.LongTensor,
        objects: torch.LongTensor,
        num_neg: int = 1,
        filtered: bool = True,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Generate negative (corrupted) triples for a batch of positive triples.

        Args:
            subjects:  (batch_size,) positive subject IDs.
            relations: (batch_size,) positive relation IDs.
            objects:   (batch_size,) positive object IDs.
            num_neg:   Number of negative samples per positive triple.
            filtered:  If True, discard corrupted triples that appear in
                       the known true triples set.

        Returns:
            neg_subjects, neg_relations, neg_objects:
                Each of shape (batch_size * num_neg,).
        """
        batch_size = subjects.shape[0]
        total_negs = batch_size * num_neg

        # Repeat positive triples num_neg times
        neg_subjects = subjects.repeat_interleave(num_neg)
        neg_relations = relations.repeat_interleave(num_neg)
        neg_objects = objects.repeat_interleave(num_neg)

        # Randomly choose whether to corrupt head or tail for each negative
        corrupt_head = torch.from_numpy(
            self.rng.binomial(1, 0.5, size=total_negs).astype(bool)
        )

        # Generate random entity IDs
        random_entities = torch.from_numpy(
            self.rng.randint(0, self.num_entities, size=total_negs).astype(
                np.int64
            )
        )

        if filtered:
            # Filter out false negatives (corruptions that happen to be true triples)
            for i in range(total_negs):
                while True:
                    if corrupt_head[i]:
                        candidate = (
                            random_entities[i].item(),
                            neg_relations[i].item(),
                            neg_objects[i].item(),
                        )
                    else:
                        candidate = (
                            neg_subjects[i].item(),
                            neg_relations[i].item(),
                            random_entities[i].item(),
                        )

                    if candidate not in self.all_true_triples:
                        break

                    # Resample
                    random_entities[i] = self.rng.randint(
                        0, self.num_entities
                    )

        # Apply corruptions
        neg_subjects = torch.where(
            corrupt_head, random_entities, neg_subjects
        )
        neg_objects = torch.where(
            ~corrupt_head, random_entities, neg_objects
        )

        return neg_subjects, neg_relations, neg_objects


def build_true_triples_set(
    train_triples: torch.LongTensor,
    valid_triples: torch.LongTensor,
    test_triples: torch.LongTensor,
) -> Set[Tuple[int, int, int]]:
    """
    Build a set of all known true triples for filtered evaluation/sampling.

    Args:
        train_triples: (N_train, 3)
        valid_triples: (N_valid, 3)
        test_triples:  (N_test, 3)

    Returns:
        Set of (s, r, o) tuples.
    """
    all_triples = torch.cat([train_triples, valid_triples, test_triples], dim=0)
    return {
        (int(s), int(r), int(o))
        for s, r, o in all_triples.tolist()
    }
