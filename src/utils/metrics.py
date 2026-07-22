"""
Evaluation metrics for Knowledge Graph link prediction.

Implements filtered MRR and Hits@K (K=1,3,10) — the standard metrics
for KG completion tasks.
"""

import torch
from typing import Set, Tuple, Dict, List


def compute_rank(
    pos_score: float,
    all_scores: torch.Tensor,
    descending: bool = False,
) -> int:
    """
    Compute the rank (1-indexed) of the positive score among all scores.

    Args:
        pos_score: Score of the true positive triple.
        all_scores: Scores for all candidates (including the positive).
        descending: If True, higher scores are better. If False (default),
                   lower scores are better (distance-based).

    Returns:
        Rank (1-indexed, 1 = best).
    """
    if descending:
        # Higher is better
        rank = (all_scores > pos_score).sum().item() + 1
    else:
        # Lower is better
        rank = (all_scores < pos_score).sum().item() + 1
        # Handle ties: count how many have exactly equal scores
        ties = (all_scores == pos_score).sum().item() - 1
        # Random tie-breaking: expected rank = rank + ties/2
        rank = rank + ties / 2.0
    return int(rank)


def compute_mrr(ranks: List[float]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not ranks:
        return 0.0
    # Filter out invalid ranks (0 or negative) and clip to min 1
    valid_ranks = [max(1.0, r) for r in ranks if r > 0]
    if not valid_ranks:
        return 0.0
    return sum(1.0 / r for r in valid_ranks) / len(valid_ranks)


def compute_hits_at_k(ranks: List[float], k: int) -> float:
    """Compute Hits@K (proportion of ranks ≤ K)."""
    if not ranks:
        return 0.0
    return sum(1.0 for r in ranks if r <= k) / len(ranks)


@torch.no_grad()
def evaluate_model(
    model,
    eval_triples: torch.LongTensor,
    all_true_triples: Set[Tuple[int, int, int]],
    num_entities: int,
    batch_size: int = 256,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate a KG model on link prediction (filtered setting).

    For each test triple (s, r, o), we corrupt the head (replace s with all
    entities) and the tail (replace o with all entities), compute scores for
    all corruptions, and determine the filtered rank of the true triple.

    Filtered: candidates that appear in all_true_triples (other than the
    test triple itself) are ignored when computing the rank.

    Args:
        model: Model with a score_triples(s, r, o) method returning distances.
        eval_triples: (N, 3) tensor of (s, r, o) triples to evaluate.
        all_true_triples: Set of all known true (s, r, o) tuples.
        num_entities: Total number of entities.
        batch_size: Batch size for scoring candidates.
        device: Device to run evaluation on.

    Returns:
        Dict with keys: 'mrr', 'hits@1', 'hits@3', 'hits@10'.
    """
    model.eval()
    model = model.to(device)

    ranks_head = []
    ranks_tail = []

    for i, triple in enumerate(eval_triples):
        s, r, o = triple[0].item(), triple[1].item(), triple[2].item()

        # --- Tail prediction: score (s, r, ?) for all entities ---
        tail_ranks = _compute_filtered_rank(
            model=model,
            head=s,
            relation=r,
            true_tail=o,
            corrupt_head=False,
            num_entities=num_entities,
            all_true_triples=all_true_triples,
            device=device,
            batch_size=batch_size,
        )
        ranks_tail.append(tail_ranks)

        # --- Head prediction: score (?, r, o) for all entities ---
        head_ranks = _compute_filtered_rank(
            model=model,
            head=s,
            relation=r,
            true_tail=o,
            corrupt_head=True,
            num_entities=num_entities,
            all_true_triples=all_true_triples,
            device=device,
            batch_size=batch_size,
        )
        ranks_head.append(head_ranks)

        if (i + 1) % 1000 == 0:
            print(
                f"  Evaluated {i+1}/{len(eval_triples)} triples "
                f"(MRR: {compute_mrr(ranks_head + ranks_tail):.4f})"
            )

    all_ranks = ranks_head + ranks_tail

    metrics = {
        "mrr": compute_mrr(all_ranks),
        "hits@1": compute_hits_at_k(all_ranks, 1),
        "hits@3": compute_hits_at_k(all_ranks, 3),
        "hits@10": compute_hits_at_k(all_ranks, 10),
        "mrr_head": compute_mrr(ranks_head),
        "mrr_tail": compute_mrr(ranks_tail),
    }

    model.train()
    return metrics


def _compute_filtered_rank(
    model,
    head: int,
    relation: int,
    true_tail: int,
    corrupt_head: bool,
    num_entities: int,
    all_true_triples: Set[Tuple[int, int, int]],
    device: str,
    batch_size: int = 256,
) -> float:
    """
    Compute filtered rank for one corruption direction.

    Args:
        corrupt_head: If True, corrupt the head (predict ? for given r, o).
                     If False, corrupt the tail (predict ? for given s, r).
    """
    all_scores = []
    true_score = None

    for start in range(0, num_entities, batch_size):
        end = min(start + batch_size, num_entities)
        batch_candidates = torch.arange(start, end, device=device)

        if corrupt_head:
            s_batch = batch_candidates
            r_batch = torch.full_like(batch_candidates, relation)
            o_batch = torch.full_like(batch_candidates, true_tail)
        else:
            s_batch = torch.full_like(batch_candidates, head)
            r_batch = torch.full_like(batch_candidates, relation)
            o_batch = batch_candidates

        scores = model.score_triples(s_batch, r_batch, o_batch).cpu()

        # Check which candidates are filtered (known true triples)
        for j, cand in enumerate(range(start, end)):
            if corrupt_head:
                triple = (cand, relation, true_tail)
            else:
                triple = (head, relation, cand)

            # Don't filter the test triple itself
            if triple == (head, relation, true_tail):
                true_score = scores[j].item()
                all_scores.append(float("inf"))  # Placeholder, will be replaced
            elif triple in all_true_triples:
                all_scores.append(float("inf"))  # Filter: push to bottom
            else:
                all_scores.append(scores[j].item())

    # Replace the placeholder with the true score
    all_scores_tensor = torch.tensor(all_scores)
    rank = compute_rank(true_score, all_scores_tensor, descending=False)

    return rank
