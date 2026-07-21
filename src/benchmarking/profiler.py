"""
Benchmarking profiler for KG models.

Measures forward/backward pass time, full epoch time, and GPU memory usage
to compare the speed of DM projections vs. Euclidean vs. hyperbolic baselines.
"""

import time
import json
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def measure_forward_time(
    model: nn.Module,
    subjects: torch.Tensor,
    relations: torch.Tensor,
    objects: torch.Tensor,
    num_warmup: int = 10,
    num_repeats: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure forward pass timing using CUDA events (GPU) or time.time (CPU).

    Args:
        model: Model to benchmark.
        subjects, relations, objects: Input tensors (batch_size,).
        num_warmup: Number of warmup iterations (not measured).
        num_repeats: Number of timed iterations.
        device: Torch device.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'.
    """
    model = model.to(device)
    model.eval()

    subjects = subjects.to(device)
    relations = relations.to(device)
    objects = objects.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.score_triples(subjects, relations, objects)

    # Measure
    if device.startswith("cuda"):
        return _measure_with_cuda_events(
            model, subjects, relations, objects, num_repeats, device
        )
    else:
        return _measure_with_timer(
            model, subjects, relations, objects, num_repeats
        )


def _measure_with_cuda_events(model, s, r, o, num_repeats, device):
    """Measure forward pass time using CUDA events."""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    timings = []
    with torch.no_grad():
        for _ in range(num_repeats):
            starter.record()
            _ = model.score_triples(s, r, o)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    timings = torch.tensor(timings)
    return {
        "mean_ms": timings.mean().item(),
        "std_ms": timings.std().item(),
        "min_ms": timings.min().item(),
        "max_ms": timings.max().item(),
    }


def _measure_with_timer(model, s, r, o, num_repeats):
    """Measure forward pass time using time.time (CPU fallback)."""
    timings = []
    with torch.no_grad():
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            _ = model.score_triples(s, r, o)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    timings = torch.tensor(timings)
    return {
        "mean_ms": timings.mean().item(),
        "std_ms": timings.std().item(),
        "min_ms": timings.min().item(),
        "max_ms": timings.max().item(),
    }


def measure_memory(
    model: nn.Module,
    subjects: torch.Tensor,
    relations: torch.Tensor,
    objects: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure peak GPU memory usage during forward+backward pass.

    Args:
        model: Model to benchmark.
        subjects, relations, objects: Input tensors.
        device: Torch device (should be cuda).

    Returns:
        Dict with 'peak_memory_mb'.
    """
    if not device.startswith("cuda"):
        return {"peak_memory_mb": 0.0}

    model = model.to(device)
    model.train()

    subjects = subjects.to(device)
    relations = relations.to(device)
    objects = objects.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Forward + backward
    scores = model.score_triples(subjects, relations, objects)
    loss = scores.mean()
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {"peak_memory_mb": peak_mb}


def benchmark_epoch(
    model: nn.Module,
    train_triples: torch.LongTensor,
    num_entities: int,
    batch_size: int = 1024,
    num_neg: int = 50,
    device: str = "cpu",
    num_batches: int = 50,
) -> Dict[str, float]:
    """
    Simulate one training epoch and measure total wall-clock time.

    Args:
        model: Model to benchmark.
        train_triples: Training triples.
        num_entities: Number of entities (for negative sampling).
        batch_size: Batch size.
        num_neg: Number of negatives per positive.
        device: Torch device.
        num_batches: Number of batches to simulate (use None for all).

    Returns:
        Dict with 'epoch_time_sec', 'batches_per_sec', 'samples_per_sec'.
    """
    from ..data.negative_sampling import NegativeSampler

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    neg_sampler = NegativeSampler(num_entities=num_entities)

    # Prepare data
    total_batches = (
        min(num_batches, len(train_triples) // batch_size)
        if num_batches
        else len(train_triples) // batch_size
    )

    # Shuffle indices
    perm = torch.randperm(len(train_triples))
    triples = train_triples[perm]

    t0 = time.perf_counter()

    for b in range(total_batches):
        start = b * batch_size
        end = start + batch_size
        batch = triples[start:end]

        s = batch[:, 0].to(device)
        r = batch[:, 1].to(device)
        o = batch[:, 2].to(device)

        neg_s, neg_r, neg_o = neg_sampler.sample(
            s.cpu(), r.cpu(), o.cpu(), num_neg=num_neg, filtered=False
        )
        neg_s = neg_s.to(device)
        neg_r = neg_r.to(device)
        neg_o = neg_o.to(device)

        pos_scores = model.score_triples(s, r, o)
        neg_scores = model.score_triples(neg_s, neg_r, neg_o)

        loss = torch.clamp(1.0 + pos_scores.unsqueeze(1) - neg_scores.view(len(s), -1), min=0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t1 = time.perf_counter()
    elapsed = t1 - t0

    total_samples = total_batches * batch_size * (1 + num_neg)

    return {
        "epoch_time_sec": elapsed,
        "batches_per_sec": total_batches / elapsed,
        "samples_per_sec": total_samples / elapsed,
    }


def compare_models(
    models: Dict[str, nn.Module],
    train_triples: torch.LongTensor,
    num_entities: int,
    batch_size: int = 1024,
    num_neg: int = 50,
    device: str = "cpu",
) -> List[Dict]:
    """
    Benchmark multiple models and return comparison results.

    Args:
        models: Dict mapping model name to model instance.
        train_triples: Training triples for epoch simulation.
        num_entities: Number of entities.
        batch_size: Batch size.
        num_neg: Number of negatives.
        device: Torch device.

    Returns:
        List of dicts with benchmark results per model.
    """
    results = []

    # Sample a batch for forward-time measurement
    perm = torch.randperm(len(train_triples))[:batch_size]
    s = train_triples[perm, 0]
    r = train_triples[perm, 1]
    o = train_triples[perm, 2]

    for name, model in models.items():
        print(f"\nBenchmarking {name}...")
        model = model.to(device)

        # Forward pass timing
        fwd = measure_forward_time(
            model, s, r, o, device=device
        )

        # Epoch timing
        epoch = benchmark_epoch(
            model,
            train_triples,
            num_entities=num_entities,
            batch_size=batch_size,
            num_neg=num_neg,
            device=device,
            num_batches=30,
        )

        # Memory
        mem = measure_memory(model, s, r, o, device=device)

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())

        result = {
            "model": name,
            "num_params": n_params,
            "forward_mean_ms": fwd["mean_ms"],
            "forward_std_ms": fwd["std_ms"],
            "epoch_time_sec": epoch["epoch_time_sec"],
            "samples_per_sec": epoch["samples_per_sec"],
            "peak_memory_mb": mem["peak_memory_mb"],
        }
        results.append(result)

        print(
            f"  Forward: {fwd['mean_ms']:.2f}±{fwd['std_ms']:.2f} ms | "
            f"Epoch: {epoch['epoch_time_sec']:.1f}s | "
            f"Mem: {mem['peak_memory_mb']:.1f} MB"
        )

    return results
