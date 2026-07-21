#!/usr/bin/env python3
"""
Benchmarking entry point: compare speed and accuracy across models.

Usage:
    python scripts/benchmark.py --config configs/fb15k237.yaml

Measures forward pass time, epoch time, GPU memory, and parameter count
for DM-KG, Euclidean TransE, and Hyperbolic TransE models.
"""

import argparse
import os
import sys
import json
import yaml

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_fb15k237
from src.models import DM_KG_Model, EuclideanTransE, HyperbolicTransE
from src.benchmarking.profiler import compare_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark KG link prediction models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fb15k237.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark.json",
        help="Output path for benchmark JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    config_path = os.path.abspath(config_path)
    config = yaml.safe_load(open(config_path))

    device = args.device or config.get("device", "cpu")
    if device == "cpu" and torch.cuda.is_available() and not args.device:
        device = "cuda"
        print(f"GPU detected — auto-switching to {device}")
    print(f"Device: {device}")

    # Load dataset
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", config.get("data_dir", "data/fb15k237")
    )
    data_dir = os.path.abspath(data_dir)
    dataset = load_fb15k237(data_dir)

    high_dim = config.get("high_dim", 256)
    low_dim = config.get("low_dim", 32)
    curvature = config.get("curvature", 1.0)

    # Build all models
    models = {
        "DM-KG": DM_KG_Model(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            high_dim=high_dim,
            low_dim=low_dim,
            curvature=curvature,
        ),
        "Euclidean TransE": EuclideanTransE(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            dim=high_dim,
        ),
        "Hyperbolic TransE": HyperbolicTransE(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            dim=high_dim,
            curvature=curvature,
        ),
    }

    batch_size = config.get("batch_size", 1024)
    num_neg = config.get("num_neg", 50)

    # Run benchmarks
    results = compare_models(
        models=models,
        train_triples=dataset.train_triples,
        num_entities=dataset.num_entities,
        batch_size=batch_size,
        num_neg=num_neg,
        device=device,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(
        f"{'Model':<20} {'Params':>10} {'Fwd(ms)':>10} "
        f"{'Epoch(s)':>10} {'Samp/s':>10} {'Mem(MB)':>10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['model']:<20} {r['num_params']:>10,} "
            f"{r['forward_mean_ms']:>10.2f} {r['epoch_time_sec']:>10.1f} "
            f"{r['samples_per_sec']:>10.0f} {r['peak_memory_mb']:>10.1f}"
        )
    print("=" * 80)

    # Compute speedups
    dm_result = next(r for r in results if r["model"] == "DM-KG")
    hyp_result = next(r for r in results if r["model"] == "Hyperbolic TransE")
    euc_result = next(r for r in results if r["model"] == "Euclidean TransE")

    if hyp_result["forward_mean_ms"] > 0:
        fwd_speedup = hyp_result["forward_mean_ms"] / dm_result["forward_mean_ms"]
        print(f"\nDM-KG forward speedup vs Hyperbolic: {fwd_speedup:.2f}x")

    if euc_result["forward_mean_ms"] > 0:
        fwd_vs_euc = dm_result["forward_mean_ms"] / euc_result["forward_mean_ms"]
        print(f"DM-KG forward slowdown vs Euclidean: {fwd_vs_euc:.2f}x")

    # Save results
    output_path = os.path.join(
        os.path.dirname(__file__), "..", args.output
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
