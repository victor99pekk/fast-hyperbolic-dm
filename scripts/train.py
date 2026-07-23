#!/usr/bin/env python3
"""
Training entry point for Knowledge Graph link prediction models.

Usage:
    # Quick smoke test (CPU, 5 epochs, subset of data, ~2 min):
    python scripts/train.py --model dm --config configs/fb15k237.yaml --quick

    # Full training (auto-detects GPU if available):
    python scripts/train.py --model dm --config configs/fb15k237.yaml

    # Explicit device:
    python scripts/train.py --model dm --config configs/fb15k237.yaml --device cpu

Models:
    dm          - DM-KG Model (Dvoretzky-Milman projections)
    euclidean   - Euclidean TransE baseline
    hyperbolic  - Hyperbolic TransE baseline (Möbius ops)
"""

import argparse
import os
import sys
import yaml

import torch

# Add project root to path — works whether installed editable or run from scripts/
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.dataset import load_dataset
from src.data.negative_sampling import build_true_triples_set
from src.models import DM_KG_Model, EuclideanTransE, HyperbolicTransE
from src.training.trainer import KGTrainer


MODEL_REGISTRY = {
    "dm": DM_KG_Model,
    "euclidean": EuclideanTransE,
    "hyperbolic": HyperbolicTransE,
}

# Quick mode defaults — small data, few epochs, no eval overhead
QUICK_CONFIG = {
    "epochs": 100,
    "eval_every": 25,
    "patience": 2,
    "orthogonalize_every": 0,
    "batch_size": 256,
    "num_neg": 10,
    "high_dim": 64,
    "low_dim": 16,
    "num_train_triples": 5000,
    "num_valid_triples": 500,
}

# Medium mode — 4× data, double dims, for diagnostic scaling tests
MEDIUM_CONFIG = {
    "epochs": 100,
    "eval_every": 25,
    "patience": 2,
    "orthogonalize_every": 0,
    "batch_size": 512,
    "num_neg": 20,
    "high_dim": 128,
    "low_dim": 32,
    "num_train_triples": 20000,
    "num_valid_triples": 1000,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a KG link prediction model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to train",
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
        "--quick",
        action="store_true",
        help="Quick smoke test: 5K triples, dim=64, CPU",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium diagnostic run: 20K triples, dim=128, CPU",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["fb15k237", "wn18rr"],
        help="Dataset to use (default: from config or fb15k237)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(model_name: str, dataset, config: dict) -> torch.nn.Module:
    """Build the specified model with appropriate dimensions."""
    model_class = MODEL_REGISTRY[model_name]

    if model_name == "dm":
        return model_class(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            high_dim=config.get("high_dim", 256),
            low_dim=config.get("low_dim", 32),
            curvature=config.get("curvature", 1.0),
            learnable_projections=config.get("learnable_projections", True),
            orthogonal=config.get("orthogonal", False),
        )
    elif model_name == "euclidean":
        return model_class(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            dim=config.get("high_dim", 256),
        )
    elif model_name == "hyperbolic":
        return model_class(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            dim=config.get("high_dim", 256),
            curvature=config.get("curvature", 1.0),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _auto_summary(dataset_key: str) -> None:
    """Print results summary and speedup comparison for the current dataset."""
    import json
    from pathlib import Path

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    models = ["euclidean", "dm", "hyperbolic"]
    all_results = {}

    for model in models:
        tag = f"{model}_{dataset_key}"
        path = os.path.join(checkpoint_dir, tag, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results[model] = json.load(f)

    if not all_results:
        return

    print(f"\n{'='*60}")
    print(f"  Results for {dataset_key}")
    print(f"{'='*60}")
    print(f"  {'Model':<16} {'MRR':>8} {'Hits@10':>8}  {'Epoch(s)':>8} {'Fwd(ms)':>8}")
    print(f"  {'-'*54}")

    for model in models:
        r = all_results.get(model)
        if not r:
            continue
        timing = r.get("timing", {})
        epoch_s = timing.get("avg_epoch_time_s", 0)
        fwd_ms = timing.get("forward_pass_ms", 0)
        metrics = r.get("metrics", {})
        print(f"  {r['model']:<16} {r['best_mrr']:>8.4f} {metrics.get('hits@10',0):>8.4f}  "
              f"{epoch_s:>8.2f} {fwd_ms:>8.2f}")

    # Speedup if all three present
    if len(all_results) == 3:
        hyp = all_results["hyperbolic"]
        dm = all_results["dm"]
        euc = all_results["euclidean"]
        hyp_fwd = hyp.get("timing", {}).get("forward_pass_ms", 0)
        dm_fwd = dm.get("timing", {}).get("forward_pass_ms", 0)
        euc_fwd = euc.get("timing", {}).get("forward_pass_ms", 0)

        print(f"\n  Speed vs. DM-KG (lower ms = faster):")
        if dm_fwd > 0 and euc_fwd > 0:
            if euc_fwd < dm_fwd:
                print(f"    Euclidean:    {dm_fwd/euc_fwd:.1f}× faster than DM-KG ({euc_fwd:.1f}ms vs {dm_fwd:.1f}ms)")
            else:
                print(f"    Euclidean:    {euc_fwd/dm_fwd:.1f}× slower than DM-KG ({euc_fwd:.1f}ms vs {dm_fwd:.1f}ms)")
        if dm_fwd > 0 and hyp_fwd > 0:
            if dm_fwd < hyp_fwd:
                print(f"    Hyperbolic:   {hyp_fwd/dm_fwd:.1f}× slower than DM-KG ({hyp_fwd:.1f}ms vs {dm_fwd:.1f}ms)")
            else:
                print(f"    Hyperbolic:   {hyp_fwd/dm_fwd:.1f}× faster than DM-KG ({hyp_fwd:.1f}ms vs {dm_fwd:.1f}ms)")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        # Try relative to project root
        project_root = os.path.join(os.path.dirname(__file__), "..")
        config_path = os.path.join(project_root, args.config)
        config_path = os.path.abspath(config_path)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # ── Quick/Medium mode: override config ──
    if args.medium:
        print("\n🔬 MEDIUM MODE — 20K triples, dim=128/32, CPU")
        for key, value in MEDIUM_CONFIG.items():
            config[key] = value
        device = "cpu"
    elif args.quick:
        print("\n⚡ QUICK MODE — 5K triples, dim=64/16, CPU")
        for key, value in QUICK_CONFIG.items():
            config[key] = value
        device = "cpu"
    else:
        # Device — auto-detect GPU unless explicitly overridden
        device = args.device or config.get("device", "cpu")
        if device == "cpu" and torch.cuda.is_available() and not args.device:
            device = "cuda"
            print(f"GPU detected — auto-switching to {device}")
    print(f"Using device: {device}")

    # Load dataset
    dataset_key = args.dataset or config.get("dataset", "fb15k237")
    data_dir = config.get("data_dir", "data")
    data_dir = os.path.join(os.path.dirname(__file__), "..", data_dir)
    data_dir = os.path.abspath(data_dir)

    print(f"Loading {dataset_key} from: {data_dir}")
    dataset = load_dataset(dataset_key, data_dir)

    # ── Quick/Medium mode: subset the data ──
    if args.quick or args.medium:
        n_train = min(config["num_train_triples"], dataset.train_triples.shape[0])
        n_valid = min(config["num_valid_triples"], dataset.valid_triples.shape[0])
        dataset.train_triples = dataset.train_triples[:n_train]
        dataset.valid_triples = dataset.valid_triples[:n_valid]
        # Only use entities that appear in the subset
        active_entities = set()
        for s, _, o in dataset.train_triples.tolist():
            active_entities.add(s)
            active_entities.add(o)
        # Re-map entity IDs to a contiguous range
        old_to_new = {old: new for new, old in enumerate(sorted(active_entities))}
        dataset.train_triples = torch.LongTensor([
            [old_to_new[s], r, old_to_new[o]]
            for s, r, o in dataset.train_triples.tolist()
        ])
        dataset.valid_triples = torch.LongTensor([
            [old_to_new[s], r, old_to_new[o]]
            for s, r, o in dataset.valid_triples.tolist()
            if s in old_to_new and o in old_to_new
        ])
        dataset.num_entities = len(old_to_new)

    print(f"  Entities: {dataset.num_entities}")
    print(f"  Relations: {dataset.num_relations}")
    print(f"  Train triples: {dataset.train_triples.shape[0]}")
    print(f"  Valid triples: {dataset.valid_triples.shape[0]}")
    print(f"  Test triples: {dataset.test_triples.shape[0]}")

    # Build model
    print(f"\nBuilding model: {args.model}")
    model = build_model(args.model, dataset, config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Build true triples set — in quick/medium mode, use remapped data
    if args.quick or args.medium:
        all_true_triples = build_true_triples_set(
            dataset.train_triples,
            dataset.valid_triples,
            dataset.valid_triples,  # use valid twice (test not remapped)
        )
    else:
        all_true_triples = build_true_triples_set(
            dataset.train_triples,
            dataset.valid_triples,
            dataset.test_triples,
        )

    # Configure trainer paths — separate logs/checkpoints per dataset
    model_tag = f"{args.model}_{dataset_key}"
    config["log_dir"] = os.path.join(config.get("log_dir", "logs"), model_tag)
    config["checkpoint_dir"] = os.path.join(config.get("checkpoint_dir", "checkpoints"), model_tag)

    # Train
    trainer = KGTrainer(
        model=model,
        train_triples=dataset.train_triples,
        valid_triples=dataset.valid_triples,
        all_true_triples=all_true_triples,
        num_entities=dataset.num_entities,
        config=config,
        device=device,
    )

    results = trainer.fit()

    print(f"\nTraining complete!")
    print(f"Best validation MRR: {results['best_mrr']:.4f}")
    print(f"Best epoch: {results['best_epoch']}")

    # ── Auto-summary: print results + speedup ──
    _auto_summary(dataset_key)

    # Final test evaluation (skip in quick/medium mode — too slow)
    if not args.quick and not args.medium:
        print(f"\nEvaluating on test set...")
        from src.utils.metrics import evaluate_model

        test_metrics = evaluate_model(
            model=model,
            eval_triples=dataset.test_triples[:1000],  # sample for speed
            all_true_triples=all_true_triples,
            num_entities=dataset.num_entities,
            batch_size=config.get("batch_size", 256),
            device=device,
        )

        print(f"Test MRR:    {test_metrics['mrr']:.4f}")
        print(f"Test Hits@1: {test_metrics['hits@1']:.4f}")
        print(f"Test Hits@3: {test_metrics['hits@3']:.4f}")
        print(f"Test Hits@10: {test_metrics['hits@10']:.4f}")
    else:
        mode = "🔬 Medium" if args.medium else "⚡ Quick"
        print(f"\n{mode} mode — skipping test evaluation.")


if __name__ == "__main__":
    main()
