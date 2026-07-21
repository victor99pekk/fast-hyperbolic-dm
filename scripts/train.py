#!/usr/bin/env python3
"""
Training entry point for Knowledge Graph link prediction models.

Usage:
    python scripts/train.py --model dm --config configs/fb15k237.yaml
    python scripts/train.py --model euclidean --config configs/fb15k237.yaml
    python scripts/train.py --model hyperbolic --config configs/fb15k237.yaml

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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_fb15k237
from src.data.negative_sampling import build_true_triples_set
from src.models import DM_KG_Model, EuclideanTransE, HyperbolicTransE
from src.training.trainer import KGTrainer


MODEL_REGISTRY = {
    "dm": DM_KG_Model,
    "euclidean": EuclideanTransE,
    "hyperbolic": HyperbolicTransE,
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
            learnable_projections=config.get(
                "learnable_projections", True
            ),
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

    # Device — auto-detect GPU unless explicitly overridden
    device = args.device or config.get("device", "cpu")
    if device == "cpu" and torch.cuda.is_available() and not args.device:
        device = "cuda"
        print(f"GPU detected — auto-switching to {device}")
    print(f"Using device: {device}")

    # Load dataset
    data_dir = config.get("data_dir", "data/fb15k237")
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", data_dir
    )
    data_dir = os.path.abspath(data_dir)

    print(f"Loading FB15k-237 from: {data_dir}")
    dataset = load_fb15k237(data_dir)
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

    # Build true triples set for filtered evaluation
    all_true_triples = build_true_triples_set(
        dataset.train_triples,
        dataset.valid_triples,
        dataset.test_triples,
    )

    # Configure trainer paths
    model_name = args.model
    config["log_dir"] = os.path.join(
        config.get("log_dir", "logs"), model_name
    )
    config["checkpoint_dir"] = os.path.join(
        config.get("checkpoint_dir", "checkpoints"), model_name
    )

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

    # Final test evaluation
    print(f"\nEvaluating on test set...")
    from src.utils.metrics import evaluate_model

    test_metrics = evaluate_model(
        model=model,
        eval_triples=dataset.test_triples,
        all_true_triples=all_true_triples,
        num_entities=dataset.num_entities,
        batch_size=config.get("batch_size", 256),
        device=device,
    )

    print(f"Test MRR:    {test_metrics['mrr']:.4f}")
    print(f"Test Hits@1: {test_metrics['hits@1']:.4f}")
    print(f"Test Hits@3: {test_metrics['hits@3']:.4f}")
    print(f"Test Hits@10: {test_metrics['hits@10']:.4f}")


if __name__ == "__main__":
    main()
