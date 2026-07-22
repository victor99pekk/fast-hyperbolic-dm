#!/usr/bin/env python3
"""Parse training results from JSON files or TensorBoard logs."""
import sys, os, json, math
from collections import defaultdict
from pathlib import Path


def parse_from_json(json_path: str) -> dict | None:
    """Read results from a results.json file."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    timing = data.get("timing", {})
    return {
        "model": data.get("model", "?"),
        "label": "",
        "mrr": data.get("best_mrr", float("nan")),
        "hits@1": metrics.get("hits@1", float("nan")),
        "hits@10": metrics.get("hits@10", float("nan")),
        "loss": float("nan"),
        "epoch_time_s": timing.get("avg_epoch_time_s", float("nan")),
        "fwd_ms": timing.get("forward_pass_ms", float("nan")),
        "infer_1k_ms": timing.get("inference_ms_per_1k", float("nan")),
        "total_time_s": timing.get("total_train_time_s", float("nan")),
        "num_epochs": timing.get("num_epochs_timed", 0),
    }


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    base = os.path.abspath(base)

    checkpoint_dir = os.path.join(base, "checkpoints")
    log_dir = os.path.join(base, "logs")

    models = ["dm", "euclidean", "hyperbolic"]
    datasets = ["fb15k237", "wn18rr"]
    results = {}

    # Scan all model + dataset combinations
    for dataset in datasets:
        for model in models:
            tag = f"{model}_{dataset}"
            json_path = os.path.join(checkpoint_dir, tag, "results.json")
            r = parse_from_json(json_path)
            if r:
                r["label"] = r["model"]
                r["dataset"] = dataset
                results[tag] = r

    # Also scan old-style directories (just model name, no dataset suffix)
    for model in models:
        if model in results:
            continue
        json_path = os.path.join(checkpoint_dir, model, "results.json")
        r = parse_from_json(json_path)
        if r:
            r["label"] = r["model"]
            r["dataset"] = "unknown"
            results[model] = r

    # Fall back to TensorBoard for models without JSON
    if len(results) < len(models):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            for model in models:
                if model in results:
                    continue
                path = os.path.join(log_dir, model)
                if not os.path.isdir(path):
                    continue
                ea = EventAccumulator(path)
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                vals = {t: ea.Scalars(t)[-1].value for t in tags if ea.Scalars(t)}
                results[model] = {
                    "model": model,
                    "dataset": "unknown",
                    "mrr": vals.get("valid/mrr", float("nan")),
                    "hits@1": vals.get("valid/hits@1", float("nan")),
                    "hits@10": vals.get("valid/hits@10", float("nan")),
                    "loss": vals.get("train/loss", float("nan")),
                }
        except ImportError:
            pass

    if not results:
        print("No results found. Run training first.", file=sys.stderr)
        sys.exit(1)

    # Group results by dataset
    by_dataset = defaultdict(list)
    for r in results.values():
        by_dataset[r.get("dataset", "unknown")].append(r)

    # Sort datasets: known ones first (alphabetical), unknown last
    dataset_order = sorted([d for d in by_dataset if d != "unknown"]) + (["unknown"] if "unknown" in by_dataset else [])

    # Build per-dataset display with DM-KG speedup annotations
    model_names = {
        "DM_KG_Model": "DM-KG (ours)",
        "EuclideanTransE": "Euclidean",
        "HyperbolicTransE": "Hyperbolic",
    }

    for dataset in dataset_order:
        if dataset == "unknown":
            continue  # skip old untagged results
        entries = by_dataset[dataset]
        if not entries:
            continue

        # Sort: DM-KG first, then Euclidean, then Hyperbolic
        model_order = {"DM_KG_Model": 0, "EuclideanTransE": 1, "HyperbolicTransE": 2}
        entries.sort(key=lambda r: model_order.get(r["model"], 99))

        label = dataset.upper() if dataset != "unknown" else "UNKNOWN DATASET"
        print(f"\n{'─'*70}")
        print(f"  {label}")
        print(f"{'─'*70}")
        print(f"  {'Model':<18} {'MRR ↑':>8} {'Hits@10':>8}  {'Speed':>10}  {'vs DM-KG'}")
        print(f"  {'─'*56}")

        # Find DM-KG for speedup reference
        dm_entry = next((e for e in entries if e["model"] == "DM_KG_Model"), None)
        dm_fwd = dm_entry.get("fwd_ms", 0) if dm_entry else 0
        dm_mrr = dm_entry.get("mrr", 0) if dm_entry else 0

        for r in entries:
            name = model_names.get(r["model"], r["model"])
            mrr = r.get("mrr", float("nan"))
            h10 = r.get("hits@10", float("nan"))
            fwd = r.get("fwd_ms", float("nan"))

            # Speed vs DM-KG
            if dm_fwd > 0 and not math.isnan(fwd):
                ratio = fwd / dm_fwd
                if ratio < 1:
                    speed_str = f"{dm_fwd/fwd:.1f}× faster"
                elif ratio > 1:
                    speed_str = f"{ratio:.1f}× slower"
                else:
                    speed_str = "same"
                fwd_str = f"{fwd:.1f}ms"
            else:
                speed_str = "—"
                fwd_str = "?"

            # Accuracy vs DM-KG
            if dm_mrr > 0 and not math.isnan(mrr):
                acc_str = f"{mrr:.4f}"
            else:
                acc_str = f"{mrr:.4f}" if not math.isnan(mrr) else "?"

            print(f"  {name:<18} {acc_str:>8} {h10:>8.4f}  {fwd_str:>6} {speed_str:>12}")

    # Also list JSON files on disk
    json_files = list(Path(checkpoint_dir).rglob("results.json"))
    if json_files:
        print(f"\nResult files on disk ({len(json_files)}):")
        for f in sorted(json_files):
            print(f"  {f}")


if __name__ == "__main__":
    main()
