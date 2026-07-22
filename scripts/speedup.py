#!/usr/bin/env python3
"""Compute speedup ratios between models from results.json files."""
import sys, os, json
from pathlib import Path


def load_results(base_dir: str) -> dict:
    """Load all results.json files, keyed by directory name."""
    results = {}
    for path in Path(base_dir).rglob("results.json"):
        tag = path.parent.name  # e.g. "dm_wn18rr"
        with open(path) as f:
            data = json.load(f)
        data["_tag"] = tag
        results[tag] = data
    return results


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    base = os.path.abspath(base)

    results = load_results(base)
    if not results:
        print("No results found.")
        sys.exit(1)

    # Find DM-KG per dataset and compare baselines to it
    for tag, r in sorted(results.items()):
        if "dm" not in tag.lower():
            continue
        dataset = tag.split("_", 1)[-1] if "_" in tag else tag
        dm = r
        dm_fwd = dm.get("timing", {}).get("forward_pass_ms", 0)
        dm_epoch = dm.get("timing", {}).get("avg_epoch_time_s", 0)

        if dm_fwd <= 0:
            continue

        for other_tag, other_r in results.items():
            if other_tag == tag:
                continue
            other_ds = other_tag.split("_", 1)[-1] if "_" in other_tag else other_tag
            if other_ds != dataset:
                continue

            other_fwd = other_r.get("timing", {}).get("forward_pass_ms", 0)
            other_epoch = other_r.get("timing", {}).get("avg_epoch_time_s", 0)

            if other_fwd > 0:
                if other_fwd < dm_fwd:
                    print(f"{other_r['model']} vs DM-KG ({dataset}): "
                          f"{dm_fwd/other_fwd:.1f}× faster than DM-KG "
                          f"({other_fwd:.1f}ms vs {dm_fwd:.1f}ms)")
                else:
                    print(f"{other_r['model']} vs DM-KG ({dataset}): "
                          f"{other_fwd/dm_fwd:.1f}× slower than DM-KG "
                          f"({other_fwd:.1f}ms vs {dm_fwd:.1f}ms)")

            if other_epoch > 0 and dm_epoch > 0:
                if other_epoch < dm_epoch:
                    print(f"  Epoch:        {dm_epoch/other_epoch:.1f}× faster than DM-KG "
                          f"({other_epoch:.1f}s vs {dm_epoch:.1f}s)")
                else:
                    print(f"  Epoch:        {other_epoch/dm_epoch:.1f}× slower than DM-KG "
                          f"({other_epoch:.1f}s vs {dm_epoch:.1f}s)")


if __name__ == "__main__":
    main()
