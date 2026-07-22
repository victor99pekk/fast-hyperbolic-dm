# fast-hyperbolic-dm

**Dvoretzky-Milman Projections for Multi-Relational Knowledge Graphs**

Master's thesis exploring whether relation-specific linear projections from hyperbolic
space into low-dimensional Euclidean slices can match hyperbolic accuracy at near-Euclidean speed.

## Quick Start

```bash
uv sync                  # install dependencies
make test                # run 20 unit tests
make quick-all           # smoke test all 3 models on FB15k-237 (~2 min)
make results             # show accuracy + speed per dataset
```

## Commands

```bash
# ── Testing ──
make test                # pytest -v (20 tests)

# ── Quick smoke tests (CPU, 100 epochs, 5K triples, dim=64) ──
make quick-dm            # DM-KG on FB15k-237
make quick-all           # All 3 models on FB15k-237
make quick-wn18rr        # DM-KG on WN18RR
make quick-all-wn18rr    # All 3 models on WN18RR

# ── Full training (auto-detects GPU, 200 epochs, full data, dim=256) ──
make train-dm            # DM-KG on FB15k-237
make train-all           # All 3 models on FB15k-237
make train-dm-wn18rr     # DM-KG on WN18RR
make train-all-wn18rr    # All 3 models on WN18RR

# ── Results ──
make results             # Per-dataset accuracy + speed table (DM-KG centered)
make speedup             # Pairwise speed comparisons
make benchmark           # CUDA timing + memory profiling

# ── Utilities ──
make data                # Pre-download all datasets
make clean               # Remove logs, checkpoints, cache
make help                # Show all commands
```

## Models

| Model | Description |
|-------|-------------|
| **DM-KG** | Hyperbolic embeddings → log-map → relation-specific DM projections (k=32) → Euclidean TransE scoring |
| **Euclidean TransE** | Standard TransE in ℝᵈ — fast, poor on hierarchical relations |
| **Hyperbolic TransE** | TransE with Möbius addition in Poincaré ball — accurate, slow |

## Datasets

| Dataset | Entities | Relations | Train | Notes |
|---------|----------|-----------|-------|-------|
| FB15k-237 | 14,541 | 237 | 272K | General KG, standard benchmark |
| WN18RR | 40,943 | 11 | 87K | WordNet taxonomy, highly hierarchical |

## Experiment Modes

### Quick mode (`--quick`)
- 100 epochs, 5K training triples, dim=64/16, CPU only
- Verifies pipeline integrity, catches bugs, tests gradient flow
- ~30 seconds per model on CPU

### Full mode (default)
- 200 epochs, full dataset, dim=256/32, auto-detects GPU
- Early stopping (patience=20, eval every 50 epochs)
- Generates `results.json` with accuracy + timing in `checkpoints/`

## Results Format

After training, `make results` prints per-dataset tables:

```
──────────────────────────────────────────────────────────
  WN18RR
──────────────────────────────────────────────────────────
  Model           MRR ↑  Hits@10   Speed     vs DM-KG
  ──────────────────────────────────────────────────
  DM-KG (ours)    0.xxx   0.xxx   x.xms          same
  Euclidean       0.xxx   0.xxx   x.xms    X.X× faster
  Hyperbolic      0.xxx   0.xxx   x.xms    X.X× slower
```

Files saved:
- `checkpoints/{model}_{dataset}/results.json` — MRR, Hits@K, timing
- `checkpoints/{model}_{dataset}/best_model.pt` — model weights
- `logs/{model}_{dataset}/` — TensorBoard event files

## Project Structure

```
src/
├── utils/          # hyperbolic.py (log/exp maps, Möbius ops) + metrics.py (MRR, Hits@K)
├── layers/         # DMRelationalLayer — relation-specific Pᵣ projection matrices
├── models/         # dm_kg_model.py, baseline_euclidean.py, baseline_hyperbolic.py
├── data/           # dataset.py (FB15k-237 + WN18RR), negative_sampling.py
├── training/       # trainer.py (KGTrainer with early stopping, timing), losses.py
└── benchmarking/   # profiler.py (CUDA events, memory tracking)
scripts/
├── train.py        # Main entry point (--model, --dataset, --quick, --device)
├── benchmark.py    # Multi-model speed/memory comparison
├── results.py      # Parse results.json and display tables
└── speedup.py      # Pairwise speedup ratios
configs/
├── fb15k237.yaml   # FB15k-237 hyperparameters
└── wn18rr.yaml     # WN18RR hyperparameters
tests/
└── test_all.py     # 20 unit tests (hyperbolic math, layers, models, losses, metrics)
```
