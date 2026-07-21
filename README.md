# fast-hyperbolic-dm

**Dvoretzky-Milman Projections for Multi-Relational Knowledge Graphs**

Master's thesis project exploring relation-specific Dvoretzky-Milman projections
to accelerate hyperbolic Knowledge Graph embeddings.

## Quick Start

```bash
uv sync
uv run pytest tests/ -v
uv run python scripts/train.py --model dm --config configs/fb15k237.yaml
```

## Models

| Model | Description |
|-------|-------------|
| `DM_KG_Model` | DM projections from hyperbolic → relation-specific Euclidean slices |
| `EuclideanTransE` | Standard Euclidean TransE baseline (fast) |
| `HyperbolicTransE` | Hyperbolic TransE with Möbius operations (accurate) |

## Project Structure

```
src/
├── utils/          # Hyperbolic math + evaluation metrics
├── layers/         # DMRelationalLayer
├── models/         # DM-KG, Euclidean, Hyperbolic models
├── data/           # FB15k-237 loader + negative sampling
├── training/       # KGTrainer + loss functions
└── benchmarking/   # CUDA timing + memory profiling
```
