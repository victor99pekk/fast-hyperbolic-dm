.PHONY: help test quick quick-all train-dm train-euclidean train-hyperbolic train-all benchmark clean

CONFIG := configs/fb15k237.yaml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Tests ──────────────────────────────────────────────
test:  ## Run unit tests
	uv run pytest tests/ -v

# ── Quick smoke tests (CPU, ~30 sec each) ───────────────
quick-dm:  ## Quick test DM-KG (FB15k-237)
	uv run python scripts/train.py --model dm --config $(CONFIG) --quick

quick-euclidean:  ## Quick test Euclidean baseline (FB15k-237)
	uv run python scripts/train.py --model euclidean --config $(CONFIG) --quick

quick-hyperbolic:  ## Quick test Hyperbolic baseline (FB15k-237)
	uv run python scripts/train.py --model hyperbolic --config $(CONFIG) --quick

quick-all: quick-dm quick-euclidean quick-hyperbolic  ## Quick test all models (FB15k-237)

quick-wn18rr:  ## Quick test DM-KG on WN18RR
	uv run python scripts/train.py --model dm --config configs/wn18rr.yaml --quick --dataset wn18rr

quick-all-wn18rr:  ## Quick test all models on WN18RR
	uv run python scripts/train.py --model euclidean --config configs/wn18rr.yaml --quick --dataset wn18rr
	uv run python scripts/train.py --model dm --config configs/wn18rr.yaml --quick --dataset wn18rr
	uv run python scripts/train.py --model hyperbolic --config configs/wn18rr.yaml --quick --dataset wn18rr

# ── Medium diagnostic runs (CPU, 20K triples, dim=128) ──
medium-dm:  ## Medium diagnostic DM-KG (FB15k-237)
	uv run python scripts/train.py --model dm --config $(CONFIG) --medium

medium-wn18rr:  ## Medium diagnostic DM-KG (WN18RR)
	uv run python scripts/train.py --model dm --config configs/wn18rr.yaml --dataset wn18rr --medium

medium-all-wn18rr:  ## Medium diagnostic all models (WN18RR)
	uv run python scripts/train.py --model euclidean --config configs/wn18rr.yaml --dataset wn18rr --medium
	uv run python scripts/train.py --model dm --config configs/wn18rr.yaml --dataset wn18rr --medium
	uv run python scripts/train.py --model hyperbolic --config configs/wn18rr.yaml --dataset wn18rr --medium

# ── Full training (auto-detects GPU) ───────────────────
train-dm:  ## Train DM-KG (full)
	uv run python scripts/train.py --model dm --config $(CONFIG)

train-euclidean:  ## Train Euclidean TransE (full)
	uv run python scripts/train.py --model euclidean --config $(CONFIG)

train-hyperbolic:  ## Train Hyperbolic TransE (full)
	uv run python scripts/train.py --model hyperbolic --config $(CONFIG)

train-all: train-euclidean train-dm train-hyperbolic  ## Train all models sequentially (FB15k-237)

train-dm-wn18rr:  ## Train DM-KG on WN18RR (full)
	uv run python scripts/train.py --model dm --config configs/wn18rr.yaml --dataset wn18rr

train-all-wn18rr:  ## Train all models on WN18RR
	uv run python scripts/train.py --model euclidean --config configs/wn18rr.yaml --dataset wn18rr
	uv run python scripts/train.py --model dm --config configs/wn18rr.yaml --dataset wn18rr
	uv run python scripts/train.py --model hyperbolic --config configs/wn18rr.yaml --dataset wn18rr

# ── Benchmark ───────────────────────────────────────────
benchmark:  ## Compare speed/memory across all models
	uv run python scripts/benchmark.py --config $(CONFIG)

# ── Training on GPU (explicit) ─────────────────────────
train-dm-gpu:  ## Train DM-KG on GPU explicitly
	uv run python scripts/train.py --model dm --config $(CONFIG) --device cuda

train-all-gpu:  ## Train all models on GPU sequentially
	uv run python scripts/train.py --model euclidean --config $(CONFIG) --device cuda
	uv run python scripts/train.py --model dm --config $(CONFIG) --device cuda
	uv run python scripts/train.py --model hyperbolic --config $(CONFIG) --device cuda

# ── Dataset ─────────────────────────────────────────────
data:  ## Download FB15k-237 dataset
	uv run python -c "from src.data.dataset import load_fb15k237; load_fb15k237('data/fb15k237')"

# ── Results ─────────────────────────────────────────────
results:  ## Show training results from TensorBoard logs
	uv run python scripts/results.py

speedup:  ## Show speedup ratios (DM-KG vs baselines)
	uv run python scripts/speedup.py

# ── Cleanup ─────────────────────────────────────────────
clean:  ## Remove logs, checkpoints, and cache
	rm -rf logs/ checkpoints/ .pytest_cache/ __pycache__ src/**/__pycache__

clean-all: clean  ## Also remove dataset and venv
	rm -rf data/ .venv/

# ── Setup ───────────────────────────────────────────────
setup:  ## Install dependencies (auto-detects CPU/GPU)
	uv sync --group dev

notebook:  ## Launch Jupyter notebook
	uv run jupyter notebook notebooks/analysis.ipynb
