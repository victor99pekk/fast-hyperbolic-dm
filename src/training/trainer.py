"""
KGTrainer: Training loop for Knowledge Graph link prediction models.

Handles training, validation, TensorBoard logging, checkpointing,
and early stopping for any model with a score_triples(s, r, o) interface.
"""

import os
import time
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from .losses import hinge_loss
from ..data.negative_sampling import NegativeSampler
from ..utils.metrics import evaluate_model


class KGTrainer:
    """
    Trainer for Knowledge Graph link prediction models.

    Args:
        model: Model with score_triples(s, r, o) method.
        train_triples: (N_train, 3) tensor of training (s, r, o).
        valid_triples: (N_valid, 3) tensor for validation.
        all_true_triples: Set of all known true (s, r, o) for filtered eval.
        num_entities: Total number of entities.
        config: Training hyperparameters dict.
        device: Torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        train_triples: torch.LongTensor,
        valid_triples: torch.LongTensor,
        all_true_triples: Set[Tuple[int, int, int]],
        num_entities: int,
        config: Dict,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.all_true_triples = all_true_triples
        self.num_entities = num_entities
        self.config = config
        self.device = device
        self.model_name = type(model).__name__

        # Extract config values with defaults
        self.batch_size = config.get("batch_size", 1024)
        self.num_neg = config.get("num_neg", 50)
        self.lr = config.get("lr", 0.001)
        self.margin = config.get("margin", 1.0)
        self.epochs = config.get("epochs", 200)
        self.eval_every = config.get("eval_every", 10)
        self.patience = config.get("patience", 20)
        self.log_dir = config.get("log_dir", "logs")
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.grad_clip = config.get("grad_clip", 1.0)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr
        )

        # Negative sampler
        self.neg_sampler = NegativeSampler(
            num_entities=num_entities,
            all_true_triples=all_true_triples,
        )

        # TensorBoard writer (optional)
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.best_mrr = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_metrics = {}

        # Timing state
        self.total_train_time_s = 0.0
        self.epoch_times_s = []
        self.avg_forward_time_ms = 0.0
        self._inference_time_ms = 0.0

    def train_epoch(self) -> float:
        """Train one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Create DataLoader for this epoch (reshuffled)
        dataset = TensorDataset(
            self.train_triples[:, 0],
            self.train_triples[:, 1],
            self.train_triples[:, 2],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        epoch_start = time.time()

        for subjects, relations, objects in pbar:
            subjects = subjects.to(self.device)
            relations = relations.to(self.device)
            objects = objects.to(self.device)

            # Generate negative samples
            neg_s, neg_r, neg_o = self.neg_sampler.sample(
                subjects.cpu(),
                relations.cpu(),
                objects.cpu(),
                num_neg=self.num_neg,
                filtered=True,
            )
            neg_s = neg_s.to(self.device)
            neg_r = neg_r.to(self.device)
            neg_o = neg_o.to(self.device)

            # Forward pass: score positive and negative triples
            pos_scores = self.model.score_triples(
                subjects, relations, objects
            )
            neg_scores = self.model.score_triples(neg_s, neg_r, neg_o)

            # Compute loss
            loss = hinge_loss(pos_scores, neg_scores, margin=self.margin)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        # Accumulate timing
        self.total_train_time_s += epoch_time
        self.epoch_times_s.append(epoch_time)

        # Measure forward pass time on first epoch using CUDA-safe events
        if self.current_epoch == 1 and num_batches > 0:
            self._benchmark_forward_pass(subjects, relations, objects)

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("train/loss", avg_loss, self.current_epoch)
            self.writer.add_scalar(
                "train/epoch_time_sec", epoch_time, self.current_epoch
            )

        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set. Returns metrics dict."""
        return evaluate_model(
            model=self.model,
            eval_triples=self.valid_triples,
            all_true_triples=self.all_true_triples,
            num_entities=self.num_entities,
            batch_size=self.batch_size,
            device=self.device,
        )

    def fit(self) -> Dict[str, float]:
        """
        Run full training loop with validation and early stopping.

        Returns:
            Dict with best validation metrics.
        """
        print(f"Starting training for {self.epochs} epochs...")
        print(
            f"  Model: {type(self.model).__name__}, "
            f"Device: {self.device}, "
            f"Params: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # Train
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

            # Validate
            if epoch % self.eval_every == 0 or epoch == self.epochs:
                metrics = self.evaluate()
                mrr = metrics["mrr"]

                # Log metrics
                if self.writer is not None:
                    for key, value in metrics.items():
                        self.writer.add_scalar(
                            f"valid/{key}", value, self.current_epoch
                        )

                print(
                    f"  Valid MRR: {mrr:.4f} | "
                    f"Hits@1: {metrics['hits@1']:.4f} | "
                    f"Hits@3: {metrics['hits@3']:.4f} | "
                    f"Hits@10: {metrics['hits@10']:.4f}"
                )

                # Checkpoint best model
                if mrr > self.best_mrr:
                    self.best_mrr = mrr
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint()
                    self.best_metrics = metrics
                    print(f"  -> New best MRR: {mrr:.4f} (saved)")
                else:
                    self.patience_counter += 1
                    print(
                        f"  No improvement ({self.patience_counter}/"
                        f"{self.patience})"
                    )

                # Early stopping
                if self.patience_counter >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best MRR: {self.best_mrr:.4f} at epoch "
                        f"{self.best_epoch})"
                    )
                    break

        if self.writer is not None:
            self.writer.close()

        # Load best model
        self._load_checkpoint()

        # Save results to JSON
        self._save_results()

        return {
            "best_mrr": self.best_mrr,
            "best_epoch": self.best_epoch,
        }

    def _benchmark_forward_pass(
        self,
        subjects: torch.Tensor,
        relations: torch.Tensor,
        objects: torch.Tensor,
        num_warmup: int = 3,
        num_repeats: int = 20,
    ) -> None:
        """
        Measure forward pass time using CUDA events (GPU) or time.perf_counter (CPU).
        Stores result in self.avg_forward_time_ms.
        """
        self.model.eval()

        s = subjects[:min(len(subjects), 256)]
        r = relations[:min(len(relations), 256)]
        o = objects[:min(len(objects), 256)]

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model.score_triples(s, r, o)

        # Measure
        if self.device.startswith("cuda"):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            timings = []
            with torch.no_grad():
                for _ in range(num_repeats):
                    starter.record()
                    _ = self.model.score_triples(s, r, o)
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender))
            self.avg_forward_time_ms = sum(timings) / len(timings)
        else:
            import time as _time
            timings = []
            with torch.no_grad():
                for _ in range(num_repeats):
                    t0 = _time.perf_counter()
                    _ = self.model.score_triples(s, r, o)
                    t1 = _time.perf_counter()
                    timings.append((t1 - t0) * 1000.0)
            self.avg_forward_time_ms = sum(timings) / len(timings)

        # Estimate inference time per 1K triples
        batch_size = s.shape[0]
        self._inference_time_ms = (self.avg_forward_time_ms / batch_size) * 1000

        self.model.train()

    def _save_results(self) -> None:
        """Save best training results to a JSON file."""
        import json
        from datetime import datetime

        # Compute timing stats
        avg_epoch_s = (
            sum(self.epoch_times_s) / len(self.epoch_times_s)
            if self.epoch_times_s else 0.0
        )

        results = {
            "model": self.model_name,
            "device": self.device,
            "num_params": sum(p.numel() for p in self.model.parameters()),
            "best_epoch": self.best_epoch,
            "best_mrr": self.best_mrr,
            "timestamp": datetime.now().isoformat(),
            "timing": {
                "total_train_time_s": round(self.total_train_time_s, 2),
                "avg_epoch_time_s": round(avg_epoch_s, 3),
                "num_epochs_timed": len(self.epoch_times_s),
                "forward_pass_ms": round(self.avg_forward_time_ms, 3),
                "inference_ms_per_1k": round(self._inference_time_ms, 3),
            },
        }
        if self.best_metrics:
            results["metrics"] = {
                k: float(v) for k, v in self.best_metrics.items()
            }

        path = os.path.join(self.checkpoint_dir, "results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {path}")

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_mrr": self.best_mrr,
            },
            path,
        )

    def _load_checkpoint(self) -> None:
        """Load best model checkpoint."""
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best model (epoch {checkpoint['epoch']}, "
                f"MRR: {checkpoint['best_mrr']:.4f})"
            )
