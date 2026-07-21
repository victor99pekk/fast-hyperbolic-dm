"""
Unit tests for the DM-KG project.

Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.utils.hyperbolic import (
    log_map_zero,
    exp_map_zero,
    mobius_add,
    hyperbolic_distance,
    poincare_clip,
)
from src.layers import DMRelationalLayer
from src.models import DM_KG_Model, EuclideanTransE, HyperbolicTransE
from src.data.negative_sampling import NegativeSampler
from src.training.losses import hinge_loss
from src.utils.metrics import compute_mrr, compute_hits_at_k


# ============================================================
# Hyperbolic Math Tests
# ============================================================

class TestHyperbolicMath:
    """Test hyperbolic geometry utilities."""

    def test_log_exp_inverse_near_origin(self):
        """log_map_zero and exp_map_zero should be inverses near origin."""
        x = torch.randn(10, 5) * 0.3
        x_ball = poincare_clip(x)
        v = log_map_zero(x_ball)
        x_recovered = exp_map_zero(v)
        assert (x_ball - x_recovered).abs().max() < 1e-3

    def test_mobius_add_stays_in_ball(self):
        """Möbius addition should keep results inside the ball."""
        a = poincare_clip(torch.randn(5, 10) * 0.5)
        b = poincare_clip(torch.randn(5, 10) * 0.5)
        result = mobius_add(a, b)
        assert result.norm(dim=-1).max() < 1.0

    def test_distance_symmetry(self):
        """Hyperbolic distance should be symmetric."""
        a = poincare_clip(torch.randn(3, 8) * 0.5)
        b = poincare_clip(torch.randn(3, 8) * 0.5)
        d1 = hyperbolic_distance(a, b)
        d2 = hyperbolic_distance(b, a)
        assert (d1 - d2).abs().max() < 1e-2

    def test_self_distance(self):
        """Distance to self should be (near) zero."""
        a = poincare_clip(torch.randn(5, 6) * 0.5)
        d = hyperbolic_distance(a, a)
        assert d.max() < 1e-3

    def test_clip_enforces_boundary(self):
        """poincare_clip should bring out-of-ball points back inside."""
        big = torch.tensor([[3.0, 0.0], [5.0, 0.0], [0.99, 0.0]])
        clipped = poincare_clip(big)
        assert clipped.norm(dim=-1).max() < 1.0
        # The third vector was already in-bounds
        assert torch.allclose(clipped[2], big[2])


# ============================================================
# Layer Tests
# ============================================================

class TestDMRelationalLayer:
    """Test the DM relational projection layer."""

    def test_projection_shapes(self):
        """Projection output should have correct shape."""
        num_r, high_d, low_d = 10, 64, 16
        layer = DMRelationalLayer(num_r, high_dim=high_d, low_dim=low_d)

        x = torch.randn(32, high_d)
        r_ids = torch.randint(0, num_r, (32,))
        out = layer.project_to_slice(x, r_ids)

        assert out.shape == (32, low_d)

    def test_projection_no_nan(self):
        """Projection should not produce NaN."""
        num_r, high_d, low_d = 5, 32, 8
        layer = DMRelationalLayer(num_r, high_dim=high_d, low_dim=low_d)

        x = torch.randn(16, high_d)
        r_ids = torch.randint(0, num_r, (16,))
        out = layer.project_to_slice(x, r_ids)

        assert not torch.isnan(out).any()

    def test_fixed_projections(self):
        """Non-learnable projections should keep values unchanged after init."""
        num_r, high_d, low_d = 3, 16, 4
        layer = DMRelationalLayer(
            num_r, high_dim=high_d, low_dim=low_d, learnable=False
        )
        orig = layer.relation_projections.clone()

        x = torch.randn(8, high_d)
        r_ids = torch.randint(0, num_r, (8,))
        _ = layer.project_to_slice(x, r_ids)

        # Fixed projections should not have changed
        assert torch.equal(layer.relation_projections, orig)


# ============================================================
# Model Tests
# ============================================================

class TestModels:
    """Test all three KG models."""

    NUM_E, NUM_R, B = 100, 8, 32
    HIGH_D, LOW_D = 64, 16

    @pytest.fixture
    def batch(self):
        s = torch.randint(0, self.NUM_E, (self.B,))
        r = torch.randint(0, self.NUM_R, (self.B,))
        o = torch.randint(0, self.NUM_E, (self.B,))
        return s, r, o

    def test_dm_model_forward(self, batch):
        """DM-KG model forward pass should produce correct shape and no NaN."""
        model = DM_KG_Model(
            self.NUM_E, self.NUM_R, high_dim=self.HIGH_D, low_dim=self.LOW_D
        )
        scores = model(*batch)
        assert scores.shape == (self.B,)
        assert not torch.isnan(scores).any()

    def test_dm_model_gradient_flow(self, batch):
        """DM-KG model gradients should flow to entity embeddings."""
        model = DM_KG_Model(
            self.NUM_E, self.NUM_R, high_dim=self.HIGH_D, low_dim=self.LOW_D
        )
        scores = model(*batch)
        loss = scores.mean()
        loss.backward()
        assert model.entity_embeddings.grad is not None
        assert not torch.isnan(model.entity_embeddings.grad).any()

    def test_euclidean_model_forward(self, batch):
        """Euclidean TransE forward should work."""
        model = EuclideanTransE(self.NUM_E, self.NUM_R, dim=self.HIGH_D)
        scores = model(*batch)
        assert scores.shape == (self.B,)
        assert not torch.isnan(scores).any()

    def test_hyperbolic_model_forward(self, batch):
        """Hyperbolic TransE forward should work and produce no NaN."""
        model = HyperbolicTransE(self.NUM_E, self.NUM_R, dim=self.HIGH_D)
        scores = model(*batch)
        assert scores.shape == (self.B,)
        assert not torch.isnan(scores).any()

    def test_hyperbolic_entity_in_ball(self, batch):
        """Entity embeddings should stay in the Poincaré ball."""
        model = HyperbolicTransE(self.NUM_E, self.NUM_R, dim=16)
        emb = model._get_entity(batch[0])
        assert emb.norm(dim=-1).max() < 1.0


# ============================================================
# Data & Sampling Tests
# ============================================================

class TestNegativeSampling:
    """Test negative sampling."""

    def test_correct_num_negatives(self):
        """Should generate batch_size * num_neg negatives."""
        sampler = NegativeSampler(1000, seed=42)
        s = torch.randint(0, 1000, (50,))
        r = torch.randint(0, 10, (50,))
        o = torch.randint(0, 1000, (50,))
        neg_s, neg_r, neg_o = sampler.sample(s, r, o, num_neg=4, filtered=False)
        assert neg_s.shape == (200,)
        assert neg_r.shape == (200,)
        assert neg_o.shape == (200,)

    def test_relations_unchanged(self):
        """Relations should not be corrupted in negative sampling."""
        sampler = NegativeSampler(1000, seed=42)
        s = torch.randint(0, 1000, (20,))
        r = torch.randint(0, 10, (20,))
        o = torch.randint(0, 1000, (20,))
        _, neg_r, _ = sampler.sample(s, r, o, num_neg=3, filtered=False)
        expected_r = r.repeat_interleave(3)
        assert torch.equal(neg_r, expected_r)


# ============================================================
# Loss Tests
# ============================================================

class TestLosses:
    """Test loss functions."""

    def test_hinge_loss_nonnegative(self):
        """Hinge loss should be non-negative."""
        pos = torch.rand(32)
        neg = torch.rand(32 * 10)
        loss = hinge_loss(pos, neg)
        assert loss.item() >= 0.0

    def test_hinge_loss_decreases_with_margin(self):
        """Higher margin should not decrease loss for same scores."""
        pos = torch.ones(8)
        neg = torch.ones(80) * 10  # negatives are far away (high distance)
        loss1 = hinge_loss(pos, neg, margin=1.0)
        loss2 = hinge_loss(pos, neg, margin=5.0)
        # With large negative scores, larger margin = larger loss
        assert loss2 >= loss1


# ============================================================
# Metrics Tests
# ============================================================

class TestMetrics:
    """Test evaluation metrics."""

    def test_mrr_perfect(self):
        """MRR should be 1.0 for perfect ranks."""
        ranks = [1.0, 1.0, 1.0]
        assert compute_mrr(ranks) == 1.0

    def test_mrr_zero(self):
        """MRR should approach 0 for very poor ranks."""
        ranks = [1000.0, 10000.0]
        mrr = compute_mrr(ranks)
        assert mrr < 0.01

    def test_hits_at_k(self):
        """Hits@K should count correctly."""
        ranks = [1.0, 3.0, 5.0, 10.0, 11.0]
        assert compute_hits_at_k(ranks, 1) == 0.2  # 1/5
        assert compute_hits_at_k(ranks, 3) == 0.4  # 2/5
        assert compute_hits_at_k(ranks, 10) == 0.8  # 4/5
