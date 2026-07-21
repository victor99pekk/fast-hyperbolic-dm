"""
Hyperbolic geometry utilities for the Poincaré ball model.

All operations use curvature c (default c=1.0 corresponds to the unit
Poincaré ball). The ball is defined as D^d = {x in R^d : ||x||^2 < 1/c}.
With c=1.0, this is the standard unit ball.
"""

import torch
import torch.nn.functional as F


def poincare_clip(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Clip embeddings to ensure they stay strictly inside the Poincaré ball.

    Maps any vector with norm >= 1/sqrt(c) back to a point with norm
    (1/sqrt(c) - eps), preserving the direction.

    Args:
        x: Tensor of shape (..., d) in R^d.
        c: Curvature constant (default 1.0).
        eps: Small margin from the boundary.

    Returns:
        Clipped tensor of same shape, guaranteed to have norm < 1/sqrt(c).
    """
    max_norm = (1.0 / (c ** 0.5)) - eps
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    # Only clip vectors whose norm exceeds the safe threshold
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def _atanh(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable arctanh: 0.5 * log((1+x)/(1-x))."""
    x = torch.clamp(x, min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def log_map_zero(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Logarithmic map from the Poincaré ball to the tangent space at the origin.

    Maps points on the Poincaré ball D^d to vectors in T_0 D^d ≅ R^d.

    Formula:
        log_0(x) = arctanh(sqrt(c) * ||x||) / (sqrt(c) * ||x||) * x

    Args:
        x: Tensor of shape (..., d), points in the Poincaré ball.
        c: Curvature constant (default 1.0).
        eps: Small constant to prevent division by zero.

    Returns:
        Tensor of shape (..., d) in the tangent (Euclidean) space.
    """
    sqrt_c = c ** 0.5
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    # Compute arctanh(sqrt(c) * norm) / (sqrt(c) * norm)
    scale = _atanh(torch.clamp(sqrt_c * norm, max=1.0 - eps)) / (sqrt_c * norm)
    return scale * x


def exp_map_zero(v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Exponential map from the tangent space at the origin to the Poincaré ball.

    Maps vectors in T_0 D^d ≅ R^d to points on the Poincaré ball.

    Formula:
        exp_0(v) = tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v

    Args:
        v: Tensor of shape (..., d), tangent vectors at the origin.
        c: Curvature constant (default 1.0).
        eps: Small constant to prevent division by zero.

    Returns:
        Tensor of shape (..., d), points in the Poincaré ball.
    """
    sqrt_c = c ** 0.5
    norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    # Compute tanh(sqrt(c) * norm) / (sqrt(c) * norm)
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
    return scale * v


def mobius_add(
    x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball.

    Formula:
        x ⊕_c y = ((1 + 2c<x,y> + c||y||^2) * x + (1 - c||x||^2) * y)
                  / (1 + 2c<x,y> + c^2||x||^2||y||^2)

    This is the hyperbolic analogue of Euclidean vector addition. It ensures
    the result stays inside the Poincaré ball.

    Args:
        x: Tensor of shape (..., d).
        y: Tensor of shape (..., d).
        c: Curvature constant (default 1.0).
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (..., d), the Möbius sum of x and y.
    """
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

    numerator = (1.0 + 2.0 * c * xy_inner + c * y_norm_sq) * x + (
        1.0 - c * x_norm_sq
    ) * y
    denominator = 1.0 + 2.0 * c * xy_inner + (c ** 2) * x_norm_sq * y_norm_sq

    # Numerical safety: denominator should be strictly positive (≥ 0),
    # and can be near zero when x ≈ -y with both near the boundary.
    denominator = denominator.clamp(min=eps)

    result = numerator / denominator

    # When x ≈ -y, the result should be close to zero, but floating-point
    # error in the numerator can produce spurious values. Return zero
    # when the Mobius addition should theoretically yield ~0.
    near_zero_mask = (numerator.norm(p=2, dim=-1, keepdim=True) < eps)
    result = torch.where(near_zero_mask, torch.zeros_like(result), result)

    # Safety: clip to ensure we stay strictly inside the ball
    result = poincare_clip(result, c=c)

    return result


def hyperbolic_distance(
    x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """
    Compute the Poincaré (hyperbolic) distance between two points.

    Formula:
        d_H(x, y) = (2 / sqrt(c)) * arctanh(sqrt(c) * ||-x ⊕_c y||)

    Args:
        x: Tensor of shape (..., d).
        y: Tensor of shape (..., d).
        c: Curvature constant (default 1.0).
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (...,), hyperbolic distances.
    """
    sqrt_c = c ** 0.5
    # Möbius addition of -x and y
    neg_x = -x
    mobius_result = mobius_add(neg_x, y, c=c)
    norm = torch.norm(mobius_result, p=2, dim=-1)
    norm = torch.clamp(norm, min=eps, max=1.0 / sqrt_c - eps)
    return (2.0 / sqrt_c) * _atanh(sqrt_c * norm)


def mobius_matrix_multiply(
    W: torch.Tensor, x: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """
    Möbius matrix-vector multiplication: W ⊗_c x.

    Maps a point x through a linear transformation W while staying in the ball.

    Formula:
        W ⊗_c x = (1/sqrt(c)) * tanh(
            (||Wx|| / ||x||) * arctanh(sqrt(c) * ||x||)
        ) * (Wx / ||Wx||)

    Args:
        W: Weight matrix of shape (out_dim, in_dim) or (batch, out_dim, in_dim).
        x: Point in the Poincaré ball, shape (..., in_dim).
        c: Curvature constant.
        eps: Small constant.

    Returns:
        Transformed point in the Poincaré ball, shape (..., out_dim).
    """
    sqrt_c = c ** 0.5
    # Handle batched matrix multiply
    if W.dim() == 3:
        # (batch, out, in) × (batch, in, 1) -> (batch, out, 1)
        Wx = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
    else:
        Wx = F.linear(x, W)

    Wx_norm = torch.norm(Wx, p=2, dim=-1, keepdim=True)
    Wx_norm = torch.clamp(Wx_norm, min=eps)
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_norm = torch.clamp(x_norm, min=eps)

    # tanh( (||Wx||/||x||) * arctanh(sqrt_c * ||x||) )
    inner = (Wx_norm / x_norm) * _atanh(torch.clamp(sqrt_c * x_norm, max=1.0 - eps))
    scale = torch.tanh(inner) / (sqrt_c * Wx_norm)

    return scale * Wx
