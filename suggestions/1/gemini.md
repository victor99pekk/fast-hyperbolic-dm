# Diagnostic Summary & Action Plan: DM-KG Benchmark Analysis

## 1. Empirical Results Overview (WN18RR Benchmark)

| Model | MRR | Hits@10 | Epoch Time (s) | Forward Pass (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **EuclideanTransE** | **0.1593** | 0.1953 | 3.86 | **0.07** |
| **HyperbolicTransE** | 0.1571 | **0.2065** | **0.58** | 0.36 |
| **DM_KG_Model (Ours)** | 0.0328 | 0.1196 | 2.16 | 3.21 |

### Key Takeaways
1. **Accuracy Deficit:** `DM_KG_Model` MRR ($0.0328$) underperforms baseline models ($\approx 0.158$).
2. **Speed Deficit:** `DM_KG_Model` forward pass ($3.21\text{ ms}$) runs **$\sim 9\times$ slower** than `HyperbolicTransE` ($0.36\text{ ms}$) and **$\sim 46\times$ slower** than `EuclideanTransE` ($0.07\text{ ms}$).

---

## 2. Root Cause Analysis

### A. Computational Bottlenecks (Why Speed Lagged)
* **`torch.bmm` & Dynamic Memory Gathering:** Assembling dynamic 3D tensors $(B, k, d)$ on every batch via `P_r = self.relation_projections[relation_ids]` creates severe GPU memory bus bottlenecks.
* **Underutilized SIMD Kernels:** Batched matrix multiplication across small 3D slices is significantly less optimized on CUDA/CPU hardware than continuous 2D matrix multiplications (`torch.matmul`).
* **Pipeline Math Stacking:** Executing hyperbolic $\to$ tangent mapping ($\log_{\mathbf{0}}$) *plus* dynamic projection ($P_r$) *plus* TransE distance scoring introduces more FLOPs than a single closed-form Poincaré distance function.

### B. Mathematical Bottlenecks (Why Accuracy Dropped)
* **Boundary Tangent Distortion:** The standard logarithmic map $\log_{\mathbf{0}}(x)$ distorts distances exponentially as points approach the boundary of the Poincaré ball ($\Vert{}x\Vert{} \to 1$).
* **Matrix Rank Collapse:** Unconstrained optimization of $P_r$ via Adam leads to linearly dependent rows, squashing high-dimensional topology into a collapsed subspace that loses relational information.

---

## 3. Actionable Code Fixes

### Fix 1: Enforce Orthogonality on Projection Matrices $P_r$
Prevent rank collapse by constraining $P_r$ to the Stiefel manifold ($P_r P_r^T = I$), preserving distance ratios during lower-dimensional projection.

```python
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization

# Define orthogonal parametrization or use PyTorch's built-in
class OrthogonalProjection(nn.Module):
    def forward(self, X):
        # Enforces semi-orthogonality via QR decomposition or SVD
        q, _ = torch.linalg.qr(X.transpose(-1, -2))
        return q.transpose(-1, -2)

# In DMRelationalLayer initialization:
register_parametrization(self, "relation_projections", OrthogonalProjection())