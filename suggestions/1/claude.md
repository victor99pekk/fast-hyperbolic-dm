# Suggestions — DM-Projected Hyperbolic KG Project
 
## 1. Literature check (novelty status)
 
| Idea | Status | Key related work | Notes |
|---|---|---|---|
| **DM projections for hyperbolic KGs** | Open, but has a close rival | RotL / Rot2L (EMNLP 2021 Findings) — Euclidean simplifications of RotH claiming the same speed/accuracy tradeoff | No prior work does learned relation-specific DM-style projections from hyperbolic space to Euclidean slices. Möbius addition sometimes empirically underperforming plain Euclidean addition (observed in prior hyperbolic KGE papers) is a good empirical hook — DM theory would explain it. |
| **Generic chaining (γ₂) for relational GNN generalization** | Cleanest open gap | γ₂/generic chaining used for shallow ReLU nets, general NN bounds; GNN generalization theory uses Rademacher complexity, VC-dimension, PAC-Bayes, graph-homomorphism/entropy, or NTK/mean-field | Nobody indexes a stochastic process over the graph's intrinsic path metric (shortest-path / effective resistance) and bounds it with γ₂. Most unclaimed of the three. |
| **Robust covariance truncation for heavy-tailed KG gradients** | Synthesis gap | Entrywise/singular-value truncation well developed in robust statistics (Fan et al.-style operators, robust gradient descent under heavy tails); KG tail-entity problem addressed via meta-learning, augmentation, reweighting (Tail-GNN, GEN, MorsE, KG-Mixup) | These two literatures don't currently talk to each other. Requires the most justification work since it's a synthesis rather than an obvious hole. |
 
**Baseline gap identified:** the current codebase only compares `EuclideanTransE` vs. `HyperbolicTransE` vs. `DM_KG_Model` (TransE family only). Missing:
1. **RotL/Rot2L** — non-negotiable, closest prior work to the exact claim being made.
2. **RotatE or ComplEx** — standard modern non-hyperbolic baselines (TransE is a 2013 model).
3. **A modern hyperbolic model** (RotH, AttH) instead of HyperbolicTransE — TransE's scoring function is known to be weak in hyperbolic space, which may be inflating the apparent advantage.
## 2. Current experimental results (wn18rr) — diagnosis
 
```
Model                 MRR  Hits@10  Epoch(s)  Fwd(ms)
EuclideanTransE    0.1593   0.1953      3.86     0.07
DM_KG_Model        0.0328   0.1196      2.16     3.21
HyperbolicTransE   0.1571   0.2065      0.58     0.36
```
 
**Verdict: both core claims currently fail.**
- **Speed:** DM_KG forward pass (3.21ms) is ~9× *slower* than HyperbolicTransE (0.36ms) and ~46× slower than EuclideanTransE (0.07ms) — the opposite of the intended result.
- **Accuracy:** MRR of 0.033 is ~5× worse than both baselines — not a modest tradeoff, likely a bug or mis-specified method.
**Likely culprits, ranked by suspicion:**
1. Projection step (log map + per-relation `P_r`) not batched/vectorized — looped per-triple would fully explain the 3ms forward pass.
2. Redundant round trips: mapping hyperbolic → tangent space → project → back to hyperbolic before scoring, instead of scoring directly in the Euclidean slice.
3. Undertrained/randomly-initialized `P_r` matrices — check loss curves, not just final metrics.
4. Numerical instability in the log map near the Poincaré ball boundary (‖x‖→1), which can simultaneously wreck accuracy and slow things down via clamping/stabilization code.
**Recommended next step:** fix the implementation before adding SOTA baselines (RotL, RotatE) — currently the method loses to plain 2013 TransE on its own turf, so there's no method yet worth benchmarking against stronger competitors.