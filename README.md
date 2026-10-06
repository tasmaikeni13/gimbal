# Gimbal

**Gimbal** is a matrix-preconditioned optimizer for neural networks. Like SOAP, it runs Adam in a
rotated frame `(Q_L, Q_R)` of each weight matrix. Unlike SOAP, it chooses that frame with the same
statistical model that Adam's diagonal assumes, so the frame and the eigenvalues are estimated
consistently. That maximum-likelihood frame is a joint-diagonalization (common principal
components) problem. Gimbal follows it online with one natural-gradient step on the orthogonal
groups per iteration. The variances that weight that step are shrunk toward their separable
(Kronecker) fit by an empirical-Bayes factor measured from the data, so the estimator behaves like
KL-Shampoo's on separable gradients and like the free maximum-likelihood estimator on
non-separable ones. The step uses only matrix multiplications: after a 50-step warm start there
are no Kronecker factor buffers and no QR or eigendecomposition.

## Why

SOAP (Vyas et al., 2024) runs Adam in the eigenbasis of Shampoo's Kronecker factors `E[GGᵀ]` and
`E[GᵀG]`. Those factors are the right statistics for a Kronecker-*product* covariance, but SOAP's own
preconditioner has a *free* diagonal of eigenvalues. When the gradient variances in the rotated frame
are not a product (the situation in which SOAP beats Shampoo), pooled factors are a statistically
inefficient way to find the frame, and they can fail to identify it at all. The theory in
`theory/gimbal_theory.md` makes this precise:

* every frame estimator built from a single weighted factor (SOAP, KL-SOAP, …) has variance at or
  above the Cramér–Rao bound, strictly above on non-separable arrays (machine-checked in Lean);
* the likelihood identifies a rotation whenever two variance *profiles* differ, while pooled
  factors need their *sums* to differ;
* Gimbal's flow solves the likelihood equation, is equivariant and scale-invariant, keeps its frames
  orthogonal, and its bias-corrected rotation schedule turns the frame estimate into a
  bias-corrected exponential average (Theorem 4.4, machine-checked).

`theory.md` explains the ideas and equations; `theory/gimbal_theory.md` has statements, proofs and
evidence labels; `formal/` has 52 Lean 4 + Mathlib theorems.

## Status

The project runs as ten phases (`phases/`), each with predeclared gates and a self-correcting,
dependency-aware protocol (`phases/README.md`). Current state: `phases/STATUS.md`.

| Phase | Content | State |
|---|---|---|
| 01 | Formal theory of Gimbal and its peers; Lean formalization | done |
| 02 | Formal, mathematical, numerical, statistical and Monte Carlo analysis against every peer (no model training) | see `phases/STATUS.md` |
| 03–08 | Reference/JAX implementations and small-LM validation, TPU v4-32 kernels, FineWeb-Edu pipeline, fair tuning, 125M × 2.5B-token runs (2 seeds per optimizer), analysis | planned |
| 09–10 | Paper; release | planned |

Results so far are generated into `experiments/phase1/results/` and `experiments/phase2/results/`;
`experiments/phase2/report.md` collects them with the gate verdicts. No language-model result exists
yet; nothing here claims one.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
```

```python
import torch
from gimbal.torch import build

model = ...  # any model with 2-D hidden weight matrices
hidden = [p for p in model.parameters() if p.ndim == 2]   # route embeddings/heads/norms to AdamW
other = [p for p in model.parameters() if p.ndim != 2]
opt = build("gimbal", hidden, other, matrix_kwargs={"lr": 3e-3}, other_kwargs={"lr": 3e-3})
```

Gimbal's knobs beyond AdamW's: `rot_rate` (frame memory, default 0.02), `damping` (default 0.003)
and `frame_every` (the frame moves every k steps with the mean score of those steps, default 4). The variance shrinkage
(`flow_shrink`) and the warm start (`init="pooled"`, `warm_start_steps=50`) have no tuning knobs.

## Repository

| Path | Contents |
|---|---|
| `src/gimbal/torch/` | Gimbal and reference implementations of SOAP (incl. real-time), KL-SOAP, Muon, NorMuon, SPlus, ARO |
| `theory/`, `theory.md` | theory |
| `formal/` | Lean proofs |
| `experiments/` | Phase 1–2 experiments and their raw results |
| `research/` | literature frontier, hypotheses, ledgers |
| `phases/` | the research protocol and phase files |

Agents working on this repository should read `AGENTS.md`.
