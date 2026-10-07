# Gimbal

Gimbal is a matrix-preconditioned optimizer. Like SOAP, it runs Adam in a rotated frame
`(Q_L, Q_R)` of each weight matrix. It differs in how that frame is chosen. SOAP takes the
eigenvectors of Shampoo's Kronecker factors `E[GGᵀ]` and `E[GᵀG]`, which are statistics of a model
in which the gradient's variances form a Kronecker product. Adam's own diagonal assumes something
else: independent coordinates with a free array of variances. Gimbal estimates the frame under
that second model, the one the preconditioner actually uses. The maximum-likelihood frame of that
model is a joint-diagonalization problem, and Gimbal follows it online with one damped
natural-gradient step on the orthogonal groups per block of steps, using only matrix products
after a 50-step warm start.

This repository holds the whole research program: the theory (its central statements checked in
Lean 4), a Monte Carlo study on synthetic gradient streams, PyTorch and JAX implementations, a
small language-model study, a 125M-parameter / 2.5B-token comparison with AdamW and SOAP on a TPU
v4-32, the analysis and the paper.

## What we found

<!-- findings: written after the Phase 08 decision -->

## Results at 125M parameters

Llama-style decoder (123.6M parameters), FineWeb-Edu `sample-10BT`, 10,172 steps of 240 × 1,024
tokens (2.5B tokens), two seeds per optimizer, every optimizer tuned with the same budget (seven
learning rates and three values of one secondary knob, two seeds each, at a quarter of the
horizon). Losses are mean token cross-entropy on the 20M-token validation split; the test split
was read once, after the decision rule had been applied.

<!-- results:start -->
_The confirmatory runs have not been analysed yet._
<!-- results:end -->

## How it works

For a frame `U = (Q_L, Q_R)` write `Z = Q_Lᵀ G Q_R`. SOAP's preconditioner family is "Adam in the
frame `U`", i.e. a Gaussian with independent entries of `Z` and variances `D`. Minimizing the KL
divergence from the gradient distribution to that family over `D` leaves a function of the frame
alone,

    J(U) = ½ (Σᵢⱼ log E[Zᵢⱼ²] − log det S),

which is small when the rotated second moment is close to diagonal. Its score is a skew-symmetric
matrix `S_L − S_Lᵀ` with `S_L = Z (Z ∘ D⁻¹)ᵀ`, its Fisher information is diagonal in the rotation
coordinates, and dividing one by the other gives a natural-gradient step on `O(m) × O(n)`. Gimbal
takes that step with a bias-corrected rate, damping and a spectral trust region, retracts with
`I + Ω + Ω²/2` and a Newton–Schulz polish, and fits it to the gradient minus the part of the
momentum that a James–Stein factor judges to be signal. The variances that weight the score are
shrunk toward their Kronecker-separable fit by an empirical-Bayes factor, so the estimator behaves
like KL-Shampoo's when the variances are separable and like the free maximum-likelihood estimator
when they are not.

The theory shows that the likelihood identifies a rotation whenever two rows of variances differ
(pooled factors need their sums to differ) and that every single-factor estimator, SOAP's and
KL-Shampoo's included, is statistically inefficient except in special cases. Start with
[`theory.md`](theory.md); statements, proofs and evidence labels are in
[`theory/gimbal_theory.md`](theory/gimbal_theory.md), the Lean proofs in [`formal/`](formal/README.md),
and the paper in [`paper/`](paper/README.md).

## Using it

```bash
pip install -e ".[dev]"            # PyTorch reference implementations
pip install -e ".[dev,jax]"        # plus the JAX/Optax versions of AdamW, SOAP and Gimbal
```

PyTorch: hidden 2-D weight matrices go to Gimbal, everything else (embeddings, norms, biases) to
AdamW.

```python
from gimbal.torch import build

hidden = [p for name, p in model.named_parameters() if p.ndim == 2 and "embed" not in name]
other = [p for name, p in model.named_parameters() if p.ndim != 2 or "embed" in name]
opt = build("gimbal", hidden, other, matrix_kwargs={"lr": 3e-3}, other_kwargs={"lr": 3e-3})
```

JAX: `gimbal.jax.optax_api.gimbal(learning_rate, gimbal.jax.GimbalConfig())` is an Optax
transformation over a pytree of 2-D matrices; `gimbal.jax.gimbal.step` is the per-matrix step that
the TPU training loop batches and shards.

The knobs beyond Adam's are the rotation rate `rot_rate` (default 0.05 = 1 − β₂, one estimation
window for the frame and the second moment), `damping` (0.003) and `frame_every` (the frame moves
every `k_t ≤ 4` steps, every step at first). The variance shrinkage, the centring of the frame
statistic and the warm start have no knobs. Use the same momentum as the method you compare with:
the 125M study runs Gimbal with SOAP's β₁ = 0.95. Update rules, defaults and sources of every
optimizer in the repository are in [`docs/optimizers.md`](docs/optimizers.md).

## Reproducing

```bash
pytest -q                                              # PyTorch and JAX implementations
cd formal && lake exe cache get && lake build && lake env lean audit/Axioms.lean
python experiments/phase1/check_identities.py          # numerical checks of the theory
experiments/phase2/run_e21_all.sh && experiments/phase2/run_e21_ext.sh
experiments/phase2/run_phase2_rest.sh && python experiments/phase2/make_report.py
python scripts/data/fetch_fineweb_edu_sample.py        # small language model (CPU)
python experiments/phase3/run_e27_sweep.py --stage A
```

The TPU study ran on a v4-32 slice (4 hosts × 4 chips) with JAX 0.6.2. Data:
`python scripts/data/prepare_fineweb_edu.py` (checksums in `data/tokens_manifest.json`). Every run
executes a frozen copy of the code (`scripts/tpu/snapshot.sh`) on all hosts
(`scripts/tpu/launch.sh`):

```bash
python scripts/tpu/tune.py                 # Phase 06: equal-budget tuning
python scripts/tpu/freeze_configs.py       # configs/frozen/<optimizer>.yaml
python scripts/tpu/main_runs.py            # Phase 07: two seeds per optimizer
python analysis/phase08.py                 # every number of the comparison, from the run logs
python analysis/compute_budget.py && python analysis/readme_results.py
make -C paper                              # the paper, numbers generated from result files
```

Performance on the TPU (attention kernels, step times, collectives) is documented in
[`docs/performance.md`](docs/performance.md).

## Repository

| Path | Contents |
|---|---|
| `src/gimbal/torch/` | PyTorch reference implementations: Gimbal, SOAP (and its real-time variant), KL-SOAP, Muon, NorMuon, SPlus, ARO |
| `src/gimbal/jax/` | JAX implementations of AdamW, SOAP and Gimbal, the distributed step and Optax wrappers |
| `src/gimbal/train/` | 125M model, data pipeline, training loop and checkpoints for the TPU study |
| `theory.md`, `theory/` | readable overview; full theory with proofs and evidence labels |
| `formal/` | Lean 4 + Mathlib proofs |
| `experiments/` | Phase 01–03 experiments and their raw results |
| `scripts/` | data preparation, TPU launch, tuning, confirmatory runs |
| `analysis/` | Phase 06–08 analysis; generated tables and figures in `analysis/results/` |
| `runs/` | logs of the tuning, diagnostic and confirmatory runs |
| `paper/` | the paper (`make -C paper`) |
| `phases/`, `research/` | the research protocol, phase files and status; literature, hypotheses and ledgers of experiments, failures, decisions and claims |

## How the research was done

The project followed a written protocol in ten phases (`phases/README.md`), each with gates
fixed before the work started, a self-correcting loop for failures, and dependency tracking so
that a change to the mathematics re-opened every phase that used it. The ledgers in
`research/ledger/` record every experiment, every failure with its mechanism, and every decision.
The work was carried out by an AI agent (Claude, Anthropic) following that protocol under the
author's direction.

## Citation

```bibtex
@software{keni2026gimbal,
  title  = {Gimbal: Adam in a Maximum-Likelihood Kronecker Frame},
  author = {Keni, Tasmai},
  year   = {2026},
  url    = {https://github.com/tasmaikeni13/gimbal}
}
```

MIT license (`LICENSE`). The vendored official SOAP implementation in `tests/third_party/` keeps
its own MIT license.
