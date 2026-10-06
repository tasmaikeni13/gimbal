# Phase 03 — Reference implementations of Gimbal and every competitor (PyTorch + JAX)

## Purpose

Produce correct, readable, tested implementations of all optimizers in two frameworks:
PyTorch (`src/gimbal/torch/`, already used in Phase 2) and JAX/Optax (`src/gimbal/jax/`, used on
TPU). Correctness is established here, on CPU, so that Phase 4 only has to make things fast.

## Depends on

Phase 02 (frozen Gimbal algorithm and defaults; peer list). If Phase 02 changed the algorithm,
`theory/gimbal_theory.md` §5 is the specification, not this file's examples. The FineWeb-Edu sample
(`scripts/data/fetch_fineweb_edu_sample.py`) for E3.1–E3.2.

## Produces

* `src/gimbal/jax/` — one Optax `GradientTransformation` per optimizer with a common config schema
* `src/gimbal/torch/` — cleaned PyTorch versions
* `tests/` — unit, invariant and cross-framework tests; `tests/golden/` fixed gradient sequences
* `docs/optimizers.md` — exact update rules and defaults for every optimizer, with sources

## Optimizers (core set must be complete; extended set as time allows, record omissions)

| Optimizer | Source of truth for the rule | Notes |
|---|---|---|
| Gimbal | `theory/gimbal_theory.md` §5 | left/right flows, retraction, NS polish, flow variance average tied to the frame memory with empirical-Bayes shrinkage and odd/even split (Lemma 5.4, Prop. 5.5), pooled warm start with one eigh at step 50 (Remark 5.6), optional transport; state `QL, QR, M, V, VF, VF_odd` (+ `L_acc, R_acc` for 50 steps) |
| AdamW | Loshchilov & Hutter 2019 | decoupled weight decay |
| SOAP (f=10) | arXiv:2409.11321 + official repo `nikhilvyas/SOAP` | first step only initializes; V re-ordered by estimated eigenvalues at refresh |
| SOAP real-time | arXiv:2607.20548 §5.4.2 | factors include current gradient; frame refreshed every step before projection |
| KL-SOAP | arXiv:2607.20548 Alg. 2; arXiv:2509.03378 | KL factor accumulation with eigenvalue EMA |
| KL-Shampoo (extended) | arXiv:2509.03378 Fig. 3/12 | separable eigenvalues, no Adam |
| Muon | Jordan 2024; arXiv:2502.16982 | Newton–Schulz (5 steps, coefficients 3.4445, −4.7750, 2.0315), Nesterov, RMS-matched scale |
| NorMuon | arXiv:2510.05491 | neuron-wise second moment after orthogonalization |
| SPlus | arXiv:2506.07254 + `kvfrans/splus` | instant-sign in a stale frame, shape scaling, iterate averaging for eval |
| ARO (extended) | arXiv:2602.09006 | one-sided per-step rotation by QR of $M f(R^\top M)^\top$ |
| Shampoo + Adam grafting (extended) | DistributedShampoo (arXiv:2309.06497) | |
| COSMOS, PSGD-Kron (extended) | arXiv:2502.17410; Xi-Lin Li's PSGD | |

Where a TPU-tested JAX implementation already exists (for example in Marin/Levanter, the code base
behind arXiv:2509.02046), compare against it; do not adopt it unverified.

Parameter routing is identical for every matrix optimizer: 2-D hidden matrices (attention q,k,v,o;
MLP gate/up/down) use the matrix optimizer; embeddings, LM head, norms and biases use AdamW with the
same settings for all optimizers. Fused QKV is treated as one matrix for every optimizer.

## Tests (all must pass)

1. **Invariants** (from Phase 01 theorems): orthogonality of Gimbal frames after 10⁴ random steps
   (‖QᵀQ−I‖ < 1e-5 in fp32 with the NS polish), equivariance of one step under random orthogonal
   $(P,R)$, scale invariance of the frame flow, descent identity with $\beta_1=0$.
2. **Golden sequences**: each optimizer applied to 50 fixed gradients on fixed shapes; PyTorch and
   JAX agree to 1e-5 relative (fp32). SOAP agrees with the official implementation; Muon with the
   reference implementation.
3. **Toy convergence**: each optimizer reduces a convex quadratic and overfits a tiny MLP.
4. **Edge cases**: $m=1$ or $n=1$, very large side (one-sided mode), zero gradients, NaN guards,
   bf16 parameters with fp32 optimizer state.

## Exit gate

* G3.1 All tests pass on CPU for both frameworks.
* G3.2 `docs/optimizers.md` documents every rule and default with its source.
* G3.3 Phase 2's E2.5 noisy-quadratic results reproduce with the JAX implementations within
  seed-level noise (a cross-framework check of the science, not only of the code).
* G3.4 (premise, formerly G2.5) E3.1 finds the non-separability index $\kappa$ clearly above 0 on
  real LM gradients: median noise-corrected $\kappa\ge0.05$ over snapshot matrices and checkpoints
  (C-005). The held-out frame comparison (Gimbal vs SOAP vs KL) is reported as supporting evidence.
* G3.5 (small LM, formerly G2.6) In E3.2 Gimbal's mean final validation loss is lower than every
  peer's at equal steps (paired over seeds; Holm-corrected one-sided paired test $p<0.05$, or lower
  on every seed when only 3 seeds are affordable), for the configuration that passed Phase 02's
  cost gate.

## Small-scale validation on real gradients (moved from Phase 02 by C-006)

These two experiments need language-model training, so they run here, on the user's hardware,
with the PyTorch reference optimizers (and again with JAX once it exists). The scripts are ready in
`experiments/phase3/`; nothing in Phase 02 trains a model.

| ID | Question | Design | Primary metric |
|---|---|---|---|
| E3.1 (formerly E2.6) | Premise check on real gradients | Train the small byte-level LM; at steps 100, 400, 800 collect 96 independent minibatch gradients of selected matrices at fixed weights; on a fit half, estimate the pooled (SOAP), KL and Gimbal (batch likelihood with empirical-Bayes variances) frames; score them on the held-out half | $\kappa$; held-out log-likelihood gain in nats |
| E3.2 (formerly E2.7) | Small-LM benchmark | Byte-level Llama-style LM (d=128, 4 layers) on the FineWeb-Edu sample; every optimizer gets the same 4-point LR grid (factor 2), extended outward while its best LR is on an edge; 3 seeds at the selected LR; Gimbal at the Phase 02 default and with `frame_every=4` | validation loss at equal steps; loss vs wall-clock |

```bash
python scripts/data/fetch_fineweb_edu_sample.py                      # data (once)
python experiments/phase3/run_e27_sweep.py --stage A --jobs 4        # E3.2 stage A (seed 0, LR grid)
python experiments/phase3/run_e27_sweep.py --stage B --seeds 1 2     # E3.2 stage B
python experiments/phase3/analyze_e27.py                             # G3.5
for m in soap gimbal; do                                             # E3.1 snapshots
  python experiments/phase3/e27_small_lm.py --method $m --lr <selected> --seed 0 \
    --snapshots 100,400,800 --tag _snap
done
python experiments/phase3/analyze_e26.py                             # G3.4
```

Use `OMP_NUM_THREADS=1` per process when running several in parallel on a CPU.

## Failure handling

Mismatch between frameworks or with an official implementation → reduce to a single step on a 3×2
matrix and compare intermediate tensors. Do not "fix" a competitor to match our expectations; match
its source of truth.

## Invalidation triggers

Changes to any update rule or default (Phase 01/02), or to the parameter-routing policy.
