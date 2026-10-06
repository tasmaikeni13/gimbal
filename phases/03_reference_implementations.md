# Phase 03 — Reference implementations of Gimbal and every competitor (PyTorch + JAX)

## Purpose

Produce correct, readable, tested implementations of all optimizers in two frameworks:
PyTorch (`src/gimbal/torch/`, already used in Phase 2) and JAX/Optax (`src/gimbal/jax/`, used on
TPU). Correctness is established here, on CPU, so that Phase 4 only has to make things fast.

## Depends on

Phase 02 (frozen Gimbal algorithm and defaults; peer list). If Phase 02 changed the algorithm,
`theory/gimbal_theory.md` §5 is the specification, not this file's examples.

## Produces

* `src/gimbal/jax/` — one Optax `GradientTransformation` per optimizer with a common config schema
* `src/gimbal/torch/` — cleaned PyTorch versions
* `tests/` — unit, invariant and cross-framework tests; `tests/golden/` fixed gradient sequences
* `docs/optimizers.md` — exact update rules and defaults for every optimizer, with sources

## Optimizers (core set must be complete; extended set as time allows, record omissions)

| Optimizer | Source of truth for the rule | Notes |
|---|---|---|
| Gimbal | `theory/gimbal_theory.md` §5 | left/right flows, retraction, NS polish, optional transport, eigh init |
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

## Failure handling

Mismatch between frameworks or with an official implementation → reduce to a single step on a 3×2
matrix and compare intermediate tensors. Do not "fix" a competitor to match our expectations; match
its source of truth.

## Invalidation triggers

Changes to any update rule or default (Phase 01/02), or to the parameter-routing policy.
