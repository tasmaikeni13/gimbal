# Phase 04 — TPU v4-32 kernels and distributed optimizer step

## Purpose

Make every optimizer as fast as it can reasonably be on a TPU v4-32 slice, with the same engineering
effort for all of them, and verify that Gimbal's step is no slower than SOAP's. Wall-clock claims in
the paper rest on this phase.

## Hardware facts to verify at the start (do not assume)

* v4-32 = 16 TPU v4 chips (megacore: one JAX device per chip), 4 hosts × 4 chips, 32 GiB HBM per chip.
* Multi-host JAX: the same program runs on all hosts (`gcloud compute tpus tpu-vm ssh <name>
  --worker=all --command=...`); call `jax.distributed.initialize()`; check `jax.device_count()==16`.
* Record JAX/jaxlib/libtpu versions, XLA flags, and the TPU runtime version in `runs/env.json`.

## Depends on

Phase 03 (`src/gimbal/jax/`, tests, golden sequences).

## Produces

* `src/gimbal/jax/distributed.py` — shape-bucketed, sharded optimizer step shared by all matrix
  optimizers
* `src/gimbal/jax/kernels/` — Pallas kernels that measurably help (keep XLA versions as fallbacks)
* `benchmarks/tpu/` — microbenchmarks per matrix shape and end-to-end step timing; raw traces
* `docs/performance.md` — methodology and tables

## Tasks

1. **Shape bucketing.** Group same-shaped matrices across layers (e.g. all 768×768, all 768×2048),
   stack them, and run each optimizer with `vmap`/batched matmuls. Small per-layer matmuls waste the
   MXU; batching is the first-order win and must be given to every optimizer.
2. **Sharded optimizer work.** The 125M model fits replicated, but replicated optimizer work wastes
   16× compute. Shard the stacked optimizer state and work across the 16 devices along the stacking
   axis (pad if needed), then all-gather the updates (ZeRO-1 style). Use `jax.sharding` with
   `NamedSharding`/`shard_map`; verify with the HLO dump that no unintended all-gathers of optimizer
   state occur.
3. **Precision policy.** fp32 optimizer state. Gimbal frame flow and Newton–Schulz polish in fp32
   (`precision=HIGHEST`); rotations of G and M may use bf16 inputs with fp32 accumulation if the
   Phase 03 invariants and golden tests still pass. Apply the same policy class to every optimizer's
   analogous operations.
4. **Competitors' best TPU paths.** SOAP/KL-SOAP: QR via `jnp.linalg.qr`, compare with Cholesky-QR
   (`R = chol(Sᵀ S)`, `Q = S R^{-1}`, as in ARO) and with QDWH `eigh`; use the fastest that passes
   the tests. Muon/NorMuon: NS5 with fp32 or bf16 as in the reference; consider Polar Express
   coefficients only if they are part of a published variant being compared. SPlus: eigh every 100
   steps as published.
5. **Gimbal options** (keep the defaults from Phase 02 unless the gate fails): amortized frame move
   (`frame_every`, default 4 since C-012; the score is accumulated, so no gradient is dropped), NS polish period, Fisher
   matrix refresh period. With `frame_every = k > 1` the momentum may be kept in rotated
   coordinates and transported only when the frame moves (saves one $m^2n+mn^2$ unit per step;
   prove equality with the reference to rounding first). The centering of the frame statistic
   (Prop. 5.7, C-013) needs the rotated previous momentum, which that layout holds directly, and
   two scalar reductions per layer ($\|M\|^2$, $\sum V$). The shrinkage step (Prop. 5.5) is
   elementwise plus row/column means and one global variance: fuse it into one kernel with the
   variance-average updates. The warm start needs one `eigh` per matrix at step 50 and a
   temporary $m^2+n^2$ buffer; run it outside the jitted steady-state step if that keeps the
   steady state free of `eigh`. Any option that changes the mathematics is a protocol §5 change.
6. **Pallas kernels** only where the profile shows an elementwise/reduction bottleneck (candidates:
   fused EMA + generator weights, fused $\Omega$ construction with damping and clipping). Each kernel
   has a test against the XLA version.
7. **Measure.** For each optimizer: steady-state training-step time at the Phase 05 configuration
   (median over steps 200–400 after compilation), optimizer-only time, peak HBM, and compile time.

## Exit gate

* G4.1 Golden and invariant tests pass on TPU (fp32 tolerances documented).
* G4.2 Gimbal's steady-state step time ≤ SOAP (f=10)'s, and < SOAP real-time's and KL-SOAP's.
* G4.3 No optimizer is slower than necessary because of missing batching or sharding (every
  optimizer's optimizer-only time is within 2× of its FLOP-model lower bound, or the reason is
  documented).
* G4.4 `docs/performance.md` reproduces from `benchmarks/tpu/` scripts.

## Failure handling

If G4.2 fails: profile first. Typical causes are unbatched shapes, fp32 emulation cost, an
accidental per-step recompilation, or the $m^3$ retraction on the 2048/3072 side. Repairs in order:
batching → amortized retraction → bf16 inputs with fp32 accumulation → kernel fusion. If the gap
remains, document it honestly; a wall-clock loss is a result, and the paper's claims must change.

## Invalidation triggers

Algorithm changes (Phase 01/02), JAX/libtpu upgrades, model-shape changes (Phase 05).
