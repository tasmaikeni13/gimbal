# Research state — Gimbal — updated 2026-10-07 (all phases)

## Contract
Target claim: Gimbal reaches lower validation loss than SOAP (and AdamW) at 125M parameters /
2.5B FineWeb-Edu tokens, with step time no worse than SOAP's, on TPU v4-32 (D-003: AdamW, SOAP
f = 10 and Gimbal). Decision rule: Phase 08 (D-005 definitions).

## Outcome
- Theory and Monte Carlo (Phases 01–02): the likelihood frame is identifiable and efficient where
  pooled factors are not; on synthetic non-separable streams Gimbal's frame is far more accurate
  than SOAP's and KL-SOAP's (C2–C7).
- Small language model (Phase 03): premise fails (κ ≈ 0.005); Gimbal slightly ahead of SOAP,
  KL-SOAP ahead of Gimbal; H1's kill criterion met at small scale (C8–C11, D-006).
- 125M / 2.5B tokens (Phases 06–08): Gimbal beats AdamW on loss and wall-clock; matches SOAP
  within seed noise and is about 2% slower per step; the headline claim is not allowed. The
  premise fails again (noise-corrected κ ≈ 0.003–0.005). The early instability found in tuning
  was Adam's momentum β₁ = 0.9 for SOAP and Gimbal alike (F-032, C-019). Results in
  `analysis/decision.md`; claims C13–C22.

## Live hypotheses (not tested here)
- Gimbal with a faster frame than the tuned edge value α = 0.2, and a gentler early frame
  schedule at high learning rates (the tuning grid's edge; early-phase fragility).
- KL-SOAP at 125M (better than Gimbal at small scale under separable statistics).
- Models, layers or regimes with measured non-separability (the last MLP projection had the
  largest pooled-frame index among the probed matrices).

## Killed hypotheses
- H1's non-separability mechanism on the two models studied (premise κ ≪ 0.05 at both scales).

## Next actions
1. Phase 10 release (README, humanized prose, fresh-clone verification, tag v1.0).
2. Owner: the project's billing account is delinquent (GCS bucket creation refused); checkpoints
   are only on the TPU hosts' disks.
