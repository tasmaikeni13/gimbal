# Failure ledger

Classes follow `phases/README.md` §4. A failure is closed only with its mechanism and evidence.

| ID | Phase | What failed | Class | Mechanism | Evidence | Resolution / reopening condition |
|---|---|---|---|---|---|---|
| F-000 | 01 (prototype) | SOAP every-step and KL-SOAP prototypes diverged or lost to SOAP f=10 on the noisy quadratic | implementation bug (prototype) | fresh `eigh` each step flips eigenvector order and sign, scrambling Adam's V; KL factors initialised at ~0 gave huge inverses | `scratchpad/proto_nqm.py` (not in repo) | replaced by faithful warm-started QR implementations (`src/gimbal/torch/soap.py`, `klsoap.py`); prototype numbers for those two methods are discarded |
| F-001 | 01 | `test_gimbal_frames_stay_orthogonal`: ‖QᵀQ−I‖ = 0.30 | numerical instability | at large rotation rates the per-step generator had ‖Ω‖₂ ≈ 0.9; the second-order retraction's Ω⁴/4 defect accumulated between polishes and one Newton–Schulz step could not remove it | pytest output, 2026-10-06 | change C-001 (spectral trust region ‖Ω‖₂ ≤ 1, polish every step, repeat polish until the defect is below working precision); test now passes at 1e-8 |
| F-002 | 01 | transport test: total second moment drifted 15% | implementation bug | renormalized the columns of P∘P; the total is preserved by unit **row** sums | pytest output | fixed (rows); test passes at 1e-6 |
| F-003 | 01 | battery: flow and tie checks converged to ±90° | evaluator defect | the check's angle variable is the negative of the generator coordinate, so the check performed likelihood *ascent*; the optimizer itself uses the correct sign (tests and prototypes converge) | `check_identities.py` first run | sign fixed in the check with an explanatory comment; stationary variance and tie identifiability then pass |
| F-004 | 01 | battery: delta-method check 28% off for the weighted estimator | proxy-fidelity failure | N = 400 too small for first-order perturbation theory at the weighted estimator's smaller eigen-gap | `check_identities.py` first run | N raised to 3000 (quick) / 8000 (full); agreement within 3% |
| F-005 | 01 | battery: Newton–Schulz identity error 4.7e-9 | evaluator defect | absolute tolerance on matrices with entries up to 10²; relative error is 1e-15 | first run | tolerance made relative |
