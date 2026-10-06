# AGENTS.md

Instructions for AI coding agents working in this repository. Human-facing documentation is in
`README.md`; the research protocol is in `phases/README.md`. When this file and a user's direct
instruction disagree, follow the user.

## What this project is

Gimbal is a matrix-preconditioned optimizer: Adam in a Kronecker frame `(Q_L, Q_R)`, like SOAP, but
the frame is the online maximum-likelihood estimate under the model Adam's diagonal assumes (a joint
diagonalization), updated by a matrix-multiply-only natural-gradient flow. The project is a research
program run in ten phases (`phases/`), ending in a paper and a TPU v4-32 comparison against SOAP and
its successors at 125M parameters / 2.5B FineWeb-Edu tokens.

## Map

| Path | Contents |
|---|---|
| `src/gimbal/torch/` | PyTorch reference optimizers: `gimbal.py` (ours), `soap.py`, `klsoap.py`, `muon.py` (Muon, NorMuon), `splus.py`, `aro.py`, `factory.py` (parameter routing) |
| `tests/` | invariant and golden tests; `tests/third_party/soap_official.py` is the vendored official SOAP (do not edit) |
| `theory.md` | readable overview of the theory and equations (keep consistent with the paper below) |
| `theory/gimbal_theory.md` | living theory: statements, proofs, evidence labels L1–L5, changelog |
| `formal/` | Lean 4 + Mathlib proofs (`formal/README.md` maps theorems to Lean names) |
| `experiments/phase1/`, `experiments/phase2/` | executable experiments; raw outputs in `results/` |
| `research/` | literature frontier, hypotheses, ledgers (experiments, failures, decisions, claims), state |
| `phases/` | protocol (`README.md`), phase files `01`–`10`, `STATUS.md`, `dependencies.md` |
| `scripts/data/` | data download scripts; downloaded data goes to `data/raw/` (git-ignored) |

## Setup

```bash
pip install -e ".[dev]"                     # Python ≥ 3.10, CPU PyTorch is enough for phases 1–3
cd formal && lake exe cache get && lake build   # Lean toolchain via elan; Mathlib cache ~5 GB
```

## Commands you must run before claiming a change works

```bash
pytest -q                                          # all tests must pass
ruff check src tests experiments scripts           # must report "All checks passed!"
cd formal && lake build && lake env lean audit/Axioms.lean   # after any Lean change
python experiments/phase1/check_identities.py --quick        # after any change to the theory or to gimbal.py
```

Experiments (CPU):

```bash
python scripts/data/fetch_fineweb_edu_sample.py              # FineWeb-Edu sample for the small LM
experiments/phase2/run_e21_all.sh && python experiments/phase2/analyze_e21.py
OMP_NUM_THREADS=1 python experiments/phase2/e25_noisy_quadratic.py
python experiments/phase2/e27_small_lm.py --method gimbal --lr 3e-3 --seed 0 --steps 800
```

Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for any multi-process experiment; otherwise BLAS
threads oversubscribe the CPU and runs slow down by an order of magnitude.

## Working rules

* **Follow the phase protocol.** To execute a phase, read `phases/README.md`, then the phase file,
  and update `phases/STATUS.md`. Failures go through the self-correcting loop (protocol §4);
  changes to mathematics propagate through `phases/dependencies.md` (protocol §5).
* **Use the research skills** from `https://github.com/tasmaikeni13/skills` (literature-frontier,
  theory-research, mechanism-transfer, ml-research, experimental-research) as the protocol directs.
* **Theory, Lean and code move together.** If you change an update rule or a default in
  `src/gimbal/torch/gimbal.py`, update `theory/gimbal_theory.md` (statement, label, changelog),
  re-check affected Lean statements, record a change-id in `research/ledger/decisions.md`, and mark
  dependent phases stale.
* **Numbers come from scripts.** Never type a result into a document; generate it from raw outputs
  under `experiments/*/results/` or `runs/` and cite the file.
* **Competitors get their best implementation.** Match each baseline to its source of truth
  (paper/official code); `tests/test_optimizers.py::test_soap_matches_official` must keep passing.
* **Keep the diff minimal and the comments useful**: explain why (numerics, invariants, theorem
  numbers), not what the next line does. PEP 8 via ruff (line length 100).
* **Commits**: one logical change per commit; message says what changed and why; push to the
  working branch given by the user.

## Boundaries

**Always**
* Run the commands above and report their actual output.
* Record failed runs and refuted statements in `research/ledger/failures.md` with their mechanism.
* Label theory results with their evidence level; a Lean proof with `sorry` is not a proof.

**Ask first**
* Changing a frozen configuration (`configs/frozen/`, after Phase 06) or a predeclared gate.
* Deleting results, branches, or data; changing repository visibility; anything that costs money
  (TPU time, paid APIs) beyond what the current phase file authorizes.
* Adding dependencies outside `pyproject.toml` extras or upgrading Lean/Mathlib.

**Never**
* Invent metrics, citations, or experiment outcomes, or report an unfinished run as finished.
* Read the held-out test split before Phase 08's decision rule has been applied.
* Weaken a competitor (fewer tuning trials, slower implementation) or tune only Gimbal.
* Edit `tests/third_party/` or the vendored license files.
* Commit `data/raw/`, `formal/.lake/`, checkpoints, or credentials.
