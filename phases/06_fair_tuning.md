# Phase 06 — Fair, equal-budget hyperparameter tuning

## Purpose

Give every optimizer the same tuning budget and the same chance to win, select configurations on
validation data only, and freeze them before the confirmatory runs.

## Depends on

Phase 05 (pipeline, base config). Phase 02's Gimbal defaults.

## Produces

`configs/tuning/*.yaml`, `runs/tuning/` (logs), `configs/frozen/<optimizer>.yaml`,
`research/ledger/tuning.md` (budget table, chosen values, curves).

## Protocol

**Stage A — learning rate, short horizon.** For every optimizer, 7 peak learning rates on a factor-2
grid centred on the literature default (record the centre and its source), at 1/4 of the token budget
(1,192 steps, same schedule shape), seed 0. If the best LR is at a grid edge, extend the grid by two
points on that side (the extension is part of the budget and is offered to every optimizer that hits
an edge).

**Stage B — learning rate, full horizon.** The best 3 LRs from Stage A, full 4,768 steps, seed 0.
This catches optimum shifts with horizon (arXiv:2609.04577).

**Stage C — one secondary hyperparameter per optimizer, 3 values, full horizon, best LR.** The most
sensitive published knob of each optimizer, with every other value at its published default:

| Optimizer | Secondary hyperparameter (3 values) |
|---|---|
| AdamW | $\beta_2$ ∈ {0.95, 0.98, 0.999} |
| SOAP / SOAP real-time | shampoo $\beta$ ∈ {0.9, 0.95, 0.99} |
| KL-SOAP | $\beta_{\text{kron}}$ ∈ {0.9, 0.95, 0.99} |
| Muon / NorMuon | momentum ∈ {0.9, 0.95, 0.98} |
| SPlus | EMA rate of the frame statistics ∈ published neighbourhood |
| Gimbal | rotation rate $\alpha$ ∈ {½, 1, 2} × Phase 02 default |
| others | as documented in `docs/optimizers.md` |

Weight decay is fixed at 0.1 (decoupled, matrices only) for every optimizer unless Stage C for all
optimizers includes it; it is not tuned for one optimizer only.

Note from Phase 02 (E2.9, `experiments/phase2/results/e29_report.md`): on synthetic streams a
smaller damping ($\delta=3\cdot10^{-4}$) and a shorter warm start (20 steps) were slightly better than
Gimbal's defaults in several cells, and every other ablated component was neutral or worse. The
equal-budget rule still allows one knob per optimizer; keep the rotation rate (the analogue of the
peers' memory knobs) unless the knob of every optimizer is re-chosen by the same rule.

**Selection.** Lowest final validation loss (5M-token subset at Stage A, full 20M at Stages B/C).
Ties within 0.002 nats → prefer the smaller LR. Record every trial, including diverged ones.

**Freeze.** Write `configs/frozen/<optimizer>.yaml`, commit, and tag `tuning-frozen`. After the tag,
configurations change only through protocol §5, and a change for one optimizer triggers the same
re-tuning budget for it alone (competitors are not re-run unless the pipeline changed).

## Exit gate

* G6.1 Every optimizer received exactly the same number of trials (table in `tuning.md`).
* G6.2 No selected value lies at a grid edge.
* G6.3 Frozen configs are committed and tagged; the test split was never read.

## Failure handling

Divergence across the whole grid for some optimizer → check its implementation against Phase 03
golden tests and its published settings (warm-up, ε, clipping); fix bugs, not hyperparameters of the
protocol. An optimizer that is unstable at every LR of a fair grid is reported as such.
