# Phase 08 — Analysis, statistics, mechanism diagnostics and the decision

## Purpose

Decide, by a rule written before seeing the Phase 07 results, whether Gimbal beats SOAP and every
peer on perplexity/loss and wall-clock; explain why with mechanism evidence; and, if it does not,
send the project back through the adaptive loop instead of re-interpreting the data.

## Depends on

Phase 07 runs, Phase 02 predictions, theory document.

## Produces

`analysis/` (scripts), `analysis/results/` (generated tables/figures), `research/ledger/claims.md`
(final), `analysis/decision.md`.

## Metrics

1. Final validation loss and perplexity (full 20M tokens); test loss/perplexity read once, after the
   decision rule is applied to validation, for the paper.
2. Loss vs tokens; loss vs wall-clock (steady-state step time × steps + measured eval/ckpt overheads
   reported separately).
3. Tokens and wall-clock to reach targets: AdamW's final loss; SOAP's final loss.
4. Token multiplier vs AdamW and vs SOAP. If budget allows, the SOAP-paper method: Gimbal runs at
   0.5, 0.625, 0.75, 0.875 of the token budget with matched schedules, fit $a+bN^{-\beta}$.
5. Stability: number of loss spikes (definition fixed in advance), max gradient norm.
6. Mechanism diagnostics: non-separability $\kappa$ of $D$ over training; frame KL $J$ of SOAP's and
   Gimbal's frames evaluated on held-out gradient snapshots; rotation magnitudes; whether layers with
   larger $\kappa$ show larger per-layer gains (Phase 01/02 prediction).

## Statistics (two seeds per optimizer)

* Unit = training run; pairing = seed.
* Report both seeds' values and their paired differences; never only the mean.
* Within-run uncertainty of final validation loss: bootstrap over the 20M-token evaluation batches.
* Between-seed variability: for each optimizer, the seed difference; pooled seed SD across
  optimizers as the noise scale.
* Holm correction across competitors for any test reported.

## Decision rule (predeclared)

Gimbal **beats** competitor *c* on loss if, for both seeds, Gimbal's final validation loss is lower
than *c*'s on the same seed, and the mean paired difference exceeds twice the pooled within-run
bootstrap SE. Gimbal **beats** *c* on wall-clock if its steady-state step time is lower **or** its
wall-clock to reach *c*'s final loss is lower. The headline claim "Gimbal outperforms SOAP and its
peers" is allowed only if Gimbal beats every core competitor on loss and is no slower than SOAP
(f=10) in steady-state step time.

## If the rule is not met

This is not the end of the project; it is the adaptive loop:

1. Write `analysis/decision.md` with the observed outcome and the diagnostics.
2. Classify the failure (protocol §4). Typical classes here: premise failure ($\kappa$ small on real
   gradients), mechanism present but gains smaller than seed noise, frame flow too slow early,
   wall-clock loss.
3. Research and repair in Phase 02 (Monte Carlo on *saved real gradients* first), update the theory,
   rewrite downstream phases (protocol §5), re-tune Gimbal with the same budget, re-run Gimbal's
   seeds, and apply this rule again **on new seeds** to avoid selecting on noise.
4. If the protocol's escalation condition is reached, report the negative result honestly: what was
   predicted, what was observed, and what the evidence rules out.

## Exit gate

* G8.1 All analysis regenerates from raw logs with one command.
* G8.2 The decision rule was applied as written and the outcome recorded.
* G8.3 Every claim in `claims.md` has its evidence, scope and allowed wording.
