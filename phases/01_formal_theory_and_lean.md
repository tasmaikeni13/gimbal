# Phase 01 — Formal theory and Lean verification (ours and peers)

## Purpose

State the mathematics of Gimbal and of its peers in one framework, prove what can be proved, try
hard to break what cannot, and machine-check the load-bearing results in Lean 4 + Mathlib. The phase
must leave a theory that *predicts* where Gimbal should beat each peer, so that Phase 2 tests
predictions instead of searching blindly.

## Depends on

* `research/literature/frontier.md` (closest work, SOAP weaknesses W1–W9)
* `research/hypotheses/portfolio.md` (H1 Gimbal and reserve branches)

## Produces

* `theory/gimbal_theory.md` — statements, proofs, evidence labels, peer table, changelog
* `formal/Formal/*.lean` — Lean proofs; `formal/README.md` — map from theorem to Lean name
* `experiments/phase1/` — counterexample battery and numerical checks of every identity
* `phases/dependencies.md` — theory item → consumers map (Section 5 of the protocol)

## Skills

`theory-research` (attack protocol, proof audit), `mechanism-transfer` (the BSS/CPC transfer),
`literature-frontier` (status of each statement).

## Tasks

1. **Freeze the statements.** For each object (KRD model, profile objective $J$, weighted-factor
   estimators, the Gimbal flow, retractions, transport) write the definition table and the quantifier
   string of each claim and its negation (`theory-research` attack protocol §1).
2. **Peers in the same framework.** Express AdamW, Shampoo, KL-Shampoo, SOAP (f-stale and real-time),
   KL-SOAP, SPlus, Muon, NorMuon and ARO as "frame + eigenvalue rule". For each derive: the frame
   estimator's statistical model, its identifiability condition, its asymptotic efficiency relative to
   the Cramér–Rao bound, its staleness, and its non-matmul cost.
3. **Prove** Propositions 1–2 and Theorems 2–8 of the theory document in full (written proofs with a
   dependency DAG). For each proof, list every assumption it uses.
4. **Counterexample battery** (`experiments/phase1/check_identities.py`): random and adversarial
   numeric checks of every identity and inequality (equality cases, near-degenerate gaps, extreme
   ratios, $m\ne n$, $n=1$), including the efficiency inequality, the separability characterization,
   the retraction identities, Newton–Schulz error, transport double-stochasticity, equivariance and
   scale invariance of one Gimbal step.
5. **Lean.** Formalize at least:
   * the weighted Cauchy–Schwarz efficiency inequality and its pooled / KL corollaries;
   * Fisher positivity ⇔ distinct profiles; pooled identifiability ⇒ profile identifiability; a tie
     witness where pooled fails and the likelihood identifies;
   * Cayley orthogonality, the $I+\Omega+\Omega^2/2$ defect identity, the Newton–Schulz error identity;
   * Kronecker–Hadamard square identity and double stochasticity of $P^{\circ2}$;
   * one-step equivariance of the rotated coordinates and of the update;
   * scale invariance of the generator and of the Fisher normalizer; the descent identity;
   * the fixed point and geometric convergence of the variance recursion of Theorem 4.
   Check `#print axioms` for each headline theorem (only `propext`, `Classical.choice`, `Quot.sound`).
6. **Fidelity audit.** For every Lean theorem, write one line comparing the formal statement with the
   natural-language theorem (AlphaProof caveat in the `theory-research` skill: a valid formal proof
   of the wrong statement is a failure).
7. **Proof audit** with the `theory-research` checklist; record it at the end of the theory doc.
8. **Write `phases/dependencies.md`.**

## Exit gate

* G1.1 `cd formal && lake build` succeeds; `grep -rn "sorry\|admit" formal/Formal` is empty; the
  headline theorems depend only on the three standard axioms.
* G1.2 Every theorem in the theory document is labelled L3 or higher, or is explicitly a conjecture.
* G1.3 At least one result that separates Gimbal from SOAP **and** from KL-SOAP (efficiency or
  identifiability) is machine-checked.
* G1.4 The counterexample battery runs clean (or every failure led to a revised statement).
* G1.5 The peer table is complete and every row cites the theorem or computation behind it.

## Failure handling

* A statement fails the battery → keep the counterexample, find the missing hypothesis or the true
  weaker statement, update the theory (changelog), and apply protocol §5 to everything that cited it.
* A Lean proof stalls → prove a special case (e.g. $2\times2$ or a fixed finite index type) and label
  the general case L3/L4; do not weaken the natural-language claim silently.
* The theory no longer predicts an advantage over a peer → that is a result: record it, revise the
  hypothesis portfolio, and decide (with evidence) whether the method or the claim changes.

## Invalidation triggers

Any change to the Gimbal update rule, the working likelihood, the defaults ($\alpha,\delta,\rho$), or
the set of peers.
