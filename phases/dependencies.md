# Dependency map (protocol §5)

Each theory item and the places that consume it. When an item changes, every row it reaches
(transitively) becomes stale.

| Theory item | Lean | Code | Experiments | Phases | Paper |
|---|---|---|---|---|---|
| KRD model, working likelihood (§1) | all files (as hypotheses) | `gimbal.py` (`_generator`) | E2.1–E2.5 generators | 02–09 | Method, Theory |
| Prop. 1 (profile KL $J$) | `Flow.lean` (per coordinate) | — | frame-KL metric in E2.1–E2.3, E2.6 | 02, 08 (diagnostics) | Theory, Experiments |
| Prop. 2 (what pooled factors estimate) | — | `soap.py`, `klsoap.py` | E2.2 | 02 | Theory |
| Lemma A (score) | `Flow.lean`, `Equivariance.lean` | `Gimbal._generator` | all | 02–08 | Method |
| Thm 2 (Fisher, identifiability) | `Identifiability.lean`, `Flow.lean` | `Gimbal._generator` (Fisher normalizer) | E2.2 | 02 | Theory |
| Thm 3 (efficiency) | `Efficiency.lean` | — | E2.1, `check_identities.py` | 01–02, 08 (mechanism check) | Theory, main claim |
| Thm 4 (flow: fixed points, rate, noise) | `Flow.lean` | `rot_rate`, `damping` defaults | E2.1, E2.3, E2.9 | 02, 06 (α grid) | Theory |
| Thm 5 (retractions) + C-001 safeguards | `Retraction.lean` | `expm2`, `polish_until_orthogonal`, `skew_spectral_norm` | tests | 03, 04 (cost) | Method |
| Thm 6 (equivariance) | `Equivariance.lean` | tests | — | 03 | Theory |
| Thm 7 (transport) | `Transport.lean` | `transport=True` | E2.9 ablation | 02, 06 | Ablations |
| Thm 8 (descent, scale invariance) | `Equivariance.lean` | tests | — | 03 | Theory |
| Prop. 9 (cost) | — | — | E2.8 | 04, 08 | Method, Experiments |
| Defaults $(\alpha,\delta,\rho,\theta_{\max})$ | — | `Gimbal.__init__` | E2.9 | 03–07 | Hyper-parameter table |
| Peer set and their rules | — | `src/gimbal/torch/*` | E2.5–E2.7 | 03–08 | Baselines |
