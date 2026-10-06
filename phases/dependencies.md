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
| Lemma 5.4 (flow variances: profile likelihood with forgetting, plug-in cost) + C-003 | `Variance.lean` | `flow_beta="tied"`, `VF` buffer | E2.1 (`gimbal_v03` ablation), E2.9 | 02, 03 (JAX state), 04 (memory) | Method, Theory |
| Prop. 5.5 (empirical-Bayes shrinkage, split noise estimate) + C-004 | `Variance.lean` | `flow_shrink`, `Gimbal._shrunk_variances`, `VF_odd` buffer | E2.1 (`gimbal_noshrink` ablation), E2.4, E2.9 | 02, 03 (JAX), 04 (elementwise kernels, memory) | Method, Theory, Ablations |
| Remark 5.6 (pooled warm start, one-step estimator) + F-012 | `Variance.lean` (`restart_closed_form`) | `init="pooled"`, `warm_start_steps`, `Gimbal._warm_start` | E2.1, E2.2 | 03 (JAX: eigh once at step 50), 04 (one-off eigh) | Method |
| Prop. 5.7 (frame statistic of a gradient with a mean; empirical-Bayes centering) + C-013 | `Variance.lean` (`ema_weight_sq_sum`, `shrinkage_risk_eq_iff`) | `flow_center="adaptive"`, `Gimbal._mean_shrinkage` | E2.5 (`gimbal_k4_nocenter`, `gimbal_k4_center` ablations), E2.1 (`*_c012` rows), E2.9 | 03 (JAX: rotate the previous momentum), 04 (two reductions per layer), 06 (no new knob) | Method, Theory, Ablations |
| Prop. 9 (cost) | — | `e28_cost_model.py` | E2.8 | 04, 08 | Method, Experiments |
| Defaults $(\alpha,\delta,\rho,\theta_{\max})$ | — | `Gimbal.__init__` | E2.9 | 03–07 | Hyper-parameter table |
| Peer set and their rules (incl. C-011, KL-SOAP initialization) | — | `src/gimbal/torch/*` | E2.1–E2.5, E3.1–E3.2 | 03–08 | Baselines |
