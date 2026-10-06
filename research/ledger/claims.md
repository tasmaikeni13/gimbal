# Claim–evidence matrix

| C-ID | Claim | Direct evidence | Controls / alternatives | Scope | Effect / uncertainty | Confidence | Allowed wording |
|---|---|---|---|---|---|---|---|
| C1 | SOAP estimates its frame under a different model (Kronecker product) than its eigenvalues (free diagonal) | SOAP paper (Alg. 3), KL-SOAP paper Claim 5 | — | definitional | — | high | "SOAP's frame is the maximum-likelihood frame of the separable model, not of its own preconditioner model" |
| C2 | Every single-factor frame estimator is at or above the Cramér–Rao bound for each pair, with strict separation on non-separable arrays | Lean `weighted_ge_mle`, `factor_estimators_strictly_inefficient` | Monte Carlo E1.1 | asymptotic, KRD model | 8.7× (pooled) and 46× (one weighting) above the bound in E1.1's instance | machine-checked (inequality); L3 (asymptotics) | "provably less efficient in the working model" |
| C3 | The likelihood flow attains the bound locally and is identifiable whenever profiles differ | Lean `mle_weights_attain`, `fisher_pos_iff`, `tie_but_identifiable`; E1.1 | — | local, KRD model | — | high (L3–L5) | "identifies frames that pooled factors cannot" |
| C4 | Gimbal beats SOAP/peers in optimization | — (Phase 02 pending) | — | — | — | — | not yet allowed |
