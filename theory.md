# The theory behind Gimbal

This note explains the ideas and the equations behind Gimbal in one pass. The full statements,
proofs and evidence labels are in [`theory/gimbal_theory.md`](theory/gimbal_theory.md); the
machine-checked parts are in [`formal/`](formal/README.md) (Lean 4 + Mathlib, 53 theorems).

## 1. What SOAP does, and the inconsistency inside it

For a weight matrix $W\in\mathbb R^{m\times n}$ with stochastic gradient $G$, SOAP keeps an orthogonal
frame $(Q_L,Q_R)$, rotates the gradient into it,

$$Z = Q_L^\top G\, Q_R ,$$

runs Adam on $Z$ (a free per-entry second moment $V\approx E[Z\odot Z]$), and rotates the step back.
The preconditioner family is therefore

$$\mathcal P(U,D):\quad Y\;\mapsto\; Q_L\big((Q_L^\top Y Q_R)\oslash\sqrt D\big)Q_R^\top ,\qquad U=(Q_L,Q_R),\ D\in\mathbb R^{m\times n}_{>0}.$$

As a model of the gradient's second moment, this family says: *in some Kronecker frame the gradient
entries are uncorrelated, with arbitrary variances $D_{ij}$.* We call it the **KRD model**
(Kronecker-rotated diagonal). Its covariance is

$$C(U,D) = (Q_L\otimes Q_R)\,\mathrm{diag}(\mathrm{vec}\,D)\,(Q_L\otimes Q_R)^\top .$$

SOAP estimates $D$ under this model (Adam's second moment), but it picks the frame from the
eigenvectors of the pooled Kronecker factors $E[GG^\top]$ and $E[G^\top G]$. Those factors are
statistics of a different model: the separable one, $D=\lambda\mu^\top$ (Shampoo's Kronecker
*product* covariance, whose two-sided maximum-likelihood rule is KL-Shampoo's). The frame and the
variances come from two different models. When the gradient really is separable the two models
agree, but then Shampoo's own preconditioner would be enough. SOAP helps exactly when $D$ is *not*
separable, and that is where its frame estimator is weakest.

Gimbal estimates the frame under the same model as the variances.

## 2. The objective: how far the frame is from diagonalizing the gradient

For a frame $U$ write $d_U = E[Z_U(G)^{\circ 2}]$, the variances that Adam would see in that frame.
Minimizing the Gaussian KL from the true second moment $S$ to the model over $D$ gives $D=d_U$ and
leaves a function of the frame alone (Proposition 1):

$$\min_{D}\ \mathrm{KL}\big(\mathcal N(0,S)\,\|\,\mathcal N(0,C(U,D))\big)
= J(U) = \tfrac12\Big(\sum_{ij}\log (d_U)_{ij}-\log\det S\Big)\ \ge 0 .$$

$J(U)=0$ exactly when $S$ is diagonal in the frame $U$. $J$ is the classical joint-diagonalization
criterion of Pham (2001), restricted to Kronecker frames, and in statistics this is a common
principal components problem (Flury, 1984). We use $J$ as the yardstick for frame quality
throughout: it is the excess KL a method pays for its frame alone, before any eigenvalue noise.

## 3. Identifiability: profiles versus sums

Inside the KRD model the pooled factors are

$$E[GG^\top] = Q_L\,\mathrm{diag}(r)\,Q_L^\top,\quad r_i=\sum_j D_{ij},\qquad
E[G^\top G] = Q_R\,\mathrm{diag}(c)\,Q_R^\top,\quad c_j=\sum_i D_{ij}.$$

So pooled factors can only separate rows $i$ and $k$ whose **sums** differ. The likelihood does
better. Parametrize a small rotation of the pair $(i,k)$ by an angle $\theta_{ik}$. Its Fisher
information per gradient is

$$F_{ik} = \sum_j \frac{(D_{ij}-D_{kj})^2}{D_{ij}\,D_{kj}},$$

which is positive as soon as the two variance **profiles** $D_{i\cdot}$ and $D_{k\cdot}$ differ
anywhere (Theorem 2). Rows $(2,1)$ and $(1,2)$ have equal sums, so pooled factors cannot tell
them apart, but $F=1>0$. For such a tie the pooled frame is arbitrary in that plane, and the cost of
a bad choice is unbounded (Example 1).

## 4. Efficiency: why one factor matrix cannot be optimal

Any "weighted factor" frame estimator (SOAP: weights $w\equiv1$; KL-Shampoo: weights $1/\mu$) has
asymptotic angle variance

$$V_w(i,k)=\frac{\sum_j w_j^2 D_{ij}D_{kj}}{\big(\sum_j w_j (D_{ij}-D_{kj})\big)^2}\ \ge\ \frac{1}{F_{ik}},$$

by Cauchy–Schwarz, with equality only for the **pair-dependent** weights

$$w_j^{\star(ik)} \propto \frac1{D_{kj}}-\frac1{D_{ij}} .$$

One weight vector serves every pair only when $1/D$ is additive in a specific sense, which holds
for separable $D$ (so KL-Shampoo is efficient there) and fails for general non-separable arrays.
There are explicit arrays on which every single positive weighting is strictly worse than the
Cramér–Rao bound for several pairs at once (Theorem 3). Under separability SOAP's own loss factor
is $n\sum_j\mu_j^2/(\sum_j\mu_j)^2$: about 28 for a $1/j$ spectrum with $n=768$.

The maximum-likelihood estimator applies the pair-dependent weights automatically. The question is
how to compute it online, cheaply, without eigendecompositions.

## 5. The Gimbal flow: one natural-gradient step on the rotation groups

The score of the log-likelihood with respect to the rotation of rows $i,k$ is
$E^L_{ik}=\sum_j Z_{ij}Z_{kj}\,(1/D_{kj}-1/D_{ij})$: exactly the efficient weights above. In matrix form,
with $A = D^{\circ -1}$,

$$S_L = Z\,(Z\odot A)^\top,\qquad E_L = S_L - S_L^\top ,$$

$$F_L = D A^\top + A D^\top - 2n\,\mathbf 1\mathbf 1^\top \quad\text{(entrywise, this is } F_{ik}\text{)} ,$$

because $x/y + y/x - 2 = (x-y)^2/(xy)$. Gimbal takes a damped natural-gradient step on $O(m)$:

$$\Omega_L = -\alpha_t\; E_L \oslash \big(F_L + \delta n\big),\qquad
Q_L \leftarrow \mathrm{polish}\Big(Q_L\big(I + \Omega_L + \tfrac12\Omega_L^2\big)\Big),$$

and the same on the right with $S_R = Z^\top (Z\odot A)$, $F_R = D^\top A + A^\top D - 2m$. Dividing by
the Fisher information is Amari's natural gradient; $\delta$ is a Levenberg–Marquardt damping for
nearly degenerate pairs. The generator is skew-symmetric, so the update is a rotation:

* $X = I+\Omega+\Omega^2/2$ satisfies $X^\top X = I + \Omega^4/4$ (orthogonal to fourth order).
* One Newton–Schulz step $Y=\tfrac12X(3I-X^\top X)$ turns a defect $E$ into $-\tfrac34E^2+\tfrac14E^3$.
* A spectral cap $\|\Omega\|_2\le 1$ keeps the iterate inside the polish's convergence region.

Everything is matrix multiplication. There are no factor buffers, and no QR or eigendecomposition
after the warm start (section 9).

**What the flow does near the answer.** At the true frame the expected score is zero (Theorem 4.1,
for any weights). The expected score is also the gradient of $J$, and the Hessian of $J$ at the true
frame is exactly the Fisher information, so near the answer the natural-gradient step is a Newton
step on $J$. Hence the mean step contracts every pair by the same factor $1-\alpha$, **independent
of the eigen-gap** (Theorem 4.2; both facts are confirmed numerically in E2.10). Power
iteration (SOAP's refresh) contracts pair $(i,k)$ by the eigenvalue ratio, which is slow precisely
for close eigenvalues. The stochastic recursion has stationary angle variance
$\frac{\alpha}{2-\alpha}\cdot\frac1{F_{ik}}$: the Cramér–Rao variance at an effective sample size of
$(2-\alpha)/\alpha$ gradients (Theorem 4.3).

**Schedule.** Gimbal uses $\alpha_t=\alpha/(1-(1-\alpha)^t)$. With this schedule the recursion
$\theta_{t+1}=\theta_t+\alpha_{t+1}(x_t-\theta_t)$ produces exactly the bias-corrected exponential average of its
inputs (Theorem 4.4, machine-checked). Early in training the frame is the plain average of all
per-gradient estimates so far; later it is a tracker with memory $\approx 2/\alpha$. It is the same
identity that justifies Adam's bias correction. The default $\alpha = 1-\beta_2 = 0.05$ gives the frame
the same effective sample size as Adam's second moment, $(2-\alpha)/\alpha = (1+\beta_2)/(1-\beta_2) = 39$
gradients, the convention SOAP uses for its factors. (The default was 0.02 until change C-018: a
100-step frame memory, chosen on stationary synthetic streams, lagged behind the drifting
statistics of a real language model, F-027.)

## 6. The variances that weight the score

The score needs $D$, and $D$ must itself be estimated. Three facts decide how.

**Profile likelihood with one forgetting factor (Lemma 5.4.1).** With forgetting weights
$\beta^{t-s}$, the weighted likelihood is maximized over $D$ exactly by the bias-corrected EMA of
$Z^{\circ 2}$ with coefficient $\beta$. Gimbal therefore keeps a separate average $V^F$ with
$\beta_D = 1-\alpha$, the frame's own memory, instead of Adam's $\beta_2$; with the default
$\alpha=1-\beta_2$ the two windows have the same length, but the flow's average is taken on the
innovation and in the frame the flow sees.

**The exact price of noisy weights (Lemma 5.4.2).** For any weights $w$, with the inner product
$\langle u,u'\rangle=\sum_j D_{ij}D_{kj}u_ju'_j$,

$$V_w = \frac{1}{F_{ik}} + \frac{\|w_\perp\|^2}{\langle w, w^\star\rangle^2},$$

where $w_\perp$ is the part of $w$ orthogonal to the efficient weights. If the plug-in variances have
relative noise $\bar\varepsilon^2$, the variance is inflated by about

$$1+\bar\varepsilon^2\Big(1+\frac{2n}{F_{ik}}\Big).$$

Pairs with similar profiles (small $F_{ik}$) suffer most. This is the leading term. The paper gives
the next one; for Gaussian errors it changes the excess by a relative amount of about
$\bar\varepsilon^2(6+3R_{ik}/F_{ik})$, and Monte Carlo confirms the corrected formula (E2.10 (b)).
At first order the plug-in is free: the frame score is odd in the entries and the variance score
even, so the Fisher information is block-diagonal. At finite memory it is not free. For Gaussian
entries an EMA has
$\bar\varepsilon^2 = 2(1-\beta)/(1+\beta)$: 0.051 for $\beta=0.95$ (Adam's $\beta_2$, and the flow's
$\beta_D$ at the default $\alpha=0.05$), 0.020 for $\beta=0.98$. The shrinkage below is what reduces
the plug-in noise at the default.

**Empirical-Bayes shrinkage toward the separable fit (Proposition 5.5).** Split
$\log V^F = A + \hat R$ into its additive fit $A_{ij}=r_i+c_j-\bar\ell$ (the separable model) and a
residual. Model the residual as true interaction $R$ plus sampling noise of energy $\nu$. Shrinking
by $c$ has risk $(c-1)^2\|R\|^2 + c^2\nu$, minimized exactly at $c^\star=\|R\|^2/(\|R\|^2+\nu)$
(machine-checked). Gimbal uses the positive-part plug-in

$$D = \kappa\,\exp\big(A + c\,\hat R\big),\qquad c = \Big(1-\frac{\hat\nu}{\|\hat R\|^2}\Big)_+ ,$$

with $\kappa$ restoring the mean. The noise $\hat\nu$ is measured, not assumed. $V^F$ is the sum of an
average over odd steps and one over even steps; the variance of the log-ratio of the two halves is
about four times the noise of the full average. This needs no distributional assumption, so heavy
tails raise $\hat\nu$ by themselves, and it cancels slow drift, which moves both halves alike.

The effect is that the estimator **chooses its own model**:

* On separable gradients $c\to0$, the weights become $\propto 1/\hat\mu_j$, and the flow behaves like
  KL-Shampoo's efficient estimator, with variances averaged over whole rows and columns.
* On strongly non-separable gradients $c\to1$ and the flow is the free maximum-likelihood
  estimator.

Shrinkage only changes variance, never bias: whatever the weights, the expected score vanishes at
the true frame. In both limits the variances are consistent, so the flow remains asymptotically
efficient. The principle, shrinking toward Kronecker separability by an amount the data decide,
is Hoff, McCormack and Zhang's core shrinkage for matrix-variate covariances (2023); here it is
applied to the diagonal core of the KRD model, on the log scale.

## 7. Gradients with a mean: fit the frame to the innovation

The likelihood so far treats the gradient as zero-mean noise. A real gradient also carries a mean:
the descent signal. Write $G_t=\mu_t+N_t$ with KRD noise $N_t$. Two facts follow (Proposition 5.7).

**The mean tilts the frame.** Subtract any estimate $\hat\mu_t$ formed from past gradients and let
$B_t$ be what remains of the mean, in rotated coordinates at the true frame. The expected score
there is

$$E\big[E^L\big]=B_t(B_t\odot A)^\top-(B_t\odot A)B_t^\top ,$$

because the noise part is diagonal and the cross terms average out. Without centering $B_t$ is the
whole mean, and the score is non-zero at the true frame: the rank-one signal pulls the estimate
toward its own direction. On a noisy quadratic at low noise this cost Gimbal its lead over KL-SOAP in
the separable configuration (F-020).

**How much of the momentum to subtract.** Using $c\,\hat M_{t-1}$, with $\hat M_{t-1}$ the previous
momentum, leaves $B=(1-c)\,\Theta-c\,\Xi$: the bias shrinks by $(1-c)^2$, and the momentum's own noise
$\Xi$ enters instead, with variance $\eta$ times the gradient noise, where

$$\eta=\sum_sw_s^2=\frac{(1-\beta_1)(1+\beta_1^{t-1})}{(1+\beta_1)(1-\beta_1^{t-1})}$$

is the momentum's sum of squared weights (machine-checked). The excess energy
$(1-c)^2\|\mu\|^2+c^2\eta\,\mathrm{tr}\,\Sigma$ is minimized exactly at the James–Stein factor
$c^\star=\|\mu\|^2/(\|\mu\|^2+\eta\,\mathrm{tr}\,\Sigma)$, the same machine-checked risk identity as in
section 6. Gimbal estimates it from quantities it already has:

$$c_t=\Big(1-\frac{\eta\,\widehat{\mathrm{tr}\,\Sigma}}{\|\hat M_{t-1}\|^2}\Big)_+,\qquad
\widehat{\mathrm{tr}\,\Sigma}=\frac{\sum\hat V_{t-1}-\|\hat M_{t-1}\|^2}{1-\eta},$$

using $E\|\hat M\|^2=\|\mu\|^2+\eta\,\mathrm{tr}\,\Sigma$ and $E\sum\hat V=\|\mu\|^2+\mathrm{tr}\,\Sigma$. Adam's step is
unchanged; only the frame's statistic (score, variance average, warm-start factors) uses
$G_t-c_t\hat M_{t-1}$.

* Zero-mean gradients give $c\approx0$: the statistic is the gradient, and the true frame stays a
  fixed point of the expected flow for any $c$.
* A dominant mean gives $c\to1$: the frame is fitted to the noise, which in the noisy quadratic
  model shares the curvature's frame.
* $c_t$ is invariant to rotations and to the gradient's scale, so equivariance and scale invariance
  survive.
* It is free: in a fixed frame $Z(\hat M_t)=\beta_1Z(\hat M_{t-1})+(1-\beta_1)Z(G_t)$, so rotating the
  previous momentum instead of the new one gives both.

Centring a second moment by the running mean is old (centred RMSProp, Graves 2013; AdaBelief, Zhuang
et al. 2020), always with $c=1$ and applied to the step's variances, and SR-Adam (2026) shrinks the
gradient toward the momentum by a Stein rule. Gimbal centres only the frame's
statistic, uses a predictable plug-in (which is what makes the cross terms vanish), and lets an
empirical-Bayes factor turn the centring off when no mean is detectable. Measured: with the change
Gimbal has the lowest final loss on the noisy quadratics in every configuration, including the
separable low-noise one it lost before (`experiments/phase2/results/e25_report_full_c013.md`),
while on zero-mean frame-estimation streams the frame KL moves by a few percent at most, in either
direction (`experiments/phase2/results/e21_report.md`, centering-effect tables).

## 8. Invariances and guarantees

* **Equivariance** (Theorem 6). Rotating every gradient by $(P,R)$ and the initial frame accordingly
  rotates every update by the same $(P,R)$. SOAP and Shampoo share this property; AdamW does not.
* **Scale invariance** (Theorem 8.2). Multiplying all gradients by $c$ leaves the frame dynamics
  unchanged; the shrinkage and centering factors are scale-free as well. In floating point this holds to rounding at
  every scale once no absolute constant enters a scale-free formula; the eigenvectors of a
  rank-deficient first-gradient factor are an arbitrary basis of its null space, so the starting
  frame itself is only defined up to that choice (as for SOAP).
* **Descent** (Theorem 8.1). For $\beta_1=0$, $\langle G, U\rangle = \sum Z_{ij}^2/(\sqrt{\hat V_{ij}}+\epsilon)\ge 0$.
* **Second-moment transport** (Theorem 7). When the frame turns by $P$, a diagonal second moment is
  carried by the doubly stochastic matrix $P\odot P$. Its Kronecker form needs two matmuls, and it
  preserves the total.

## 9. Warm start: a one-step estimator

For the first $T_w=50$ steps Gimbal also accumulates the pooled factors. At step $T_w$ it restarts
the frame from their eigenvectors, transports the averages to the new frame, frees the factors,
and continues the flow. This is Le Cam's one-step construction: a consistent but inefficient
preliminary estimator followed by Fisher-scoring steps, which is efficient. With the bias-corrected
schedule the restart value enters the frame's average with exactly the weight $1-\beta^{T_w}$ of the
gradients it replaces (Remark 5.6, machine-checked). The reason for the warm start is finite-sample:
the local theory needs a start inside the linear regime, and the frame of a single gradient is far
from it. One consequence: where pooled factors are wrong (tied sums), the restart value's weight
decays only like $T_w/t$ while the memory exceeds the elapsed time, so at a fixed horizon very long
memories pay a small bias; with the memory matched to the horizon the error still vanishes
(E2.2b).

## 10. Cost

Per step, for an $m\times n$ matrix, Gimbal needs:

* four matmul "units" of $m^2n+mn^2$: rotate the gradient and the momentum, rotate the update back,
  and compute the score;
* every $k$ steps, the Fisher matrix plus $O(m^3+n^3)$ for the retraction and polish. The default
  moves the frame on an adaptive schedule (change C-018): every step while the bias-corrected rate
  $\alpha_t$ is large, then every $K=4$ steps (from step 48 on at $\alpha=0.05$). The amortized step
  matches the per-step flow only to first order in $k\alpha_t$, and a fixed $k=4$ in the first steps
  cost the small language model a deficit it never recovered (F-027);
* $O(mn)$ elementwise work.

There is no QR or eigendecomposition after step 50. With $k=1$ its state is $m^2+n^2+4mn$ floats
(two frames, momentum, Adam's second moment, the flow's variance average and its odd-step half),
never more than SOAP's $2m^2+2n^2+2mn$ because $2mn\le m^2+n^2$. The default $k=4$ also keeps the
accumulated scores ($m^2+n^2$), which puts it $2mn$ per layer above SOAP (F-023, open). For the 125M
configuration, the cost model (`experiments/phase2/results/e28_cost_model_c013.json`) gives, as a
percentage of forward plus backward compute:

| | SOAP (f=10) | SOAP real-time | KL-SOAP | Gimbal k=1 | Gimbal k=4 | Gimbal k=10 |
|---|---|---|---|---|---|---|
| matmul | 0.71% | 1.57% | 1.94% | 1.61% | 0.76% | 0.60% |
| plus QR/eigh | yes, every 10 steps | every step | every step | none | none | none |

QR and eigendecomposition run far below matmul throughput on TPUs, which is where the
matmul-only design pays. Gimbal's steady state uses $k=4$: with $k=1$ the frame moves pay the $m^3$
terms of tall layers every step and the optimizer costs more than SOAP. CPU timings taken inside
the warm start are in `experiments/phase2/results/e28_cost_model.json`; a steady-state CPU benchmark
was not completed in Phase 02. On the TPU v4-32 slice (Phase 04, `docs/performance.md`) the
training step with Gimbal at $k=4$ is about 2% slower than with SOAP: the $2048\times2048$ retraction
and Newton–Schulz polish run at full float32 precision, which the TPU executes far below its
bfloat16 throughput.

## 11. Where the peers sit

| Method | Frame from | Frame efficiency | Eigenvalues |
|---|---|---|---|
| AdamW | identity | none | free (Adam) |
| Shampoo, KL-Shampoo | pooled / KL factors | KL efficient only if separable | separable |
| SOAP | pooled factors, refreshed every $f$ steps by power iteration + QR | loses $n/n_{\mathrm{eff}}$ even if separable; ties unidentified | free (Adam) |
| KL-SOAP | KL factors | efficient only if separable | free (Adam) |
| Muon, NorMuon | singular vectors of the momentum, instantaneous | no averaging | all set to one (NorMuon: per-row normalization) |
| ARO | loss-driven rotation | not a covariance fit | base optimizer |
| **Gimbal** | likelihood flow on the innovation, with empirical-Bayes variances | efficient in both regimes; identifies ties | free (Adam) |

## 12. Where the idea comes from

The generator $E_L = Z\,g(Z)^\top - g(Z)\,Z^\top$ with $g(z)=z/D$ is the rotation part of the
EASI relative-gradient update for source separation (Cardoso and Laheld, 1996), with the score of
a heteroscedastic Gaussian as the non-linearity. Fisher normalization is Amari's natural gradient.
The identifiability condition (profiles must differ) is Pham and Cardoso's condition for
separating non-stationary sources, with the other tensor mode playing the role of time. What
transfers is the estimating equation; what does not is the independence of sources, which is why
everything here is stated for the working likelihood. The variance shrinkage of section 6 transfers
core shrinkage (Hoff, McCormack and Zhang, 2023) from matrix-variate statistics.

## 13. What is proved, what is measured, what is open

* **Machine-checked** (Lean): the efficiency inequality and its equality cases, KL-Shampoo's
  efficiency under separability, SOAP's loss factor, the strict separation example, Fisher
  positivity and the tie example, the retraction and polish identities, transport, equivariance
  and descent algebra, scale invariance, the bias-corrected schedule, the profile likelihood, the
  excess-variance identity, the optimal shrinkage factor, the warm-start weighting, the momentum's
  noise factor, the frame KL of a single-pair rotation ($\le\tfrac12F\theta^2$), and the memory bound
  against SOAP.
* **Proved on paper**: asymptotic normality of factor estimators (delta method), the local
  contraction (Fisher identity), orthogonality of frame and variance scores, the second-order
  plug-in correction, the score bias of a plug-in mean.
* **Measured** (Phase 02, `experiments/phase2/report.md`; no model is trained in Phase 02): frame
  efficiency against every peer on synthetic KRD streams (separable and non-separable, ties,
  drift, heavy tails) and consistency in tied planes; optimization on noisy quadratics; agreement
  between the asymptotic formulas and simulation; the landscape of $J$ under the noise-free flow;
  floating-point behaviour; cost. Real language-model gradients and a small-LM benchmark come in
  Phase 03.
* **Open**: a proof of global convergence (Conjecture 4.1, which has numerical support: every
  random start reached the global minimum and constructed critical points are strict saddles); a
  Student-$t$ score for heavy tails; richer shrinkage targets; a mean that drifts within Adam's
  windows (the centering factor assumes it is stationary there).
