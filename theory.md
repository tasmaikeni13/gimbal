# The theory behind Gimbal

This note explains the ideas and the equations behind Gimbal in one pass. The full statements,
proofs and evidence labels are in [`theory/gimbal_theory.md`](theory/gimbal_theory.md); the
machine-checked parts are in [`formal/`](formal/README.md) (Lean 4 + Mathlib, 52 theorems).

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
eigenvectors of the pooled Kronecker factors $E[GG^\top]$ and $E[G^\top G]$. Those factors are the
maximum-likelihood statistics of a different model: the separable one, $D=\lambda\mu^\top$
(Shampoo's Kronecker *product* covariance). The frame and the variances come from two different
models. When the gradient really is separable SOAP is fine, but then Shampoo would be enough. SOAP
helps exactly when $D$ is *not* separable, and that is where its frame estimator is weakest.

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
after the warm start (section 8).

**What the flow does near the answer.** At the true frame the expected score is zero (Theorem 4.1,
for any weights). Near it, the expected score is Fisher × displacement, so the mean step contracts
every pair by the same factor $1-\alpha$, **independent of the eigen-gap** (Theorem 4.2). Power
iteration (SOAP's refresh) contracts pair $(i,k)$ by the eigenvalue ratio, which is slow precisely
for close eigenvalues. The stochastic recursion has stationary angle variance
$\frac{\alpha}{2-\alpha}\cdot\frac1{F_{ik}}$: the Cramér–Rao variance at an effective sample size of
$(2-\alpha)/\alpha$ gradients (Theorem 4.3).

**Schedule.** Gimbal uses $\alpha_t=\alpha/(1-(1-\alpha)^t)$. With this schedule the recursion
$\theta_{t+1}=\theta_t+\alpha_{t+1}(x_t-\theta_t)$ produces exactly the bias-corrected exponential average of its
inputs (Theorem 4.4, machine-checked). Early in training the frame is the plain average of all
per-gradient estimates so far; later it is a tracker with memory $\approx 2/\alpha$. It is the same
identity that justifies Adam's bias correction.

## 6. The variances that weight the score

The score needs $D$, and $D$ must itself be estimated. Three facts decide how.

**Profile likelihood with one forgetting factor (Lemma 5.4.1).** With forgetting weights
$\beta^{t-s}$, the weighted likelihood is maximized over $D$ exactly by the bias-corrected EMA of
$Z^{\circ 2}$ with coefficient $\beta$. Gimbal therefore keeps a separate average $V^F$ with
$\beta_D = 1-\alpha$, the frame's own memory, instead of Adam's short $\beta_2$.

**The exact price of noisy weights (Lemma 5.4.2).** For any weights $w$, with the inner product
$\langle u,u'\rangle=\sum_j D_{ij}D_{kj}u_ju'_j$,

$$V_w = \frac{1}{F_{ik}} + \frac{\|w_\perp\|^2}{\langle w, w^\star\rangle^2},$$

where $w_\perp$ is the part of $w$ orthogonal to the efficient weights. If the plug-in variances have
relative noise $\bar\varepsilon^2$, the variance is inflated by about

$$1+\bar\varepsilon^2\Big(1+\frac{2n}{F_{ik}}\Big).$$

Pairs with similar profiles (small $F_{ik}$) suffer most. At first order the plug-in is free: the
frame score is odd in the entries and the variance score even, so the Fisher information is
block-diagonal. At finite memory it is not free. For Gaussian entries an EMA has
$\bar\varepsilon^2 = 2(1-\beta)/(1+\beta)$: 0.051 for Adam's $\beta_2=0.95$, 0.020 for $\beta_D=0.98$.

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

## 7. Invariances and guarantees

* **Equivariance** (Theorem 6). Rotating every gradient by $(P,R)$ and the initial frame accordingly
  rotates every update by the same $(P,R)$. SOAP and Shampoo share this property; AdamW does not.
* **Scale invariance** (Theorem 8.2). Multiplying all gradients by $c$ leaves the frame dynamics
  unchanged; the shrinkage factor is scale-free as well.
* **Descent** (Theorem 8.1). For $\beta_1=0$, $\langle G, U\rangle = \sum Z_{ij}^2/(\sqrt{\hat V_{ij}}+\epsilon)\ge 0$.
* **Second-moment transport** (Theorem 7). When the frame turns by $P$, a diagonal second moment is
  carried by the doubly stochastic matrix $P\odot P$. Its Kronecker form needs two matmuls, and it
  preserves the total.

## 8. Warm start: a one-step estimator

For the first $T_w=50$ steps Gimbal also accumulates the pooled factors. At step $T_w$ it restarts
the frame from their eigenvectors, transports the averages to the new frame, frees the factors,
and continues the flow. This is Le Cam's one-step construction: a consistent but inefficient
preliminary estimator followed by Fisher-scoring steps, which is efficient. With the bias-corrected
schedule the restart value enters the frame's average with exactly the weight $1-\beta^{T_w}$ of the
gradients it replaces (Remark 5.6, machine-checked). The reason for the warm start is finite-sample:
the local theory needs a start inside the linear regime, and the frame of a single gradient is far
from it.

## 9. Cost

Per step, for an $m\times n$ matrix, Gimbal needs:

* four matmul "units" of $m^2n+mn^2$: rotate the gradient and the momentum, rotate the update back,
  and compute the score;
* every $k$ steps (`frame_every`), the Fisher matrix plus $O(m^3+n^3)$ for the retraction and polish;
* $O(mn)$ elementwise work.

There is no QR or eigendecomposition after step 50. Its state is $m^2+n^2+4mn$ floats (two frames,
momentum, Adam's second moment, the flow's variance average and its odd-step half). That is never
more than SOAP's $2m^2+2n^2+2mn$, because $2mn\le m^2+n^2$. For the 125M configuration, the cost
model in `experiments/phase2/e28_cost_model.py` gives, as a percentage of forward plus backward
compute:

| | SOAP (f=10) | SOAP real-time | KL-SOAP | Gimbal k=1 | Gimbal k=4 | Gimbal k=10 |
|---|---|---|---|---|---|---|
| matmul | 0.71% | 1.57% | 1.94% | 1.61% | 0.76% | 0.60% |
| plus QR/eigh | yes, every 10 steps | every step | every step | none | none | none |

QR and eigendecomposition run far below matmul throughput on TPUs, which is where the
matmul-only design pays.

## 10. Where the peers sit

| Method | Frame from | Frame efficiency | Eigenvalues |
|---|---|---|---|
| AdamW | identity | none | free (Adam) |
| Shampoo, KL-Shampoo | pooled / KL factors | KL efficient only if separable | separable |
| SOAP | pooled factors, refreshed every $f$ steps by power iteration + QR | loses $n/n_{\mathrm{eff}}$ even if separable; ties unidentified | free (Adam) |
| KL-SOAP | KL factors | efficient only if separable | free (Adam) |
| Muon, NorMuon | singular vectors of the momentum, instantaneous | no averaging | all set to one (NorMuon: per-row normalization) |
| ARO | loss-driven rotation | not a covariance fit | base optimizer |
| **Gimbal** | likelihood flow with empirical-Bayes variances | efficient in both regimes; identifies ties | free (Adam) |

## 11. Where the idea comes from

The generator $E_L = Z\,g(Z)^\top - g(Z)\,Z^\top$ with $g(z)=z/D$ is the rotation part of the
EASI relative-gradient update for source separation (Cardoso and Laheld, 1996), with the score of
a heteroscedastic Gaussian as the non-linearity. Fisher normalization is Amari's natural gradient.
The identifiability condition (profiles must differ) is Pham and Cardoso's condition for
separating non-stationary sources, with the other tensor mode playing the role of time. What
transfers is the estimating equation; what does not is the independence of sources, which is why
everything here is stated for the working likelihood. The variance shrinkage of section 6 transfers
core shrinkage (Hoff, McCormack and Zhang, 2023) from matrix-variate statistics.

## 12. What is proved, what is measured, what is open

* **Machine-checked** (Lean): the efficiency inequality and its equality cases, KL-Shampoo's
  efficiency under separability, SOAP's loss factor, the strict separation example, Fisher
  positivity and the tie example, the retraction and polish identities, transport, equivariance
  and descent algebra, scale invariance, the bias-corrected schedule, the profile likelihood, the
  excess-variance identity, the optimal shrinkage factor, and the warm-start weighting.
* **Proved on paper**: asymptotic normality of factor estimators (delta method), the local
  contraction (Fisher identity), orthogonality of frame and variance scores.
* **Measured** (Phase 02, `experiments/phase2/`): frame efficiency against every peer on synthetic
  KRD streams (separable and non-separable, ties, drift, heavy tails), optimization on noisy
  quadratics, gradients of a real language model, a small-LM benchmark, and cost.
* **Open**: global convergence of the flow (Conjecture 4.1); a Student-$t$ score for heavy tails;
  richer shrinkage targets; the interaction of the frame flow with momentum.
