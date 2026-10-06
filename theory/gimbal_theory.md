# Gimbal: theory

Living theory document. Every result carries an evidence label from the `theory-research` skill:
**L1** observed numerically, **L2** conjectured with explicit domain, **L3** proved under
additional assumptions, **L4** proved as stated, **L5** machine-verified (Lean 4 + Mathlib, file
named). When a result is revised, its old statement moves to the changelog at the end and every
dependent result is re-checked (see `phases/README.md`, adaptive protocol).

## 0. Summary

SOAP preconditions a matrix gradient by running Adam in a rotated frame
$(Q_L, Q_R)$, i.e. with the preconditioner family

$$\mathcal P(U,D):\; Y \mapsto Q_L\big(Z_U(Y) \oslash \sqrt{D}\big) Q_R^\top,\qquad Z_U(Y)=Q_L^\top Y Q_R,$$

where $D>0$ is a *free* $m\times n$ array of variances. SOAP estimates $D$ under exactly this model
(Adam's second moment) but estimates the frame from pooled Kronecker factors $E[GG^\top]$,
$E[G^\top G]$, which is the maximum-likelihood recipe for a different model (a Kronecker *product*
covariance, $D=\lambda\mu^\top$). Gimbal estimates the frame under SOAP's own model. The
maximum-likelihood frame is a joint diagonalization (common principal components) problem; Gimbal
solves it online with a natural-gradient flow on $O(m)\times O(n)$ that uses only matrix
multiplications and, after a short warm start (two symmetric eigendecompositions in total, at steps
1 and $T_w=50$), needs no Kronecker factor buffers and no QR or eigendecomposition. The theory below
shows the estimator is identifiable in strictly more situations than pooled
factors (Theorem 2), statistically efficient where every single-factor estimator is not (Theorem 3),
locally contracting at a gap-independent rate (Theorem 4), orthogonality-preserving (Theorem 5),
equivariant (Theorem 6), and scale-invariant (Theorem 8). The variances that weight the flow's score
are estimated with the frame's own memory and shrunk toward their separable (Kronecker) fit by an
empirical-Bayes factor (Lemma 5.4, Proposition 5.5), so the estimator behaves like KL-Shampoo's
efficient estimator on separable spectra and like the free maximum-likelihood estimator on
non-separable ones, choosing between them from the measured non-separability. The frame is fitted to
the gradient's innovation with respect to the momentum, scaled by a second empirical-Bayes factor
(Proposition 5.7): a deterministic descent signal does not tilt the frame, and zero-mean gradients
are used unchanged.

## 1. Setting and notation

* $W, G \in \mathbb R^{m\times n}$: weight and stochastic gradient of one matrix-shaped parameter.
* $\mathcal G = O(m)\times O(n)$ acts on $\mathbb R^{m\times n}$ by $(P,R)\cdot X = PXR^\top$.
* For $U=(Q_L,Q_R)\in\mathcal G$: $Z_U(X) = Q_L^\top X Q_R$ ("rotated coordinates").
* $\odot,\oslash$: entrywise product and division; $A^{\circ 2}=A\odot A$.
* **KRD model** $\mathrm{KRD}(U,D)$, $D\in\mathbb R^{m\times n}_{>0}$: the centred Gaussian law under which
  the entries of $Z_U(X)$ are independent with $E[Z_U(X)_{ij}^2]=D_{ij}$. Its covariance (row-major
  vectorization) is $C(U,D)=(Q_L\otimes Q_R)\,\mathrm{diag}(\mathrm{vec}\,D)\,(Q_L\otimes Q_R)^\top$.
* **Separable sub-model**: $D=\lambda\mu^\top$ (Kronecker product covariance: Shampoo, K-FAC,
  KL-Shampoo).
* Adam's step uses the *uncentred* second moment, as Shampoo, SOAP and Adam do. The frame is fitted
  to the innovation $G_t-c\,\hat M_{t-1}$ (Proposition 5.7, change C-013), which is the uncentred
  model when the gradient's mean is small against its noise ($c\approx0$). The Gaussian is a working
  likelihood (a surrogate fixing the estimating equation), not a claim that gradients are Gaussian.

## 2. The objective behind SOAP's own preconditioner

**Proposition 1 (profile KL; L4, per-coordinate minimization L5: `eigenvalue_cost_nonneg`,
`eigenvalue_cost_eq_iff`; Hadamard's inequality cited).** Let $S\succ0$ be the second-moment operator of $\mathrm{vec}\,G$.
For a frame $U$ let $d_U\in\mathbb R^{m\times n}$, $(d_U)_{ij}=E_S[Z_U(G)_{ij}^2]$. Then
$$\min_{D>0}\mathrm{KL}\big(\mathcal N(0,S)\,\|\,\mathrm{KRD}(U,D)\big)
= J(U) := \tfrac12\Big(\sum_{ij}\log (d_U)_{ij} - \log\det S\Big)\ \ge 0,$$
attained at $D=d_U$, with $J(U)=0$ iff $S=C(U,d_U)$.

*Proof.* $\mathrm{KL}(\mathcal N(0,S)\|\mathcal N(0,C)) = \tfrac12[\mathrm{tr}(C^{-1}S)-mn+\log\det C-\log\det S]$.
With $\tilde U=Q_L\otimes Q_R$ orthogonal, $\mathrm{tr}(C^{-1}S)=\sum_{ij}(d_U)_{ij}/D_{ij}$ and
$\log\det C=\sum_{ij}\log D_{ij}$; each term is minimized at $D_{ij}=(d_U)_{ij}$. The remainder is
$J(U)$. Non-negativity and the equality case are Hadamard's inequality applied to
$\tilde U^\top S\tilde U$, whose diagonal is $\mathrm{vec}\,d_U$ and whose determinant is $\det S$. ∎

Hence the KL-best member of SOAP's preconditioner family uses $U^\star\in\arg\min J$ and
$D=d_{U^\star}$: the frame that makes the second moment "as diagonal as possible" in the
log-determinant sense (Pham's joint-diagonalization criterion, restricted to Kronecker frames).
Adam's EMA of $Z_U(G)^{\circ2}$ estimates $d_U$ for whatever frame is used; the frame is the part
SOAP gets from a different model.

**Proposition 2 (what pooled factors estimate; L4).** If $S=C(U^\star,D)$ then
$L:=E[GG^\top]=Q_L^\star\,\mathrm{diag}(r)\,Q_L^{\star\top}$ with $r_i=\sum_jD_{ij}$ and
$R:=E[G^\top G]=Q_R^\star\,\mathrm{diag}(c)\,Q_R^{\star\top}$ with $c_j=\sum_iD_{ij}$.

*Proof.* $GG^\top = Q_L^\star ZZ^\top Q_L^{\star\top}$ with $Z=Z_{U^\star}(G)$, and
$E[(ZZ^\top)_{ik}]=\sum_jE[Z_{ij}Z_{kj}]=\delta_{ik}r_i$ by independence. Same on the right. ∎

So pooled factors are consistent inside the KRD model **when the row sums $r$ (column sums $c$) are
distinct**. When $r_i=r_k$ the pooled eigenbasis is not identified in the plane of
$q_i,q_k$ and any rotation there is an equally valid "Shampoo eigenbasis". The next theorem shows
the likelihood still identifies it, and Example 1 shows how much an arbitrary choice can cost.

**Example 1 (unbounded cost of a pooled tie; L4).** $m=n=2$, $D=\begin{pmatrix}a&b\\b&a\end{pmatrix}$,
$a\ne b$. Pooled row sums tie ($r_1=r_2=a+b$), so the pooled frame may be rotated by $45^\circ$
in the left plane. There the diagonal model sees variances $(a+b)/2$ for both rotated rows of each
column and $J = 2\log\frac{(a+b)/2}{\sqrt{ab}}\to\infty$ as $a/b\to\infty$, while
$J(U^\star)=0$. (Lean, L5: `tie_cost_pos`, `tie_cost_unbounded`; Monte Carlo: pooled frames spread over
≈0.47 rad in the tied plane while the likelihood flow converges to within 0.3°.)

## 3. Identifiability and information

Parametrize the frame near $U^\star$ by $Q_L=Q_L^\star e^{\Omega}$, $Q_R=Q_R^\star e^{\Psi}$ with
$\Omega,\Psi$ skew; coordinates $\theta_{ik}=\Omega_{ik}$ ($i<k$), $\phi_{jl}=\Psi_{jl}$ ($j<l$).
Write $\ell=\tfrac12\sum_{ij}[\log D_{ij}+Z_{ij}^2/D_{ij}]$ for the negative log-likelihood of one
observation.

**Lemma A (score; L4).** At $U^\star$,
$\partial\ell/\partial\theta_{ik} = E^L_{ik} := \sum_j Z_{ij}Z_{kj}\big(D_{kj}^{-1}-D_{ij}^{-1}\big)$
and $\partial\ell/\partial\phi_{jl} = E^R_{jl} := \sum_i Z_{ij}Z_{il}\big(D_{il}^{-1}-D_{ij}^{-1}\big)$.
Both $E^L$ and $E^R$ are skew-symmetric. In matrix form, with $A=D^{\circ-1}$:
$E^L = S_L-S_L^\top$, $S_L=Z(Z\odot A)^\top$; $E^R=S_R-S_R^\top$, $S_R=Z^\top(Z\odot A)$.

*Proof.* Under $Q_L\mapsto Q_L(I+\Omega)$, $Z\mapsto(I-\Omega)Z$ to first order, so
$d\ell[\Omega]=-\sum_{ik}\Omega_{ik}\sum_j Z_{ij}Z_{kj}/D_{ij} = \tfrac12\langle\Omega,E^L\rangle_F$
using $\Omega^\top=-\Omega$. Setting $\Omega_{ik}=\theta=-\Omega_{ki}$ gives
$\partial\ell/\partial\theta_{ik}=E^L_{ik}$. The right side is identical with $Z\mapsto Z(I+\Psi)$. ∎

**Theorem 2 (Fisher information and identifiability; L4 for the moment computation, L5 for the
finite-sum statements; Fisher value also checked by Monte Carlo).**
In $\mathrm{KRD}(U^\star,D)$ the Fisher information of one observation is diagonal in the rotation
coordinates, block-orthogonal to $D$, with entries
$$F^L_{ik}=\sum_j\frac{(D_{ij}-D_{kj})^2}{D_{ij}D_{kj}},\qquad
F^R_{jl}=\sum_i\frac{(D_{ij}-D_{il})^2}{D_{ij}D_{il}}.$$
Consequently the rotation of the pair $(i,k)$ is locally identifiable iff the variance **profiles**
differ, $D_{i\cdot}\ne D_{k\cdot}$; pooled factors identify it iff the **row sums** differ. The
first condition is implied by, and strictly weaker than, the second.

*Proof.* $E[(E^L_{ik})^2]=\sum_{j,j'}E[Z_{ij}Z_{kj}Z_{ij'}Z_{kj'}](\cdot)(\cdot)$; independence and
zero means kill $j\ne j'$, and $E[Z_{ij}^2Z_{kj}^2]=D_{ij}D_{kj}$ for $i\ne k$, giving
$\sum_jD_{ij}D_{kj}(D_{kj}^{-1}-D_{ij}^{-1})^2=F^L_{ik}$. A product of scores of two different
pairs, or of a rotation score with a $D$-score $\tfrac12(Z_{ij}^2/D_{ij}^2-1/D_{ij})$, contains an
entry to an odd power and has mean zero. Each summand of $F^L_{ik}$ is $\ge0$ and vanishes iff
$D_{ij}=D_{kj}$. If $\sum_jD_{ij}\ne\sum_jD_{kj}$ some $D_{ij}\ne D_{kj}$; the converse fails for
$D_{i\cdot}=(2,1)$, $D_{k\cdot}=(1,2)$. ∎
(Lean, L5: `fisher_pos_iff`, `pooled_implies_profile`, `tie_but_identifiable`, `score_variance_term`,
`fisher_term_identity`.)

## 4. Efficiency: why every single-factor estimator loses information

Consider $N$ i.i.d. observations from $\mathrm{KRD}(U^\star,D)$ and the **weighted-factor estimator**
with column weights $w\in\mathbb R^n_{>0}$ (expressed in the true right frame):
$\hat L_w = \frac1N\sum_t G_tQ_R^\star\mathrm{diag}(w)Q_R^{\star\top}G_t^\top$, $\hat Q_L=$ eigenvectors of
$\hat L_w$. Shampoo/SOAP use $w\equiv1$; KL-Shampoo uses $w=1/\hat\mu$ (its $R^{-1}$ weighting).

**Theorem 3 (efficiency; item 1 L3 — asymptotics by the delta method under distinct population
eigenvalues $\ell_i(w)=\sum_jw_jD_{ij}$; items 2–4 L5 except the necessity half of item 3, L4).**

1. $\sqrt N\,\hat\theta^{\,w}_{ik}\Rightarrow\mathcal N\big(0,V_w(i,k)\big)$ with
   $$V_w(i,k)=\frac{\sum_jw_j^2D_{ij}D_{kj}}{\big(\sum_jw_j(D_{ij}-D_{kj})\big)^2}.$$
2. $V_w(i,k)\ge 1/F^L_{ik}$ (the Cramér–Rao bound attained by the MLE), with equality iff
   $w_j\propto D_{kj}^{-1}-D_{ij}^{-1}$.
3. One weight vector is efficient for **all** pairs iff the inverse variances have the form
   $D^{\circ-1}=\mathbf 1\beta^\top+\alpha w^\top$ for some $\alpha\in\mathbb R^m,\beta\in\mathbb R^n$.
   * For separable $D=\lambda\mu^\top$ this holds with $w=1/\mu$, $\alpha=1/\lambda$, $\beta=0$: KL-Shampoo's
     weights. **KL-Shampoo is efficient exactly under separability.**
   * SOAP's $w\equiv1$ requires $D^{\circ-1}$ to be additive ($\alpha_i+\beta_j$). Under separability its
     efficiency loss is $V_1F=n\sum_j\mu_j^2/(\sum_j\mu_j)^2=n/n_{\mathrm{eff}}(\mu)$, which is large for
     steep spectra (for $\mu_j\propto1/j$, $n=768$: $n/n_{\mathrm{eff}}\approx28$).
4. The MLE needs **pair-dependent** weights $w^{(ik)}_j\propto D_{kj}^{-1}-D_{ij}^{-1}$, which attain
   $1/F$ exactly. No single factor matrix can supply them unless item 3 holds; there are
   non-separable arrays on which **every** positive weighting (SOAP's, KL-Shampoo's, any other) is
   strictly above the bound for several pairs at once — e.g. rows $(1,1),(2,1),(1,2)$, where any
   $w$ with $w_2>0$ is strictly inefficient for the pair 1–2 and any $w$ with $w_1>0$ for the pair
   1–3. Gimbal's generator applies exactly the pair-dependent weights (Lemma A), so its estimating
   equation is the likelihood equation.

*Proof.* (1) In the true left frame the off-diagonal entry $(i,k)$ of $Q_L^{\star\top}\hat L_wQ_L^\star$ is
$\frac1N\sum_t\sum_jw_jZ^{(t)}_{ij}Z^{(t)}_{kj}$ with variance $\frac1N\sum_jw_j^2D_{ij}D_{kj}$; first-order
eigenvector perturbation divides by the eigen-gap $\ell_i(w)-\ell_k(w)=\sum_jw_j(D_{ij}-D_{kj})$.
(2) Write $r_j=w_j(D_{ij}-D_{kj})$, $f_j=w_j^2D_{ij}D_{kj}$, $g_j=(D_{ij}-D_{kj})^2/(D_{ij}D_{kj})$. Then
$r_j^2=f_jg_j$ and Cauchy–Schwarz gives $(\sum r_j)^2\le(\sum f_j)(\sum g_j)$, i.e. $V_w\ge1/F$, with
equality iff $w_j\sqrt{D_{ij}D_{kj}}\propto(D_{ij}-D_{kj})/\sqrt{D_{ij}D_{kj}}$.
(3) Equality for all pairs means $D_{kj}^{-1}-D_{ij}^{-1}=c_{ik}w_j$; fixing $k=k_0$ gives
$D_{ij}^{-1}=D_{k_0j}^{-1}-c_{ik_0}w_j$, the stated form; the converse is direct. The separable
efficiency ratio follows by substitution. ∎
(Lean, L5: `weighted_cs`, `weighted_ge_mle`, `pooled_ge_mle` (item 2); `kl_weights_efficient_separable`,
`pooled_loss_separable`, `pooled_loss_ge_one`, `efficient_if_additive_inverse'` (item 3, sufficiency;
necessity is the written proof, L4); `mle_weights_attain`, `factor_estimators_strictly_inefficient`
(item 4). Item 1 is L3 and was checked by Monte Carlo: predicted vs. simulated $N\,\mathrm{Var}$ agree
within 2% for pooled and weighted factors, `experiments/phase1/check_identities.py`.)

**Corollary 3.1 (L3).** Under a basis that drifts at angular velocity $\omega$ per step in pair
$(i,k)$, an EMA estimator with memory $\tau$ has asymptotic tracking MSE
$\approx\omega^2\tau^2+V/\big((2\tau)\big)$ up to constants, with $V=V_w$ for factor methods and $V=1/F$
for the likelihood flow (Theorem 4). Since $V_w\ge1/F$, the best achievable tracking error of the
likelihood flow is no larger than that of any single-factor method at its best memory.

## 5. The Gimbal flow

Write $\tilde G_t=G_t-c_t\hat M_{t-1}$ for the innovation the frame is fitted to (Proposition 5.7;
$\tilde G_t=G_t$ before C-013) and, in this section, $Z=Z_U(\tilde G_t)$. Let $V^F$ be an exponential
moving average of $Z^{\circ2}$ with coefficient $\beta_D=1-\alpha$
(bias corrected; "tied" to the frame's memory, change C-003, Lemma 5.4), and let $D$ be its
empirical-Bayes shrinkage toward the separable fit (change C-004, Proposition 5.5), with a
relative floor $\rho$. Adam's own second moment $V$ (coefficient $\beta_2$) is used only for the
step. The rotation rate follows the
bias-corrected schedule $\alpha_t=\alpha/(1-(1-\alpha)^t)$, capped at $\alpha_{\max}=0.5$ (Theorem 4.4
explains why), with defaults $\alpha=0.02$, $\delta=0.003$ (change C-002). One step of the left flow:
$$\Omega_L=-\alpha_t\,\frac{S_L-S_L^\top}{F_L+\delta n},\qquad
F_L=DA^\top+AD^\top-2n\,\mathbf 1\mathbf 1^\top,\quad A=D^{\circ-1},$$
(entrywise division; diagonal set to zero; entries clipped to $[-\theta_{\max},\theta_{\max}]$; spectral norm
capped at $1$), then $Q_L\leftarrow \mathrm{polish}\big(Q_L\,(I+\Omega_L+\tfrac12\Omega_L^2)\big)$. The
right flow is symmetric with $S_R=Z^\top(Z\odot A)$, $F_R=D^\top A+A^\top D-2m$, damping $\delta m$. Note
$(DA^\top+AD^\top)_{ik}-2n=\sum_j(D_{ij}/D_{kj}+D_{kj}/D_{ij}-2)=F^L_{ik}$, so $F_L$ is the Fisher
information of Theorem 2 evaluated at the current estimates; dividing by it is Amari's natural
gradient, and $\delta$ is a Levenberg–Marquardt damping that bounds the step for near-degenerate
pairs.

**Amortized flow** (`frame_every` $=k$, default $k=4$ since C-012). The score $S-S^\top$ is
accumulated for $k$ steps with the frame fixed, and the frame then takes one step with the mean
score and the effective rate $1-\prod_{s}(1-\alpha_s)$ over those $k$ steps. Every gradient still enters the estimate; the
retraction, the polish and the Fisher matrix are paid once per $k$ steps. In the linear regime the
amortized and per-step recursions agree to first order in $k\alpha$ (L3); Monte Carlo (E2.1 pilot)
shows unchanged final frame quality for $k\in\{4,10\}$ and a slower first ~$200$ steps.

**Theorem 4 (fixed points, local rate, noise; item 1 L5, item 2 L3 (linearization around $U^\star$ with
$D$ known), item 3 L5 for the recursion and checked by Monte Carlo within 7%).**

1. If the rotated second moment has no within-column cross terms, $E[Z_{ij}Z_{kj}]=0$ for all
   $i\ne k$ and $j$, then $E[E^L]=0$: such frames are fixed points of the mean flow. This holds at
   $U^\star$ under KRD and, more generally, at every stationary point of $J$.
2. Near $U^\star$, $E[E^L_{ik}]=F^L_{ik}\theta_{ik}+O(\|\theta\|^2)$ (Fisher identity), so the undamped
   mean step is $\theta\mapsto(1-\alpha)\theta$ **for every pair simultaneously**, independent of the
   eigen-gap. The damped step contracts pair $(i,k)$ by $1-\alpha F_{ik}/(F_{ik}+\delta n)$.
3. With $\mathrm{Var}(E^L_{ik})=F^L_{ik}$, the linearized recursion
   $\theta_{t+1}=(1-\alpha)\theta_t+\alpha\xi_t$, $\mathrm{Var}(\xi_t)=1/F^L_{ik}$, has stationary variance
   $\frac{\alpha}{2-\alpha}\cdot\frac1{F^L_{ik}}$, i.e. the Cramér–Rao variance at an effective sample size
   $(2-\alpha)/\alpha$. The second-moment recursion $v\mapsto(1-\alpha)^2v+\alpha^2\sigma^2$ converges
   geometrically to this value.

*Proof.* (1) $E[E^L_{ik}]=\sum_jE[Z_{ij}Z_{kj}](D_{kj}^{-1}-D_{ij}^{-1})=0$. (2) The expected score at a
displaced parameter equals Fisher times displacement to first order; the natural-gradient step
divides by $F$. (3) Standard AR(1) algebra. ∎ (Lean, L5: `stationary_generator`,
`variance_recursion_closed_form`, `variance_recursion_tendsto`.)

**Theorem 4.4 (bias-corrected rotation schedule; L5).** For $\beta=1-\alpha\in[0,1)$ and inputs
$x_s$, the recursion $\theta_{t+1}=\theta_t+a_{t+1}(x_t-\theta_t)$ with $a_t=(1-\beta)/(1-\beta^t)$
satisfies $\theta_t=\big(\sum_{s<t}\beta^{t-1-s}(1-\beta)x_s\big)/(1-\beta^t)$, the bias-corrected
exponential average. In the linearized frame dynamics (Theorem 4.2) the inputs are the per-sample
natural-gradient estimates, so with Gimbal's schedule the frame error is the bias-corrected EMA of
i.i.d. estimates: a uniform (batch) average for $t\ll1/\alpha$ and a tracker with memory $\approx2/\alpha$
afterwards. This is the same identity that justifies Adam's bias correction. (Lean:
`bias_corrected_schedule`.) The cap $\alpha_{\max}$ makes the first step(s) more conservative than the
exact average.

**Lemma 5.4 (the variances seen by the flow; item 1 L5, item 2 L5 for the identity and L3 for the
expansion, checked by Monte Carlo in E2.10 (b), item 3 L3).** Fix a pair $(i,k)$ of rows (the right side is symmetric) and write
$\langle u,u'\rangle=\sum_jD_{ij}D_{kj}u_ju'_j$.

1. *Profile likelihood with forgetting.* For weights $w_s\ge0$ and a fixed frame, the weighted
   log-likelihood $\sum_sw_s\,\ell(G_s;U,D)$ is maximized over $D$, entry by entry, exactly at
   $D_{ij}=\sum_sw_sZ_{s,ij}^2/\sum_sw_s$. With $w_s=\beta^{t-s}$ this is the bias-corrected EMA with
   coefficient $\beta$. A flow driven by $V^F$ with $\beta_D=1-\alpha$ is therefore a stochastic
   approximation of the profile likelihood in which the frame and the variances share one forgetting
   factor (effective sample size $\approx(2-\alpha)/\alpha$ for both, Theorem 4.3). (Lean:
   `weighted_profile_variance`, `weighted_profile_variance_eq_iff`.)
2. *The cost of plug-in weights.* The flow's score for the pair uses column weights
   $w_j=1/\hat D_{kj}-1/\hat D_{ij}$; the efficient weights are $w^\star_j=1/D_{kj}-1/D_{ij}$, with
   $\|w^\star\|^2=F_{ik}$. For any weights, the variance of the weighted estimating equation is exactly
   $$V_w=\frac{\|w\|^2}{\langle w,w^\star\rangle^2}=\frac1{F_{ik}}+\frac{\|w_\perp\|^2}{\langle w,w^\star\rangle^2},$$
   where $w_\perp\perp w^\star$ is the part of $w$ orthogonal to the efficient weights (Lean:
   `excess_variance_identity`, `excess_component_orthogonal`). If $\hat D=D\odot(1+\varepsilon)$ with
   independent relative errors of variance $\bar\varepsilon^2$, independent of the current gradient,
   then to second order $E\|w_\perp\|^2=\bar\varepsilon^2(F_{ik}+2n-R_{ik})$ with
   $R_{ik}=\sum_j(D_{ij}-D_{kj})^2(D_{ij}^{-2}+D_{kj}^{-2})/F_{ik}\ge0$, so
   $$V/V_{\mathrm{CR}}=1+\bar\varepsilon^2\,\frac{K_{ik}}{F_{ik}}+\bar\varepsilon^4C_{ik}+O(\bar\varepsilon^6),
   \qquad K_{ik}=F_{ik}+2n-R_{ik}.$$
   Pairs whose profiles are similar ($F_{ik}\ll n$) lose most. For Gaussian relative errors the
   next coefficient is (v0.5, F-013), with $c_j=D_{ij}-D_{kj}$, $m_j=D_{ij}D_{kj}$, $g_j=c_j^2/m_j$,
   $$C_{ik}F_{ik}=6F+12n+2R+8+\frac{Q-18U+6nR-9R^2}{F}+\frac{18X-6Q-6W+6Z_1}{F^2},$$
   $Q=\sum g_j^2$, $U=\sum m_jc_j(D_{kj}^{-3}-D_{ij}^{-3})$, $X=\sum c_j^3(D_{kj}^{-3}-D_{ij}^{-3})$,
   $W=\sum g_jc_j^2(D_{ij}^{-2}+D_{kj}^{-2})$, $Z_1=\sum m_jc_j^2(D_{ij}^{-2}+D_{kj}^{-2})^2$ (pair indices
   dropped). It comes from the growth of $\mathrm{Var}(1/(1+\varepsilon))=\bar\varepsilon^2+8\bar\varepsilon^4$, the
   mean shift of the weights along $w^\star$, the fluctuation of the sensitivity $\langle w,w^\star\rangle$, and
   third- and fourth-moment cross terms; it vanishes for $n=1$, where the exact excess is 0. Its
   dominant part makes the relative correction to the leading term about
   $\bar\varepsilon^2(6+3R_{ik}/F_{ik})$, which is not negligible at $\bar\varepsilon^2=0.02$ (measured in
   E2.10 (b)).
3. *Orthogonality.* At the true frame the frame score is odd under the sign flip $Z_{i\cdot}\to-Z_{i\cdot}$
   while the variance score is even, so the Fisher information is block-diagonal between frame and
   variances. Estimating $D$ therefore costs nothing at first order; item 2 is a finite-memory
   (second-order) effect. For Gaussian entries an EMA with coefficient $\beta$ has
   $\bar\varepsilon^2=2(1-\beta)/(1+\beta)=2/N_{\mathrm{eff}}$: Adam's $\beta_2=0.95$ gives $0.051$, the tied
   $\beta_D=0.98$ gives $0.020$. For a 48-column layer and a pair with $F_{ik}=5$ the inflation is
   $\approx2.0$ versus $\approx1.4$.

*Proof.* (1) $\sum_sw_s\ell=-\tfrac12\sum_{ij}[W\log D_{ij}+\sum_sw_sZ_{s,ij}^2/D_{ij}]$ with
$W=\sum_sw_s$; each term is the per-coordinate cost of Proposition 1. (2) Since
$\sum_jw_j(D_{ij}-D_{kj})=\langle w,w^\star\rangle$, the variance formula of Theorem 3 is
$\|w\|^2/\langle w,w^\star\rangle^2$; decompose $w=cw^\star+w_\perp$. For the expansion,
$\delta w_j=\varepsilon_{ij}/D_{ij}-\varepsilon_{kj}/D_{kj}$ to first order, whose squared norm has mean
$\bar\varepsilon^2\sum_j(D_{kj}/D_{ij}+D_{ij}/D_{kj})=\bar\varepsilon^2(F_{ik}+2n)$; its component along
$w^\star$ contributes $\bar\varepsilon^2R_{ik}$. Exactly, $V_wF_{ik}-1=(d_2-d_1^2)/(1+d_1)^2$ with
$d_1=\langle\delta w,w^\star\rangle/F_{ik}$ and $d_2=\|\delta w\|^2/F_{ik}$; expanding $(1+d_1)^{-2}$ and taking
Gaussian moments of $1/(1+\varepsilon)$ through fourth order gives $C_{ik}$. (3) Sign symmetry of the Gaussian KRD law; adaptivity of
estimating equations under block-diagonal information (Newey and McFadden, 1994, §6). ∎

**Proposition 5.5 (empirical-Bayes variances; the risk identity L5, the noise estimate L3).** Write
$\log V^F=A+\hat R$, where $A_{ij}=r_i+c_j-\bar\ell$ is the additive (row plus column) fit and $\hat R$
the residual, and model $\hat R=R+E$: $R$ is the true interaction ($R=0$ iff $D$ is separable,
$D=\lambda\mu^\top$) and $E$ is sampling noise, independent of $R$, with energy $\nu$. Among the
estimates $A+c\hat R$, the risk $E\|c\hat R-R\|^2=(c-1)^2S+c^2\nu$ ($S=\|R\|^2$) is minimized exactly at
$c^\star=S/(S+\nu)$, with value $S\nu/(S+\nu)<\min(S,\nu)$ (Lean: `shrinkage_risk_ge`,
`shrinkage_risk_eq_iff`). Gimbal uses the positive-part plug-in
$$D=\kappa\,\exp\!\big(A+c\hat R\big),\qquad c=\Big(1-\frac{\hat\nu}{\|\hat R\|^2}\Big)_+,$$
with $\kappa$ restoring the mean of $V^F$. The noise is measured by an interleaved split: $V^F$ is the
sum of an average over odd steps and one over even steps; the two halves are nearly independent
and each has about twice the noise of the full average, so
$\hat\nu=\tfrac14\,mn\,(1-\tfrac1m)(1-\tfrac1n)\,\mathrm{Var}_{ij}\big(\log V^{\mathrm{odd}}_{ij}/W_{\mathrm{odd}}-\log V^{\mathrm{even}}_{ij}/W_{\mathrm{even}}\big)$
(the last two factors account for the degrees of freedom used by the additive fit). The split needs no
distributional assumption (heavy tails raise $\hat\nu$ by themselves) and cancels slow drift of $D$,
which affects both halves alike. Consequences:

* *Separable* $D$ ($S=0$): $c\to0$, and $D$ is the separable fit $\hat\lambda\hat\mu^\top$. Its weights are
  $\propto1/\hat\mu_j$, KL-Shampoo's efficient weights (Theorem 3.3), with $\hat\mu$ averaged over $m$ rows.
* *Strongly non-separable* $D$ ($S\gg\nu$): $c\to1$ and $D\to V^F$, the free maximum-likelihood
  variances.
* *Bias.* Whatever $D$, the weights depend on squared entries only, so the expected generator
  vanishes at the true frame (Theorem 4.1): shrinkage changes the frame estimate's variance, never
  its mean. In both limits $D$ is consistent, so by Lemma 5.4.3 the flow stays asymptotically
  efficient; at finite memory it shrinks the plug-in term of Lemma 5.4.2.
* *Invariances.* A common scale of $V^F$ cancels in $c$ and $\kappa$ (Theorem 8.2 holds); the fit acts on
  rotated coordinates and row/column means commute with permutations (Theorem 6 holds).

The cost is $O(mn)$ elementwise work (two logarithms, one exponential, row and column means) and one
extra $m\times n$ buffer for the odd-step average. *Source.* Shrinking toward Kronecker separability
by an empirical-Bayes factor that adapts to the degree of separability is the core-shrinkage idea
of Hoff, McCormack and Zhang (JRSSB 2023, arXiv:2207.12484) for matrix-variate covariances. Here
the core is diagonal (the KRD model), the shrinkage acts on log-variances, the noise level comes
from an interleaved split, and the result weights the frame's likelihood score.

**Remark 5.6 (pooled warm start as a one-step estimator; the weighting L5, the rest L3).** With
`init="pooled"`, Gimbal accumulates the pooled factors $\sum_{s\le T_w}G_sG_s^\top$ and
$\sum_{s\le T_w}G_s^\top G_s$ for $T_w=50$ steps (two temporary $m^2+n^2$ buffers), then replaces the frame by
their eigenvectors, transports $V$ and $V^F$ to the new frame (Theorem 7), frees the factors, and
skips the flow move of step $T_w$ (that gradient is already in the pooled estimate).
(i) This is Le Cam's one-step construction: a consistent preliminary estimator (pooled factors are
consistent wherever row and column sums are distinct, Theorem 2) followed by Fisher-scoring steps,
which is asymptotically efficient (van der Vaart, 1998, §5.7). (ii) Because the bias-corrected
schedule continues at $t=T_w$, the restart value carries exactly the weight $1-\beta^{T_w}$ that the first
$T_w$ per-sample estimates would have had in the linearized dynamics, and every later gradient enters
through the efficient score (Lean: `restart_closed_form`). (iii) The reason is finite-sample:
Theorem 4.2 is local, and a flow started from the frame of one gradient is far outside its linear
regime and slowed by the trust-region caps. Where pooled factors are not consistent (tied sums),
the warm start gains nothing for the tied pair, and the flow's contraction removes the initial error
at the rate of Theorem 4.2. (iv) A consequence of (ii): while the memory exceeds the elapsed time
($\alpha t\ll1$), the restart weight $(1-\beta^{T_w})\beta^{t-T_w}/(1-\beta^t)$ decays only like $T_w/t$, so an
inconsistent warm-start estimate leaves a bias of that order; for moderate $\alpha$ it decays
geometrically. At a fixed horizon the frame error is therefore U-shaped in the memory in tied
cells (observed in E2.2, F-016), and it still vanishes as the horizon grows with the memory matched
to it (E2.2b, `experiments/phase2/results/e22b_report.md`).

**Proposition 5.7 (the frame statistic of a gradient with a mean; items 1–2 and item 3 for a fixed
$c$ L4, the optimal factor and $\eta$ L5, the plug-in factor L3; checked by Monte Carlo in E2.10 (f),
effect measured in E2.5).** Let $G_t=\mu_t+N_t$, where $N_t$ is KRD
noise at the true frame $U^\star$ ($Z_{U^\star}(N_t)=D^{\circ1/2}\odot\varepsilon_t$, entries of $\varepsilon_t$
independent, zero-mean, unit-variance, independent over $t$) and the mean $\mu_t$ is predictable
(a function of the past). Write $\Theta_t=Z_{U^\star}(\mu_t)$ and let $\mathcal F_{t-1}$ be the past.
1. *Likelihood.* The Gaussian log-likelihood of $G_t$ under $\mathrm{KRD}(U,D)$ with mean $\mu_t$ is the
   KRD log-likelihood of $G_t-\mu_t$, so the score in $U$ at a given mean is Lemma A's score of the
   innovation. Fitting the frame to $G_t$ itself treats the mean as noise.
2. *Bias of a plug-in mean.* Let $\hat\mu_t$ be $\mathcal F_{t-1}$-measurable, $B_t=\Theta_t-Z_{U^\star}(\hat\mu_t)$,
   $Z=Z_{U^\star}(G_t-\hat\mu_t)$, and $A>0$ any fixed weight array. Then
   $$E\big[Z(Z\odot A)^\top-(Z\odot A)Z^\top\,\big|\,\mathcal F_{t-1}\big]=B_t(B_t\odot A)^\top-(B_t\odot A)B_t^\top ,$$
   because the noise term $E[W(W\odot A)^\top]$ ($W=D^{\circ1/2}\odot\varepsilon_t$) is diagonal and the cross
   terms have conditional mean zero. The uncentred statistic ($\hat\mu=0$, $B=\Theta$) therefore has a
   non-zero expected score at the true frame whenever $\Theta_t\ne0$: the rank-one "signal" term tilts
   the estimate away from $U^\star$, the mechanism of F-020.
3. *Momentum as the mean, and its factor.* Let $\mu_t=\mu$ over the momentum's window and
   $\hat\mu_t=c\,\hat M_{t-1}$, where $\hat M_{t-1}=\sum_sw_sG_s$ is the bias-corrected momentum over $t-1$
   gradients (weights $w_s=(1-\beta_1)\beta_1^{t-1-s}/(1-\beta_1^{t-1})$). Then $B=(1-c)\Theta-c\,\Xi$ with
   $\Xi=\sum_sw_s\,D^{\circ1/2}\odot\varepsilon_s$, whose entries are independent with variance $\eta D_{ij}$,
   $$\eta=\sum_sw_s^2=\frac{(1-\beta_1)(1+\beta_1^{t-1})}{(1+\beta_1)(1-\beta_1^{t-1})}$$
   (Lean: `ema_weight_sq_sum`). For a fixed $c$, averaging item 2 over the momentum's noise as well
   gives $(1-c)^2\big(\Theta(\Theta\odot A)^\top-(\Theta\odot A)\Theta^\top\big)$, the uncentred bias scaled by
   $(1-c)^2$, since $E[\Xi(\Xi\odot A)^\top]$ is diagonal; the price is noise in the statistic, which
   becomes serially correlated through $\Xi$. The excess second moment of the statistic,
   $E\|B\|^2=(1-c)^2\|\mu\|^2+c^2\eta\,\mathrm{tr}\,\Sigma$ ($\mathrm{tr}\,\Sigma=\sum_{ij}D_{ij}$), is minimized exactly at
   $c^\star=\|\mu\|^2/(\|\mu\|^2+\eta\,\mathrm{tr}\,\Sigma)$ (Lean: `shrinkage_risk_eq_iff` with $S=\|\mu\|^2$,
   $\nu=\eta\,\mathrm{tr}\,\Sigma$). Gimbal uses the positive-part plug-in
   $$c_t=\Big(1-\frac{\eta\,\widehat{\mathrm{tr}\,\Sigma}}{\|\hat M_{t-1}\|^2}\Big)_+,\qquad
   \widehat{\mathrm{tr}\,\Sigma}=\frac{\big(\textstyle\sum_{ij}\hat V_{t-1}-\|\hat M_{t-1}\|^2\big)_+}{1-\eta},$$
   from $E\|\hat M_{t-1}\|^2=\|\mu\|^2+\eta\,\mathrm{tr}\,\Sigma$ and $E\sum\hat V=\|\mu\|^2+\mathrm{tr}\,\Sigma$ for Adam's
   bias-corrected second moment $\hat V$ (its total does not depend on the frame). With one gradient
   ($t=2$, $\eta=1$) or no momentum ($\beta_1=0$) mean and noise are not separable and $c_t=0$.

*Proof.* (1) Substitute $G_t-\mu_t$ for the zero-mean variable in the KRD density. (2) Expand
$Z=W+B_t$; $E[W\mid\mathcal F_{t-1}]=0$ and $B_t$ is $\mathcal F_{t-1}$-measurable, so the cross terms vanish,
and $E[W(W\odot A)^\top]_{ik}=\sum_jE[W_{ij}W_{kj}]A_{kj}=\delta_{ik}\sum_jD_{ij}A_{ij}$ is diagonal. (3) The
entries of $\Xi$ are weighted sums of independent zero-mean terms, so the same computation makes
$E[\Xi(\Xi\odot A)^\top]$ diagonal and kills the cross terms with $\Theta$; the risk identity is
Proposition 5.5's with $(S,\nu)=(\|\mu\|^2,\eta\,\mathrm{tr}\,\Sigma)$; $\eta$ is a geometric sum. ∎

Consequences. *Zero-mean gradients* ($\mu=0$): $c^\star=0$ and the statistic is the gradient, as
before C-013; the plug-in fluctuates near 0 (test `test_mean_shrinkage_separates_signal_from_noise`)
and, since $E[\Xi(\Xi\odot A)^\top]$ is diagonal, the true frame stays a fixed point of the expected flow
for any $c$ (Theorem 4.1 is unchanged). *A dominant mean* ($\|\mu\|^2\gg\eta\,\mathrm{tr}\,\Sigma$): $c\to1$, and
the frame is fitted to the noise, which on a quadratic with gradient noise proportional to the
curvature (the noisy quadratic model) shares the curvature's frame. *Invariances*: $c_t$ is a ratio of
quantities that are invariant under $(P,R)\in\mathcal G$ and scale as $\gamma^2$ under $G\mapsto\gamma G$, so
Theorems 6 and 8.2 hold. *Cost*: $Z_U(\hat M_t)=\beta_1Z_U(\hat M_{t-1})+(1-\beta_1)Z_U(G_t)$ in a fixed frame,
so rotating $\hat M_{t-1}$ instead of $\hat M_t$ gives both; the centering adds $O(mn)$ elementwise work and
two reductions, no matrix product. *Limits*: the plug-in assumes the mean is stationary over Adam's
windows; a mean that drifts within them (a decaying descent signal) leaves a lag bias in $B$, which
the factor treats as part of $\|\hat M\|^2$. *Sources*: centring a second moment by the running mean of
the gradient is centred RMSProp (Graves, 2013, arXiv:1308.0850) and AdaBelief (Zhuang et al.,
NeurIPS 2020, arXiv:2010.07468), both with $c=1$, the current gradient included, and applied to the
step's variances; a Stein rule toward the momentum also appears in SR-Adam (arXiv:2602.01777),
which shrinks the gradient itself. Here only the frame's statistic is centred, the plug-in is
predictable (so item 2's cross terms vanish), and $c$ is a James–Stein factor (James and Stein,
1961) that returns the uncentred statistic when the mean is not detectable.

**Proposition 4.5 (the flow descends $J$; Newton step at the optimum; L4, checked numerically in
E2.10 (e)).** Let $D=d_U$ (the profile variances at the current frame). Then
1. the expected score is the gradient of $J$ in left-trivialized skew coordinates: with
   $Q_L\mapsto Q_L(I+\Omega)$, $dJ=\sum_{i<k}\Omega_{ik}\,E[E^L]_{ik}$ (and the same on the right), so the
   noise-free flow $\Omega=-\alpha E[E]\oslash(F+\delta n)$ decreases $J$ monotonically;
2. at $U^\star$ the Hessian of $J$ in these coordinates is diagonal with entries $F^L_{ik}$, $F^R_{jl}$
   (left–right and pair–pair cross terms vanish), so near the optimum the natural-gradient step is
   a Newton step on $J$, which is why its contraction does not depend on the eigen-gap.

*Proof.* (1) $d\,(d_U)_{ij}=-2\sum_k\Omega_{ik}E[Z_{ij}Z_{kj}]$, so
$dJ=-\sum_{ik}\Omega_{ik}\sum_jE[Z_{ij}Z_{kj}]/(d_U)_{ij}=\sum_{i<k}\Omega_{ik}E[E^L]_{ik}$ by skew-symmetry.
(2) For a rotation by $\theta$ in one pair, $J(\theta)=\tfrac12\sum_j\log\big(1+\sin^2\theta\cos^2\theta\,g_j\big)$ with
$g_j=(D_{ij}-D_{kj})^2/(D_{ij}D_{kj})$ (Lean: `pair_rotation_product`), whose second derivative at 0 is
$\sum_jg_j=F_{ik}$; cross terms vanish because the Fisher information is diagonal (Theorem 2). ∎

**Conjecture 4.1 (L2).** For $S$ in the KRD family with all profiles pairwise distinct, the minimizers of
$J$ on $\mathcal G$ are $U^\star$ up to signed permutations, every other critical point is a strict
saddle, and the stochastic flow with steps $\alpha_t$, $\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$
converges almost surely to a minimizer. (ODE-method route; not proved here.) Numerical evidence
(L1, E2.11): from Haar-random frames the noise-free flow reached the global minimum in every run on
separable, non-separable, tied and near-degenerate arrays up to 24×32, and the critical points
obtained by 45° rotations of one or two pairs are strict saddles that the flow escapes.

**Theorem 5 (retractions; L5).** Let $\Omega$ be real skew-symmetric.
1. $I-\Omega$ is invertible and the Cayley transform $(I-\Omega)^{-1}(I+\Omega)$ is orthogonal.
2. $X=I+\Omega+\tfrac12\Omega^2$ satisfies $X^\top X=I+\tfrac14\Omega^4$: the second-order exponential
   retraction loses orthogonality only at fourth order.
3. For any $X$ with $X^\top X=I+E$, one Newton–Schulz step $Y=\tfrac12X(3I-X^\top X)$ gives
   $Y^\top Y-I=-\tfrac34E^2+\tfrac14E^3$ (quadratic convergence).
(Lean: `cayley_orthogonal`, `expm2_defect`, `ns_error`.) In the implementation each step's
generator is additionally limited to $\|\Omega\|_2\le1$ (spectral trust region): then the retraction's
singular values lie in $[1,\sqrt{5}/2]$, inside the convergence region of the Newton–Schulz polish,
which is repeated until the defect is below working precision.

**Theorem 6 (equivariance; L5 for the one-step algebra).** For $(P,R)\in\mathcal G$, run Gimbal on
gradients $G_t$ from frame $(Q_L,Q_R)$ and on $PG_tR^\top$ from $(PQ_L,RQ_R)$ with $M\mapsto PMR^\top$,
same $V$. Then all rotated quantities ($Z$, $V$, $S$, $F$, $\Omega$) coincide, the frames stay related
by $(P,R)$, and the parameter updates satisfy $\Delta W'=P\,\Delta W\,R^\top$. AdamW is not equivariant
in this sense; SOAP and Shampoo are. (Lean: `rotated_coords_invariant`, `update_covariant`. The eigh
initialization is equivariant up to the sign of each eigenvector, to which the update is
invariant; with repeated eigenvalues the initial frame is a gauge choice and equivariance holds
from any pair of initial frames related by $(P,R)$.)

**Theorem 7 (second-moment transport; L5).** If the frame moves by an orthogonal $P$ (new frame
$QP$), the diagonal of a diagonal covariance $\mathrm{diag}(d)$ seen in the new frame is $(P^{\circ2})^\top d$;
$P^{\circ2}$ is doubly stochastic, so total variance is preserved and the transported variances are
majorized by the old ones. For Kronecker frames $(P_L\otimes P_R)^{\circ2}=P_L^{\circ2}\otimes P_R^{\circ2}$,
so the transport of the matrix $V$ is $V\mapsto P_L^{\circ2\top}VP_R^{\circ2}$ (two matmuls). SOAP's
re-ordering of $V$ is the special case where $P$ is a signed permutation. (Lean: `hadamard_sq_kronecker`,
`rowsum_hadamard_sq`, `colsum_hadamard_sq`, `transport_preserves_total`.)

**Theorem 8 (descent and scale invariance; L5 for the algebra).**
1. With $\beta_1=0$, the update $U=Q_L(Z\oslash(\sqrt{\hat V}+\epsilon))Q_R^\top$ satisfies
   $\langle G,U\rangle_F=\sum_{ij}Z_{ij}^2/(\sqrt{\hat V_{ij}}+\epsilon)\ge0$, with equality iff $G=0$.
2. Replacing every gradient by $cG$ ($c\ne0$) leaves $\Omega_L,\Omega_R$ unchanged (with a relative floor):
   $S$ is homogeneous of degree $0$ in $(Z,D)$ jointly and so is $F$. The frame dynamics does not
   depend on gradient scale.
(Lean: `frobenius_adjoint`, `descent`, `descent_strict`, `generator_scale_invariant`,
`fisher_scale_invariant`.) The statement is about the flow from a given frame. The implementation
keeps it to rounding at every scale only if no absolute constant enters a scale-free formula (F-014)
and the initial frame is well defined: the eigenvectors of a rank-deficient single-gradient factor
(for example $G^\top G$ when $m<n$) are an arbitrary basis of its null space, so tiny perturbations
can select different, equivalent initial frames (F-015; E2.12 measures both cases).

**Proposition 9 (cost; L4).** Per step for an $m\times n$ layer, counting multiply–adds (generated
totals for the 125M model are in `experiments/phase2/results/e28_cost_model.json`; with fused QKV
and a 0.5M-token batch: SOAP 0.71% of forward+backward MACs plus QR, real-time SOAP 1.57% plus a QR
every step, KL-SOAP 1.94% plus QR, Gimbal $k=1$ 1.61%, $k=4$ 0.76%, $k=10$ 0.60%, all without
QR/eigh):

| | SOAP (f, QR) | SOAP real-time | KL-SOAP (F=1) | Gimbal |
|---|---|---|---|---|
| rotate G, M; rotate back | $3(m^2n+mn^2)$ | same | same | same |
| factor update | $m^2n+mn^2$ | same | $2(m^2n+mn^2)$ | — |
| frame update | $(m^3+n^3)$ matmul $+$ QR every $f$ | QR every step | QR every step | $S$: $m^2n+mn^2$ per step; every $k$ steps $F$: $m^2n+mn^2$, retraction + polish $4(m^3+n^3)$ |
| non-matmul linear algebra | QR every $f$ steps | QR every step | QR every step | **none** after the warm start (two eigendecompositions in total) |
| elementwise per step | $O(mn)$ | $O(mn)$ | $O(mn)$ | $O(mn)$ (variance averages, shrinkage: two logs, one exp) |
| optimizer state | $2m^2+2n^2+2mn$ | same | $2m^2+2n^2+2mn$ | $m^2+n^2+4mn$ ($k=1$; never more than SOAP since $2mn\le m^2+n^2$); $+m^2+n^2$ score accumulators for $k>1$, so the default $k=4$ exceeds SOAP by $2mn$ (F-023); $+m^2+n^2$ during the 50-step warm start |

The amortized flow divides the $m^3$ terms and the Fisher matrix by $k$.

**Proposition 10 (mechanism transfer; L4).** $E^L=Z\,g(Z)^\top-g(Z)Z^\top$ with $g(z)=z/D$ is the skew
(rotation) part of the EASI relative-gradient serial update of Cardoso & Laheld (1996) with the
heteroscedastic-Gaussian score as non-linearity; the Fisher normalization is Amari's natural gradient;
the identifiability condition of Theorem 2 is the Pham–Cardoso condition for separating nonstationary
sources, with "time blocks" replaced by the other tensor mode's index $j$. The transfer is exact at
the level of estimating equations; what does not transfer is the source-independence assumption
(gradients are not independent sources), which is why the result is stated for the working
likelihood only.

## 6. The peers in the same framework

Every method below is "a frame plus an eigenvalue rule". Using Proposition 1, the
KL cost of a method's preconditioner decomposes into a **frame cost** $J(U)$ and an **eigenvalue
cost** $\sum_{ij}[\log(\hat D_{ij}/(d_U)_{ij}) + (d_U)_{ij}/\hat D_{ij}-1]/2$.

| Method | Frame | Eigenvalue rule | Frame cost (stationary) | Eigenvalue cost |
|---|---|---|---|---|
| AdamW | identity | Adam (free) | $J(I)$, large when the gradient is rotated | sampling noise only |
| Shampoo / KL-Shampoo | pooled / KL factors | separable $\lambda\otimes\mu$ | inefficient / efficient only under separability (Thm 3) | KL of the separable fit of $D$ (zero iff separable) |
| SOAP | pooled factors, stale by $\le f$ steps | Adam (free) | inefficient by $n/n_{\mathrm{eff}}$, ties unidentified (Thm 2–3) | sampling noise + basis-change mismatch (Thm 7) |
| SOAP real-time | pooled, fresh | Adam | same inefficiency, no staleness | sampling noise |
| KL-SOAP | KL factors | Adam | efficient only if separable | sampling noise |
| SPlus | pooled, very stale | sign (instantaneous) | stale | no variance adaptation |
| Muon | singular frame of the momentum, instantaneous | all singular values $\to1$ | frame from one (momentum-averaged) sample; no averaging | ignores noise level per direction |
| ARO | Procrustes of $M f(R^\top M)^\top$, one-sided | base optimizer | criterion is loss-decrease, not a covariance fit | base optimizer |
| **Gimbal** | likelihood flow, fresh; flow variances shrunk toward the separable fit by the measured non-separability (Prop. 5.5) | Adam (free) | efficient (Thm 3), identifiable (Thm 2); plug-in noise reduced (Lemma 5.4, Prop. 5.5) | sampling noise; transport optional |

Phase 2 measures both costs by Monte Carlo for every method on the same gradient streams.

## 7. Open items

* Global convergence (Conjecture 4.1).
* Heavy-tailed gradients: the Gaussian score can be replaced by a Student-$t$ score
  $g(z)=z/(D+z^2/\nu)$ (hypothesis H6); its Fisher weights change accordingly. The split-sample
  noise estimate of Proposition 5.5 already adapts the variance estimate to heavy tails.
* Richer shrinkage targets for $\log D$ (low rank instead of additive; per-row factors $c_i$).
* Flat separable spectra: there the KL-factor weights are themselves efficient (Theorem 3.3), and
  KL-SOAP's frame KL is slightly below the default's in the two flattest separable E2.1 cells (the
  only cells where a practical peer is not significantly worse; `experiments/phase2/results/
  e21_report.md`, within G2.1's non-inferiority margin). The E2.9 ablations point to the damping
  $\delta$, which slows nearly degenerate pairs; a damping that decays as the variance estimates
  sharpen is a candidate repair, to be derived before it is tested.
* Interaction with momentum: the frame flow uses the innovation with respect to the momentum
  (Proposition 5.7) while the step uses the momentum itself; Theorem 8.1 covers $\beta_1=0$ only. The
  mean plug-in of Proposition 5.7 assumes a mean that is stationary over Adam's windows.

## 8. Proof audit (Phase 01, `theory-research` checklist)

| Item | Quantifiers and scope | Divisions / limits | Degenerate cases tested | Status |
|---|---|---|---|---|
| Prop. 1 | for every frame $U$, every $S\succ0$ | $D_{ij}>0$ required; $\log$ of positive reals | $S$ diagonal in $U$ (equality) | L4 (+L5 per coordinate) |
| Prop. 2 | $S\in$ KRD | none | tied row sums (Example 1) | L4 |
| Thm 2 | per pair, at the true frame | $D>0$ | equal profiles ($F=0$), tied sums | L4/L5 |
| Thm 3 | asymptotic in $N$ (item 1); exact inequalities (2–4) | gap $\ne0$, $F>0$ assumed explicitly | near-degenerate gaps, ratios $10^{\pm8}$, $n=1$ | L3 (1), L5 (2–4) |
| Thm 4 | local, $D$ known (item 2) | $\alpha\in(0,2)$ | $\alpha\to0$, $\alpha\to2$ | L3/L5 |
| Prop. 4.5 | any frame (1); at $U^\star$ (2) | $d_U>0$ | pairs with $F=0$ (zero curvature) | L4 (+ L1: E2.10 (e)) |
| Thm 5 | all real skew $\Omega$ | $I\pm\Omega$ invertible (proved) | large $\|\Omega\|$ (relative errors) | L5 |
| Thm 6 | all orthogonal $(P,R)$, same hyper-parameters | none | repeated eigenvalues at init (gauge) | L5 (one step) + test |
| Thm 7 | orthogonal $P$ | none | signed permutations | L5 |
| Thm 8 | $c\ne0$, relative floor | $D\ne0$ | $\beta_1=0$ only for item 1 | L5 |
| Lemma 5.4 | fixed frame (1); one pair, independent relative errors (2); Gaussian KRD (3) | $W>0$, weighted mean $>0$; $\langle w,w^\star\rangle\ne0$ | $F_{ik}\to0$ (expansion invalid: angle bounded), $\beta\to1$ | L5 (1, 2 identity), L3 (2 expansion, 3) |
| Prop. 5.5 | residual = signal + independent noise | $S+\nu>0$; $\|\hat R\|^2>0$ (else $c=0$) | separable ($S=0$), pure noise, $m=1$ or $n=1$ (no shrinkage possible) | L5 (risk), L3 (noise estimate) |
| Remark 5.6 | linearized dynamics, $\beta\in[0,1)$ | $1-\beta^{T+k}>0$ | $T_w=1$, ties | L5 (weighting), L3 |
| Prop. 5.7 | predictable plug-in (2); stationary mean, fixed $c$ (3) | $1-\eta>0$ (else $c=0$); $\|\hat M\|^2>0$ | $\mu=0$, $t=2$, $\beta_1=0$, frameless layers | L4 (1, 2; 3 for fixed $c$), L5 ($c^\star$, $\eta$), L3 (plug-in), L1 (E2.10 (f)) |

Failures found while auditing (all in the checking code, none in a theorem) are recorded in
`research/ledger/failures.md` (F-001 to F-004).

## Changelog

* 2026-10-06 — v0.1: initial statement of Propositions 1–2, 9–10, Theorems 2–8, Conjecture 4.1.
* 2026-10-06 — v0.2 (Phase 01): 37 statements machine-checked in Lean (`formal/README.md`); Theorem 3
  item 4 strengthened with a strict, machine-checked separation from every single-factor estimator;
  algorithm §5 gains the spectral trust region ($\|\Omega\|_2\le1$) and the adaptive Newton–Schulz
  polish after a unit test exposed a 1% orthogonality defect at large rotation rates (change-id
  C-001, `research/ledger/decisions.md`).
* 2026-10-06 — v0.3 (Phase 02, change C-002): bias-corrected rotation-rate schedule with
  Theorem 4.4 (machine-checked), defaults $\alpha=0.02$, $\delta=0.003$, amortized flow
  (`frame_every`), cost table updated from the generated cost model. Reason: the constant-rate flow
  with damping 0.1 escaped poor initial frames slowly (E2.1 pilot, F-009).
* 2026-10-06 — v0.4 (Phase 02, changes C-003 and C-004): the flow's variances are a separate average
  with the frame's memory (Lemma 5.4: profile likelihood with one forgetting factor; exact cost of
  noisy plug-in weights; orthogonality makes it a second-order effect), shrunk toward the separable
  fit by the empirical-Bayes factor of Proposition 5.5 with a split-sample noise estimate; pooled
  warm start as a one-step estimator (Remark 5.6), skipping the flow move at the restart step
  (F-012). Seven new Lean theorems (`formal/Formal/Variance.lean`, 45 in total). Reasons: the
  separable-cell failure of the exploratory E2.1 run (F-010) and the pilot that followed C-003
  (flat separable spectra still behind SOAP, diagnosed as plug-in noise for pairs with small
  Fisher information).
* 2026-10-06 — v0.5 (Phase 02, F-013): Lemma 5.4.2 gains its next-order coefficient $C_{ik}$. The
  pre-registered check of the leading-order formula (E2.10 (b), gate G2.5 (ii)) failed near the
  edge of the gated range; with the second-order term the error fell well inside the tolerance on
  the same pairs, and the gate is re-run on fresh pairs (`experiments/phase2/results/e210_report.md`,
  `research/ledger/failures.md`). Remark 5.6 gains the warm-start bias consequence (F-016, E2.2b).
* 2026-10-06 — v0.6 (Phase 02, C-012): default `frame_every` $=4$. With $k=1$ the cost model puts the
  optimizer above SOAP even with QR at 10× matmul cost (F-019); $k=4$ was pre-registered in the
  confirmatory runs and keeps the frame quality within a few percent. Proposition 4.5 (the flow
  descends $J$; Hessian of $J$ at $U^\star$ = Fisher) and numerical evidence for Conjecture 4.1 added.
* 2026-10-06 — v0.7 (Phase 02, C-013): Proposition 5.7. The frame is fitted to the innovation
  $G_t-c_t\hat M_{t-1}$ with an empirical-Bayes factor $c_t$; item 2 shows that the uncentred statistic
  has a non-zero expected score at the true frame when the gradient has a mean, which is how the
  noisy quadratic's deterministic signal tilted the frame and G2.4 failed in its separable low-noise
  configuration (F-020). One new Lean theorem (`ema_weight_sq_sum`, the momentum's noise factor
  $\eta$); the optimal factor reuses `shrinkage_risk_eq_iff`. Notation §1, §5 and the open items
  updated; earlier statements are unchanged (they concern the zero-mean model, where $c^\star=0$).

