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
multiplications, needs no Kronecker factor buffers and no QR or eigendecomposition after the first
step. The theory below shows the estimator is identifiable in strictly more situations than pooled
factors (Theorem 2), statistically efficient where every single-factor estimator is not (Theorem 3),
locally contracting at a gap-independent rate (Theorem 4), orthogonality-preserving (Theorem 5),
equivariant (Theorem 6), and scale-invariant (Theorem 8).

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
* We model the *uncentred* second moment, as Shampoo, SOAP and Adam do. The Gaussian is a working
  likelihood (a surrogate fixing the estimating equation), not a claim that gradients are Gaussian.

## 2. The objective behind SOAP's own preconditioner

**Proposition 1 (profile KL; L4).** Let $S\succ0$ be the second-moment operator of $\mathrm{vec}\,G$.
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
$J(U^\star)=0$. (Lean: `Formal/Identifiability.lean`, the tie witness.)

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

**Theorem 2 (Fisher information and identifiability; L4, finite-sum parts L5).**
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
(Lean: `Formal/Identifiability.lean`: `fisher_pos_iff`, `pooled_implies_profile`,
`tie_but_identifiable`.)

## 4. Efficiency: why every single-factor estimator loses information

Consider $N$ i.i.d. observations from $\mathrm{KRD}(U^\star,D)$ and the **weighted-factor estimator**
with column weights $w\in\mathbb R^n_{>0}$ (expressed in the true right frame):
$\hat L_w = \frac1N\sum_t G_tQ_R^\star\mathrm{diag}(w)Q_R^{\star\top}G_t^\top$, $\hat Q_L=$ eigenvectors of
$\hat L_w$. Shampoo/SOAP use $w\equiv1$; KL-Shampoo uses $w=1/\hat\mu$ (its $R^{-1}$ weighting).

**Theorem 3 (efficiency; L3: asymptotics by the delta method under distinct population eigenvalues
$\ell_i(w)=\sum_jw_jD_{ij}$; the inequality itself L5).**

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
4. The MLE needs **pair-dependent** weights $w^{(ik)}_j\propto D_{kj}^{-1}-D_{ij}^{-1}$. No single
   factor matrix can supply them unless item 3 holds. Gimbal's generator applies exactly these weights
   (Lemma A), so its estimating equation is the likelihood equation.

*Proof.* (1) In the true left frame the off-diagonal entry $(i,k)$ of $Q_L^{\star\top}\hat L_wQ_L^\star$ is
$\frac1N\sum_t\sum_jw_jZ^{(t)}_{ij}Z^{(t)}_{kj}$ with variance $\frac1N\sum_jw_j^2D_{ij}D_{kj}$; first-order
eigenvector perturbation divides by the eigen-gap $\ell_i(w)-\ell_k(w)=\sum_jw_j(D_{ij}-D_{kj})$.
(2) Write $r_j=w_j(D_{ij}-D_{kj})$, $f_j=w_j^2D_{ij}D_{kj}$, $g_j=(D_{ij}-D_{kj})^2/(D_{ij}D_{kj})$. Then
$r_j^2=f_jg_j$ and Cauchy–Schwarz gives $(\sum r_j)^2\le(\sum f_j)(\sum g_j)$, i.e. $V_w\ge1/F$, with
equality iff $w_j\sqrt{D_{ij}D_{kj}}\propto(D_{ij}-D_{kj})/\sqrt{D_{ij}D_{kj}}$.
(3) Equality for all pairs means $D_{kj}^{-1}-D_{ij}^{-1}=c_{ik}w_j$; fixing $k=k_0$ gives
$D_{ij}^{-1}=D_{k_0j}^{-1}-c_{ik_0}w_j$, the stated form; the converse is direct. The separable
efficiency ratio follows by substitution. ∎
(Lean: `Formal/Efficiency.lean`: `weighted_cs`, `pooled_ge_mle`, `kl_weights_efficient_separable`.)

**Corollary 3.1 (L3).** Under a basis that drifts at angular velocity $\omega$ per step in pair
$(i,k)$, an EMA estimator with memory $\tau$ has asymptotic tracking MSE
$\approx\omega^2\tau^2+V/\big((2\tau)\big)$ up to constants, with $V=V_w$ for factor methods and $V=1/F$
for the likelihood flow (Theorem 4). Since $V_w\ge1/F$, the best achievable tracking error of the
likelihood flow is no larger than that of any single-factor method at its best memory.

## 5. The Gimbal flow

Let $V$ be Adam's second-moment EMA of $Z_U(G)^{\circ2}$ (bias corrected) and
$D=V+\rho\,\overline V$ (relative floor $\rho$, $\overline V$ the mean). One step of the left flow:
$$\Omega_L=-\alpha\,\frac{S_L-S_L^\top}{F_L+\delta n},\qquad
F_L=DA^\top+AD^\top-2n\,\mathbf 1\mathbf 1^\top,\quad A=D^{\circ-1},$$
(entrywise division; diagonal set to zero), and $Q_L\leftarrow Q_L\,(I+\Omega_L+\tfrac12\Omega_L^2)$. The
right flow is symmetric with $S_R=Z^\top(Z\odot A)$, $F_R=D^\top A+A^\top D-2m$, damping $\delta m$. Note
$(DA^\top+AD^\top)_{ik}-2n=\sum_j(D_{ij}/D_{kj}+D_{kj}/D_{ij}-2)=F^L_{ik}$, so $F_L$ is the Fisher
information of Theorem 2 evaluated at the current estimates; dividing by it is Amari's natural
gradient, and $\delta$ is a Levenberg–Marquardt damping that bounds the step for near-degenerate
pairs.

**Theorem 4 (fixed points, local rate, noise; L3: linearization around $U^\star$ with $D$ known).**

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
divides by $F$. (3) Standard AR(1) algebra. ∎ (Lean: `Formal/Recursion.lean`, fixed point and
geometric convergence of the variance recursion.)

**Conjecture 4.1 (L2).** For $S$ in the KRD family with all profiles pairwise distinct, the minimizers of
$J$ on $\mathcal G$ are $U^\star$ up to signed permutations, every other critical point is a strict
saddle, and the stochastic flow with steps $\alpha_t$, $\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$
converges almost surely to a minimizer. (ODE-method route; not proved here.)

**Theorem 5 (retractions; L5).** Let $\Omega$ be real skew-symmetric.
1. $I-\Omega$ is invertible and the Cayley transform $(I-\Omega)^{-1}(I+\Omega)$ is orthogonal.
2. $X=I+\Omega+\tfrac12\Omega^2$ satisfies $X^\top X=I+\tfrac14\Omega^4$: the second-order exponential
   retraction loses orthogonality only at fourth order.
3. For any $X$ with $X^\top X=I+E$, one Newton–Schulz step $Y=\tfrac12X(3I-X^\top X)$ gives
   $Y^\top Y-I=-\tfrac34E^2+\tfrac14E^3$ (quadratic convergence).
(Lean: `Formal/Retraction.lean`.)

**Theorem 6 (equivariance; L5 for the one-step algebra).** For $(P,R)\in\mathcal G$, run Gimbal on
gradients $G_t$ from frame $(Q_L,Q_R)$ and on $PG_tR^\top$ from $(PQ_L,RQ_R)$ with $M\mapsto PMR^\top$,
same $V$. Then all rotated quantities ($Z$, $V$, $S$, $F$, $\Omega$) coincide, the frames stay related
by $(P,R)$, and the parameter updates satisfy $\Delta W'=P\,\Delta W\,R^\top$. AdamW is not equivariant
in this sense; SOAP and Shampoo are. (Lean: `Formal/Equivariance.lean`.)

**Theorem 7 (second-moment transport; L5).** If the frame moves by an orthogonal $P$ (new frame
$QP$), the diagonal of a diagonal covariance $\mathrm{diag}(d)$ seen in the new frame is $(P^{\circ2})^\top d$;
$P^{\circ2}$ is doubly stochastic, so total variance is preserved and the transported variances are
majorized by the old ones. For Kronecker frames $(P_L\otimes P_R)^{\circ2}=P_L^{\circ2}\otimes P_R^{\circ2}$,
so the transport of the matrix $V$ is $V\mapsto P_L^{\circ2\top}VP_R^{\circ2}$ (two matmuls). SOAP's
re-ordering of $V$ is the special case where $P$ is a signed permutation. (Lean:
`Formal/Transport.lean`.)

**Theorem 8 (descent and scale invariance; L5 for the algebra).**
1. With $\beta_1=0$, the update $U=Q_L(Z\oslash(\sqrt{\hat V}+\epsilon))Q_R^\top$ satisfies
   $\langle G,U\rangle_F=\sum_{ij}Z_{ij}^2/(\sqrt{\hat V_{ij}}+\epsilon)\ge0$, with equality iff $G=0$.
2. Replacing every gradient by $cG$ ($c\ne0$) leaves $\Omega_L,\Omega_R$ unchanged (with a relative floor):
   $S$ is homogeneous of degree $0$ in $(Z,D)$ jointly and so is $F$. The frame dynamics does not
   depend on gradient scale.
(Lean: `Formal/Descent.lean`.)

**Proposition 9 (cost; L4).** Per step for an $m\times n$ layer, counting multiply–adds:

| | SOAP (f, QR) | SOAP real-time | KL-SOAP (F=1) | Gimbal |
|---|---|---|---|---|
| rotate G, M; rotate back | $3(m^2n+mn^2)$ | same | same | same |
| factor update | $m^2n+mn^2$ | same | $2(m^2n+mn^2)$ | — |
| frame update | $(m^3+n^3)$ matmul $+$ QR every $f$ | QR every step | QR every step | $S,F$: $2(m^2n+mn^2)$; retraction $2(m^3+n^3)$ |
| non-matmul linear algebra | QR every $f$ steps | QR every step | QR every step | **none** after step 1 |
| optimizer state | $2m^2+2n^2+2mn$ | same | $2m^2+2n^2+2mn$ | $m^2+n^2+2mn$ |

The $m^3$ retraction can be amortized by accumulating $\Omega$ for $k$ steps (Phase 4 option).

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
| **Gimbal** | likelihood flow, fresh | Adam (free) | efficient (Thm 3), identifiable (Thm 2) | sampling noise; transport optional |

Phase 2 measures both costs by Monte Carlo for every method on the same gradient streams.

## 7. Open items

* Global convergence (Conjecture 4.1).
* Heavy-tailed gradients: the Gaussian score can be replaced by a Student-$t$ score
  $g(z)=z/(D+z^2/\nu)$ (hypothesis H6); its Fisher weights change accordingly.
* Interaction with momentum: the frame flow uses the instantaneous gradient while the step uses the
  momentum; Theorem 8.1 covers $\beta_1=0$ only.

## Changelog

* 2026-10-06 — v0.1: initial statement of Propositions 1–2, 9–10, Theorems 2–8, Conjecture 4.1.
