"""Phase 1 counterexample battery (phases/01_formal_theory_and_lean.md, task 4).

Tries to break every identity and inequality of theory/gimbal_theory.md numerically, including
the asymptotic statements that Lean does not cover (delta-method variances, Fisher information,
stationary variance of the flow). Writes results/check_identities.json and exits non-zero if any
check fails.

Usage: python experiments/phase1/check_identities.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

RESULTS = pathlib.Path(__file__).parent / "results"


def fisher(a, b):
    return np.sum((a - b) ** 2 / (a * b))


def v_weighted(w, a, b):
    return np.sum(w**2 * a * b) / np.sum(w * (a - b)) ** 2


def rand_orth(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def check_efficiency_inequality(rng, trials):
    """Theorem 3.2 on random and adversarial profiles; equality for MLE weights."""
    worst = np.inf
    eq_err = 0.0
    for t in range(trials):
        n = rng.integers(1, 40)
        scale = 10.0 ** rng.uniform(-8, 8, size=2)
        a = scale[0] * np.exp(rng.normal(0, rng.uniform(0.01, 4), n))
        b = scale[1] * np.exp(rng.normal(0, rng.uniform(0.01, 4), n))
        if t % 7 == 0:  # near-degenerate pair
            b = a * (1 + 1e-6 * rng.standard_normal(n))
        w = np.exp(rng.normal(0, 2, n))
        f = fisher(a, b)
        gap = np.sum(w * (a - b))
        if f <= 0 or gap == 0:
            continue
        ratio = v_weighted(w, a, b) * f  # must be >= 1
        worst = min(worst, ratio)
        w_mle = (a - b) / (a * b)
        eq_err = max(eq_err, abs(v_weighted(w_mle, a, b) * f - 1))
    return {
        "min_ratio_V_times_F": worst,
        "mle_equality_max_err": eq_err,
        "pass": bool(worst >= 1 - 1e-9 and eq_err < 1e-6),
    }


def check_separable_formulas(rng, trials):
    """KL weights efficient under separability; SOAP loss = n sum mu^2 / (sum mu)^2."""
    err_kl, err_soap = 0.0, 0.0
    for _ in range(trials):
        n = rng.integers(1, 50)
        mu = np.exp(rng.normal(0, 2, n))
        li, lk = np.exp(rng.normal(0, 2, 2))
        a, b = li * mu, lk * mu
        f = fisher(a, b)
        err_kl = max(err_kl, abs(v_weighted(1 / mu, a, b) * f - 1))
        pred = n * np.sum(mu**2) / np.sum(mu) ** 2
        err_soap = max(err_soap, abs(v_weighted(np.ones(n), a, b) * f / pred - 1))
    return {
        "kl_efficiency_max_err": err_kl,
        "soap_loss_formula_max_err": err_soap,
        "pass": bool(err_kl < 1e-8 and err_soap < 1e-8),
    }


def angle_between(q_est, q_true, i, k):
    """Rotation angle of the estimated pair (i,k) relative to the true frame (small-angle)."""
    p = q_true.T @ q_est
    return 0.5 * (p[k, i] - p[i, k])


def check_delta_method(rng, quick):
    """Theorem 3.1: N * Var(theta_hat) -> V_w for pooled factors (w = 1) and KL (w = 1/mu)."""
    m, n = 2, 6
    d = np.array([[4.0, 1.0, 2.0, 0.5, 3.0, 1.0], [1.0, 2.0, 1.0, 1.0, 0.8, 2.5]])
    q_true = rand_orth(m, rng)
    # The statement is asymptotic in N; use N large enough for first-order perturbation theory
    # to hold for the smaller (weighted) eigen-gap.
    n_samples, reps = (3000, 600) if quick else (8000, 2000)
    th_pool, th_kl = [], []
    w_kl = 1.0 / d.mean(axis=0)  # a fixed positive weighting, as KL-Shampoo's R^{-1} would give
    for _ in range(reps):
        z = np.sqrt(d)[None] * rng.standard_normal((n_samples, m, n))
        g = np.einsum("ab,tbj->taj", q_true, z)  # right frame = identity
        lp = np.einsum("taj,tbj->ab", g, g) / n_samples
        lw = np.einsum("taj,j,tbj->ab", g, w_kl, g) / n_samples
        for target, mat in ((th_pool, lp), (th_kl, lw)):
            evals, evecs = np.linalg.eigh(mat)
            evecs = evecs[:, ::-1]
            # align signs/order with the true frame
            order = np.argmax(np.abs(q_true.T @ evecs), axis=0)
            q_est = evecs[:, np.argsort(order)]
            q_est = q_est * np.sign(np.diag(q_true.T @ q_est))
            target.append(angle_between(q_est, q_true, 0, 1))
    a, b = d[0], d[1]
    pred_pool = v_weighted(np.ones(n), a, b) / n_samples
    pred_kl = v_weighted(w_kl, a, b) / n_samples
    emp_pool, emp_kl = np.var(th_pool), np.var(th_kl)
    cr = 1 / (fisher(a, b) * n_samples)
    rel = lambda e, p: abs(e / p - 1)  # noqa: E731
    tol = 0.15 if quick else 0.08
    return {
        "pooled_pred": pred_pool,
        "pooled_emp": emp_pool,
        "kl_pred": pred_kl,
        "kl_emp": emp_kl,
        "cramer_rao": cr,
        "pass": bool(
            rel(emp_pool, pred_pool) < tol
            and rel(emp_kl, pred_kl) < tol
            and emp_pool > cr
            and emp_kl > cr
        ),
    }


def check_fisher_mc(rng, quick):
    """Theorem 2: E[score^2] = F at the true frame."""
    n = 5
    a = np.exp(rng.normal(0, 1, n))
    b = np.exp(rng.normal(0, 1, n))
    reps = 200_000 if quick else 2_000_000
    zi = np.sqrt(a) * rng.standard_normal((reps, n))
    zk = np.sqrt(b) * rng.standard_normal((reps, n))
    score = np.sum(zi * zk * (1 / b - 1 / a), axis=1)
    emp = np.mean(score**2)
    f = fisher(a, b)
    return {
        "fisher": f,
        "mc_score_second_moment": emp,
        "mean_score": float(np.mean(score)),
        "pass": bool(abs(emp / f - 1) < 0.02 and abs(np.mean(score)) < 0.02 * np.sqrt(f)),
    }


def check_flow_stationary_variance(rng, quick):
    """Theorem 4.3: stationary angle variance of the online flow is alpha/(2-alpha)/F."""
    n = 8
    a = np.exp(rng.normal(0, 0.8, n)) * 2
    b = np.exp(rng.normal(0, 0.8, n))
    f = fisher(a, b)
    alpha = 0.05
    steps = 60_000 if quick else 400_000
    theta = 0.0
    trace = []
    for t in range(steps):
        c, s = np.cos(theta), np.sin(theta)
        # true coordinates -> observed in a frame rotated by theta
        zi_t = np.sqrt(a) * rng.standard_normal(n)
        zk_t = np.sqrt(b) * rng.standard_normal(n)
        zi = c * zi_t + s * zk_t
        zk = -s * zi_t + c * zk_t
        score = np.sum(zi * zk * (1 / b - 1 / a))
        # theta parametrizes the frame error with the opposite sign of the generator coordinate
        # (z_i -> z_i + dtheta z_k, whereas Q(I+Omega) gives z_i -> z_i - phi z_k), so the
        # natural-gradient descent step on phi is an ascent-looking step on theta.
        theta = theta + alpha * score / f
        if t > 5000:
            trace.append(theta)
    trace = np.asarray(trace)
    emp = np.var(trace)
    pred = alpha / (2 - alpha) / f
    # AR(1) autocorrelation makes the variance estimate noisy: tolerance 15%.
    return {
        "pred": pred,
        "emp": float(emp),
        "mean": float(trace.mean()),
        "pass": bool(abs(emp / pred - 1) < 0.15 and abs(trace.mean()) < 3 * np.sqrt(pred)),
    }


def check_retractions(rng, trials):
    err_exp, err_ns, err_cay = 0.0, 0.0, 0.0
    for _ in range(trials):
        n = rng.integers(2, 30)
        x = rng.standard_normal((n, n)) * rng.uniform(1e-3, 0.5)
        om = x - x.T
        eye = np.eye(n)
        r = eye + om + 0.5 * om @ om
        om4 = np.linalg.matrix_power(om, 4)
        err_exp = max(err_exp, np.abs(r.T @ r - eye - 0.25 * om4).max() / max(1, np.abs(om4).max()))
        e = r.T @ r - eye
        y = r @ (1.5 * eye - 0.5 * r.T @ r)
        rhs = -0.75 * e @ e + 0.25 * e @ e @ e
        scale = max(1.0, np.abs(e @ e @ e).max(), np.abs(y.T @ y).max())
        err_ns = max(err_ns, np.abs(y.T @ y - eye - rhs).max() / scale)
        cay = np.linalg.solve(eye - om, eye + om)
        err_cay = max(err_cay, np.abs(cay.T @ cay - eye).max())
    return {
        "expm2_identity_err": err_exp,
        "ns_identity_err": err_ns,
        "cayley_orth_err": err_cay,
        "pass": bool(err_exp < 1e-10 and err_ns < 1e-10 and err_cay < 1e-10),
    }


def check_transport(rng, trials):
    err_ds, err_kron = 0.0, 0.0
    for _ in range(trials):
        m, n = rng.integers(2, 12, size=2)
        pl, pr = rand_orth(m, rng), rand_orth(n, rng)
        t = pl * pl
        err_ds = max(err_ds, np.abs(t.sum(0) - 1).max(), np.abs(t.sum(1) - 1).max())
        k = np.kron(pl, pr)
        err_kron = max(err_kron, np.abs(k * k - np.kron(pl * pl, pr * pr)).max())
    return {
        "double_stochastic_err": err_ds,
        "kron_hadamard_err": err_kron,
        "pass": bool(err_ds < 1e-12 and err_kron < 1e-12),
    }


def check_tie_identifiability(rng):
    """Example 1 / Theorem 2: pooled factors tie, the likelihood flow still finds the frame."""
    d = np.array([[4.0, 1.0], [1.0, 4.0]])  # rows have equal sums
    q_true = rand_orth(2, rng)
    pooled_spread = []
    for _ in range(50):  # pooled eigenvectors are arbitrary in the tied plane
        z = np.sqrt(d)[None] * rng.standard_normal((5000, 2, 2))
        g = np.einsum("ab,tbj->taj", q_true, z)
        lp = np.einsum("taj,tbj->ab", g, g)
        _, ev = np.linalg.eigh(lp)
        p = q_true.T @ ev
        pooled_spread.append(abs(np.arctan2(p[1, 0], p[0, 0])) % (np.pi / 2))
    # online likelihood flow from a 40-degree error
    theta_err = np.deg2rad(40)
    alpha = 0.02
    f = fisher(d[0], d[1])
    for _ in range(4000):
        c, s = np.cos(theta_err), np.sin(theta_err)
        zt = np.sqrt(d) * rng.standard_normal((2, 2))
        zi = c * zt[0] + s * zt[1]
        zk = -s * zt[0] + c * zt[1]
        score = np.sum(zi * zk * (1 / d[1] - 1 / d[0]))
        theta_err += alpha * score / f  # same sign convention as check_flow_stationary_variance
    spread = float(np.std(pooled_spread))
    return {
        "pooled_angle_spread_rad": spread,
        "flow_final_error_deg": float(np.rad2deg(theta_err)),
        "pass": bool(spread > 0.2 and abs(np.rad2deg(theta_err)) < 5),
    }


def check_gimbal_step_invariances():
    """Theorem 6 and 8 on the reference implementation (also covered by tests/)."""
    import torch

    from gimbal.torch import Gimbal

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    grads = [
        torch.randn(6, 6, generator=gen, dtype=torch.float64)
        * torch.linspace(0.2, 2, 6, dtype=torch.float64)
        for _ in range(30)
    ]

    def run(gs, w0):
        w = torch.nn.Parameter(w0.clone())
        opt = Gimbal([w], lr=1e-2, rot_rate=0.2, weight_decay=0.05)
        for g in gs:
            w.grad = g.clone()
            opt.step()
        return w.detach()

    def orth(n):
        q, r = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
        return q * torch.sign(torch.diag(r))

    p, r = orth(6), orth(6)
    w0 = torch.randn(6, 6, generator=gen, dtype=torch.float64)
    wa = run(grads, w0)
    wb = run([p @ g @ r.T for g in grads], p @ w0 @ r.T)
    err = float((wb - p @ wa @ r.T).abs().max())
    return {"equivariance_err": err, "pass": bool(err < 1e-8)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(20261006)
    trials = 2000 if args.quick else 20000
    results = {
        "efficiency_inequality": check_efficiency_inequality(rng, trials),
        "separable_formulas": check_separable_formulas(rng, trials),
        "delta_method_variances": check_delta_method(rng, args.quick),
        "fisher_monte_carlo": check_fisher_mc(rng, args.quick),
        "flow_stationary_variance": check_flow_stationary_variance(rng, args.quick),
        "retractions": check_retractions(rng, trials // 10),
        "transport": check_transport(rng, trials // 10),
        "tie_identifiability": check_tie_identifiability(rng),
        "gimbal_step_equivariance": check_gimbal_step_invariances(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("check_identities_quick.json" if args.quick else "check_identities.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    ok = all(r["pass"] for r in results.values())
    for name, r in results.items():
        print(
            f"{'PASS' if r['pass'] else 'FAIL'}  {name}: "
            + ", ".join(
                f"{k}={v:.4g}" for k, v in r.items() if k != "pass" and isinstance(v, (int, float))
            )
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
