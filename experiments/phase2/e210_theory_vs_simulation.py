"""E2.10: does the theory predict the simulations? (Phase 02, gate G2.5)

(a) Frame KL of the exact pooled-factor frames ("pooled_eigh", the asymptotically exact control)
    predicted from Theorem 3.1's per-pair variances V_1 and the expansion
    J ~ 1/2 sum_p F_p theta_p^2 (Lean: `pair_rotation_cost_le`), against the E2.1
    measurements on the same arrays (seeds 10-19).
    Also reported: SOAP and KL-SOAP against the same predictions, and Gimbal against the
    Cramer-Rao value 1/2 (number of pairs) sum w^2 at its rate.
(b) Lemma 5.4.2: the second-order plug-in inflation 1 + eps2 (1 + (2n - R)/F) against the exact
    E[V_w F] over random relative errors of the variances.
(c) Proposition 5.5: the split-sample noise estimate and the shrinkage factor, computed exactly as
    in `Gimbal._shrunk_variances` (checked), against the replication variance of log V and the
    oracle factor S / (S + nu).
(d) Theorem 4.2: one step of the noise-free flow from a small displacement in one pair; measured
    contraction against 1 - alpha F / (F + delta n), across pairs whose F spans >= 100x; SOAP's
    power-iteration contraction (eigenvalue ratio) for the same pairs, for contrast.
(e) The Hessian of J at the true frame against the Fisher information, and the gradient of J at a
    random frame against the expected score (finite differences).
(f) Proposition 5.7: the momentum's noise factor eta against the Monte Carlo variance of the
    bias-corrected momentum; the mean skew score at the true frame with a plug-in mean c M_{t-1}
    against (1 - c)^2 times the uncentred bias; the optimizer's plug-in factor (its own code path)
    against the oracle c* = |mu|^2 / (|mu|^2 + eta tr Sigma). Criteria fixed before the run (C-014).

Usage: python experiments/phase2/e210_theory_vs_simulation.py
"""

from __future__ import annotations

import gzip
import json
import pathlib
import zlib
from collections import defaultdict

import numpy as np
import torch
from analyze_e21 import load as load_e21
from common import make_variances, rand_orth
from population import (
    ema_sum_sq_weights,
    expected_scores,
    frame_kl,
    givens,
    kl_fixed_point,
    pair_quantities,
    population_step,
    weighted_variance,
)

from gimbal.torch import Gimbal

RESULTS = pathlib.Path(__file__).parent / "results"
STEPS = 800


def read_jsonl(name: str) -> list[dict]:
    rows = []
    for path in (RESULTS / name, RESULTS / (name + ".gz")):
        if path.exists():
            text = (
                gzip.decompress(path.read_bytes()).decode()
                if path.suffix == ".gz"
                else path.read_text()
            )
            rows += [json.loads(x) for x in text.splitlines() if x.strip()]
    return rows


def regenerate_variances(cfg: dict, seed: int) -> np.ndarray:
    """The variance array of an E2.1 cell, drawn exactly as in e21_frame_efficiency.run_cell."""
    key = {k: cfg[k] for k in ("gamma", "slope", "shape", "tie", "nu", "drift")}
    key["shape"] = tuple(key["shape"])
    cell_id = zlib.crc32(json.dumps(key, default=str, sort_keys=True).encode()) % 100_003
    rng = np.random.default_rng(1000 * seed + cell_id)
    m, n = key["shape"]
    return make_variances(m, n, key["gamma"], key["slope"], rng, tie=key["tie"])


# ---------------------------------------------------------------------------------------------
# (a) predicted vs measured frame KL
# ---------------------------------------------------------------------------------------------


def per_sample_costs(d: np.ndarray, rho_start: float) -> dict:
    """1/2 sum_p F_p V_p for pooled and KL weights, the number of pairs (Cramer-Rao), and the share
    of the pooled prediction carried by pairs whose predicted angle s.d. at the start of the window
    exceeds 0.15 rad (outside the linear regime of the expansion; C-009)."""
    out = {"pairs": 0, "pooled": 0.0, "kl": 0.0, "nonlinear": 0.0}
    ell, r = kl_fixed_point(d)
    for side, w in (("left", 1.0 / r), ("right", 1.0 / ell)):
        f, v1, _ = pair_quantities(d, side)
        vkl = weighted_variance(d, w, side)
        out["pairs"] += len(f)
        out["pooled"] += 0.5 * float(np.sum(f * v1))
        out["kl"] += 0.5 * float(np.sum(f * vkl))
        out["nonlinear"] += 0.5 * float(np.sum((f * v1)[np.sqrt(v1 * rho_start) > 0.15]))
    out["nonlinear_share"] = out["nonlinear"] / out["pooled"]
    return out


def part_a() -> dict:
    # the analysis's own loader: peer-fix (C-011) and C-013 re-runs replace earlier rows (F-022)
    rows = load_e21("main")
    measured = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    cfgs = {}
    for r in rows:
        key = (r["gamma"], r["slope"], tuple(r["shape"]))
        cfgs[key] = r
        measured[key][r["method"]][r["memory"]][r["seed"]] = r["kl_second_half"]
    t = np.arange(STEPS // 2 + 1, STEPS + 1)  # samples seen at the second-half steps
    cells = []
    for key in sorted(measured):
        seeds = sorted(measured[key]["pooled_eigh"][0.99])
        pe_best = min(
            measured[key]["pooled_eigh"],
            key=lambda mem: np.mean(list(measured[key]["pooled_eigh"][mem].values())),
        )
        rho_start = float(ema_sum_sq_weights(pe_best, t[:1])[0])
        costs = [per_sample_costs(regenerate_variances(cfgs[key], s), rho_start) for s in seeds]
        mean_cost = {k: float(np.mean([c[k] for c in costs])) for k in costs[0]}
        cell = {
            "gamma": key[0],
            "slope": key[1],
            "shape": list(key[2]),
            "methods": {},
            "nonlinear_share_pooled": mean_cost["nonlinear_share"],
        }
        for method, cost_key in (
            ("pooled_eigh", "pooled"),
            ("soap", "pooled"),
            ("soap_rt", "pooled"),
            ("kl_eigh", "kl"),
            ("klsoap", "kl"),
            ("gimbal", "pairs"),
            ("gimbal_k4", "pairs"),
        ):
            per_mem = {}
            for mem, by_seed in measured[key][method].items():
                beta = 1.0 - mem if method.startswith("gimbal") else mem
                rho = ema_sum_sq_weights(beta, t)
                scale = 0.5 if cost_key == "pairs" else 1.0
                pred = scale * mean_cost[cost_key] * float(rho.mean())
                meas = float(np.mean(list(by_seed.values())))
                per_mem[str(mem)] = {"measured": meas, "predicted": pred, "ratio": meas / pred}
            best = min(per_mem, key=lambda k: per_mem[k]["measured"])
            cell["methods"][method] = {"best_memory": best, "by_memory": per_mem}
        cells.append(cell)
    # gate (i): pooled_eigh at its best memory, in cells where pairs outside the linear regime
    # carry at most 15% of the predicted frame KL (C-009)
    checked = []
    for c in cells:
        pe = c["methods"]["pooled_eigh"]
        b = pe["by_memory"][pe["best_memory"]]
        if c["nonlinear_share_pooled"] <= 0.15:
            checked.append((c["gamma"], c["slope"], c["shape"], b["ratio"]))
    ok = bool(checked) and all(0.75 <= x[3] <= 1.33 for x in checked)
    return {"cells": cells, "gate_cells": checked, "pass": ok}


# ---------------------------------------------------------------------------------------------
# (b) Lemma 5.4.2: plug-in inflation
# ---------------------------------------------------------------------------------------------


def plugin_excess(a: np.ndarray, b: np.ndarray, s: float) -> tuple[float, float]:
    """Lemma 5.4.2 for Gaussian relative errors of variance s: (leading, through second order).

    Leading: s K / F with K = F + 2n - R. Second order (theory v0.5, F-013): adds s^2 C_2 with
    C_2 F = 6F + 12n + 2R + 8 + (Q - 18U + 6nR - 9R^2)/F + (18X - 6Q - 6W + 6 Z_1)/F^2, where
    g = c^2/m, c = a - b, m = a b, Q = sum g^2, U = sum m c (b^-3 - a^-3),
    X = sum c^3 (b^-3 - a^-3), W = sum g c^2 (a^-2 + b^-2), Z_1 = sum m c^2 (a^-2 + b^-2)^2.
    (Exactly 0 when n = 1, as the exact excess is.)
    """
    n = len(a)
    c, m = a - b, a * b
    g = c**2 / m
    f = g.sum()
    r = (c**2 * (a**-2 + b**-2)).sum() / f
    k = f + 2 * n - r
    q = (g**2).sum()
    u = (m * c * (b**-3 - a**-3)).sum()
    x = (c**3 * (b**-3 - a**-3)).sum()
    w = (g * c**2 * (a**-2 + b**-2)).sum()
    z1 = (m * c**2 * (a**-2 + b**-2) ** 2).sum()
    c2f = (
        6 * f
        + 12 * n
        + 2 * r
        + 8
        + (q - 18 * u + 6 * n * r - 9 * r**2) / f
        + (18 * x - 6 * q - 6 * w + 6 * z1) / f**2
    )
    lead = s * k / f
    return float(lead), float(lead + s**2 * c2f / f)


def part_b(seed: int, formula: str, draws: int = 4000) -> dict:
    """Exact E[V_w F] - 1 over random relative errors vs Lemma 5.4.2's prediction.

    The pre-registered run used the leading-order formula on seed 61 (failed, F-013); the re-test
    after the theory revision uses the second-order formula on fresh pairs (seed 62).
    """
    rng = np.random.default_rng(seed)
    out, ok = [], True
    for gamma in (0.0, 0.5, 1.0, 2.0):
        for slope in (0.5, 1.0, 1.5):
            d = make_variances(32, 48, gamma, slope, rng)
            pairs = rng.choice(32 * 31 // 2, size=30, replace=False)
            iu = np.triu_indices(32, 1)
            for p in pairs:
                i, k = iu[0][p], iu[1][p]
                a, b = d[i], d[k]
                f = float(np.sum((a - b) ** 2 / (a * b)))
                for eps2 in (0.005, 0.01, 0.02, 0.05):
                    e = rng.standard_normal((draws, 2, len(a))) * np.sqrt(eps2)
                    a_hat = a * np.clip(1 + e[:, 0], 0.05, None)
                    b_hat = b * np.clip(1 + e[:, 1], 0.05, None)
                    w = 1 / b_hat - 1 / a_hat
                    vw = (w**2 * a * b).sum(-1) / ((w * (a - b)).sum(-1)) ** 2
                    measured = float(np.mean(vw * f)) - 1.0
                    lead, second = plugin_excess(a, b, eps2)
                    predicted = lead if formula == "leading" else second
                    rel = abs(measured - predicted) / predicted
                    gated = lead <= 1.0 and eps2 <= 0.02
                    ok &= (rel <= 0.15) or not gated
                    out.append(
                        {
                            "gamma": gamma,
                            "slope": slope,
                            "F": f,
                            "eps2": eps2,
                            "measured_excess": measured,
                            "leading": lead,
                            "second_order": second,
                            "rel_error": rel,
                            "gated": gated,
                        }
                    )
    gated = [x for x in out if x["gated"]]
    return {
        "seed": seed,
        "formula": formula,
        "pairs": out,
        "n_gated": len(gated),
        "max_rel_error_gated": max(x["rel_error"] for x in gated),
        "median_rel_error_gated": float(np.median([x["rel_error"] for x in gated])),
        "pass": bool(ok),
    }


# ---------------------------------------------------------------------------------------------
# (c) Proposition 5.5: split-sample noise estimate and shrinkage factor
# ---------------------------------------------------------------------------------------------


def eb_parts(vf, vf_odd, w_full, w_odd, floor=1e-8):
    """Noise and shrinkage factor exactly as in Gimbal._shrunk_variances (batched over axis 0)."""
    m, n = vf.shape[-2:]
    v_hat = vf / w_full
    fl = floor * v_hat.mean(axis=(-2, -1), keepdims=True) + 1e-30
    log_v = np.log(v_hat + fl)
    additive = (
        log_v.mean(-1, keepdims=True)
        + log_v.mean(-2, keepdims=True)
        - log_v.mean(axis=(-2, -1), keepdims=True)
    )
    resid = log_v - additive
    ratio = np.log(vf_odd / w_odd + fl) - np.log(
        np.clip(vf - vf_odd, 0, None) / (w_full - w_odd) + fl
    )
    noise = ratio.var(axis=(-2, -1)) * 0.25 * (1 - 1 / m) * (1 - 1 / n)
    shrink = np.clip(1 - noise / np.maximum((resid**2).mean(axis=(-2, -1)), 1e-30), 0, 1)
    return log_v, noise, shrink, additive, resid


def part_c(reps: int = 300) -> dict:
    rng = np.random.default_rng(71)
    m, n = 32, 48
    out, ok = [], True
    fidelity = None
    for gamma in (0.0, 0.5, 1.0, 2.0):
        d = make_variances(m, n, gamma, 1.0, rng)
        log_d = np.log(d)
        r_true = log_d - log_d.mean(1, keepdims=True) - log_d.mean(0, keepdims=True) + log_d.mean()
        s_bar = float((r_true**2).mean())
        for dist in ("gaussian", "t5"):
            for beta in (0.98, 0.995):
                vf = np.zeros((reps, m, n))
                vf_odd = np.zeros((reps, m, n))
                w_odd = 0.0
                for t in range(1, STEPS + 1):
                    if dist == "gaussian":
                        z = rng.standard_normal((reps, m, n))
                    else:
                        z = rng.standard_t(5.0, size=(reps, m, n)) / np.sqrt(5.0 / 3.0)
                    z = z * np.sqrt(d)
                    vf = beta * vf + (1 - beta) * z * z
                    vf_odd *= beta
                    w_odd *= beta
                    if t % 2 == 1:
                        vf_odd += (1 - beta) * z * z
                        w_odd += 1 - beta
                    if t in (100, 400, 800):
                        w_full = 1 - beta**t
                        log_v, noise, shrink, _, _ = eb_parts(vf, vf_odd, w_full, w_odd)
                        nu_true = float(log_v.var(axis=0, ddof=1).mean())
                        nu_res = nu_true * (1 - 1 / m) * (1 - 1 / n)
                        nu_hat = float(noise.mean()) / ((1 - 1 / m) * (1 - 1 / n))
                        c_star = s_bar / (s_bar + nu_res)
                        rec = {
                            "gamma": gamma,
                            "dist": dist,
                            "beta": beta,
                            "t": t,
                            "nu_true": nu_true,
                            "nu_hat": nu_hat,
                            "rel_bias": nu_hat / nu_true - 1,
                            "c_mean": float(shrink.mean()),
                            "c_oracle": c_star,
                        }
                        if dist == "gaussian":
                            ok &= abs(rec["rel_bias"]) <= 0.15
                            ok &= abs(rec["c_mean"] - c_star) <= 0.1
                        out.append(rec)
                        if fidelity is None:  # same numbers as the optimizer's code path?
                            d_code = Gimbal._shrunk_variances(
                                torch.from_numpy(vf[0]),
                                torch.from_numpy(vf_odd[0]),
                                float(w_full),
                                float(w_odd),
                                1e-8,
                            ).numpy()
                            lv, _, sh, add, res = eb_parts(vf[:1], vf_odd[:1], w_full, w_odd)
                            d_here = np.exp(add[0] + sh[0] * res[0])
                            d_here *= np.exp(lv[0]).mean() / d_here.mean()
                            fidelity = float(np.abs(d_here / d_code - 1).max())
    return {
        "records": out,
        "fidelity_max_rel_diff_vs_gimbal_code": fidelity,
        "pass": bool(ok and fidelity is not None and fidelity < 1e-10),
    }


# ---------------------------------------------------------------------------------------------
# (d) Theorem 4.2: gap-independent contraction
# ---------------------------------------------------------------------------------------------


def part_d(alpha: float = 0.5, damping: float = 0.003, theta0: float = 1e-4) -> dict:
    rng = np.random.default_rng(81)
    out, ok = [], True
    for gamma, slope in ((0.0, 0.5), (0.0, 1.5), (1.0, 1.0), (2.0, 0.5)):
        d = make_variances(32, 48, gamma, slope, rng)
        m, n = d.shape
        f, _, iu = pair_quantities(d, "left")
        order = np.argsort(f)
        picks = order[np.linspace(0, len(order) - 1, 12).astype(int)]
        rows_sum = d.sum(1)
        for p in picks:
            i, k = iu[0][p], iu[1][p]
            ql = givens(m, i, k, theta0)
            ql_new, _, _, _ = population_step(ql, np.eye(n), d, alpha, damping)
            theta1 = float(ql_new[k, i])
            measured = theta1 / theta0
            predicted = 1 - alpha * f[p] / (f[p] + damping * n)
            # SOAP's refresh: columns sorted by decreasing eigenvalue estimate, then one
            # power-iteration step Q <- qr(L Q) on the exact pooled factor
            perm = np.argsort(-rows_sum)
            q0 = givens(m, i, k, theta0)[:, perm]
            q, rr = np.linalg.qr(np.diag(rows_sum) @ q0)
            q1 = np.empty_like(q)
            q1[:, perm] = q * np.sign(np.diag(rr))
            power = abs(float(q1[k, i])) / theta0
            ratio = min(rows_sum[i], rows_sum[k]) / max(rows_sum[i], rows_sum[k])
            rel = abs(measured - predicted) / predicted
            ok &= rel <= 0.10
            out.append(
                {
                    "gamma": gamma,
                    "slope": slope,
                    "F": float(f[p]),
                    "gimbal_measured": measured,
                    "gimbal_predicted": predicted,
                    "rel_error": rel,
                    "power_iteration": power,
                    "eigenvalue_ratio": ratio,
                }
            )
    fs = [x["F"] for x in out]
    span = max(fs) / min(fs)
    return {"pairs": out, "F_span": span, "pass": bool(ok and span >= 100)}


# ---------------------------------------------------------------------------------------------
# (e) Hessian of J at the optimum = Fisher; gradient of J = expected score
# ---------------------------------------------------------------------------------------------


def part_e(h: float = 1e-4) -> dict:
    rng = np.random.default_rng(91)
    d = make_variances(12, 16, 1.0, 1.0, rng)
    m, n = d.shape
    f_l, _, iu_l = pair_quantities(d, "left")
    f_r, _, iu_r = pair_quantities(d, "right")
    diag_err, cross = [], []

    def j_at(left: dict, right: dict) -> float:
        ql, qr = np.eye(m), np.eye(n)
        for (i, k), th in left.items():
            ql = ql @ givens(m, i, k, th)
        for (i, k), th in right.items():
            qr = qr @ givens(n, i, k, th)
        return frame_kl(ql, qr, d)

    for side, f, iu in (("left", f_l, iu_l), ("right", f_r, iu_r)):
        for p in range(len(f)):
            pair = (iu[0][p], iu[1][p])
            arg = (
                (lambda th, pair=pair: ({pair: th}, {}))
                if side == "left"
                else (lambda th, pair=pair: ({}, {pair: th}))
            )
            hpp = (j_at(*arg(h)) - 2 * j_at(*arg(0.0)) + j_at(*arg(-h))) / h**2
            diag_err.append(abs(hpp - f[p]) / f[p])
    # mixed second derivatives: two left pairs, and a left with a right pair
    for pa, pb in (((0, 1), (2, 3)), ((0, 1), (1, 2))):

        def jm(sa, sb, pa=pa, pb=pb):
            return j_at({pa: sa * h, pb: sb * h}, {})

        cross.append(abs(jm(1, 1) - jm(1, -1) - jm(-1, 1) + jm(-1, -1)) / (4 * h * h))

    def jlr(sa, sb):
        return j_at({(0, 1): sa * h}, {(0, 1): sb * h})

    cross.append(abs(jlr(1, 1) - jlr(1, -1) - jlr(-1, 1) + jlr(-1, -1)) / (4 * h * h))
    # gradient of J at a random frame against the expected score
    ql, qr = rand_orth(m, rng), rand_orth(n, rng)
    e_l, e_r, _, _ = expected_scores(ql, qr, d)
    grad_err = []
    for i, k in ((0, 1), (3, 7), (5, 11)):
        om = np.zeros((m, m))
        om[i, k], om[k, i] = 1e-6, -1e-6
        fd = (
            frame_kl(ql @ (np.eye(m) + om + om @ om / 2), qr, d)
            - frame_kl(ql @ (np.eye(m) - om + om @ om / 2), qr, d)
        ) / 2e-6
        grad_err.append(abs(fd - e_l[i, k]) / max(abs(e_l[i, k]), 1e-12))
    fmax = float(max(f_l.max(), f_r.max()))
    ok = max(diag_err) <= 1e-4 and max(cross) <= 1e-4 * fmax
    return {
        "max_rel_err_hessian_diag": max(diag_err),
        "max_abs_cross": max(cross),
        "max_F": fmax,
        "max_rel_err_gradient": max(grad_err),
        "pass": bool(ok),
    }


# ---------------------------------------------------------------------------------------------
# (f) Proposition 5.7: centering the frame statistic
# ---------------------------------------------------------------------------------------------


def part_f(seed: int = 63) -> dict:
    rng = np.random.default_rng(seed)
    b1, b2 = 0.9, 0.95
    # (f1) eta = sum of squared weights of the bias-corrected momentum (Lean: ema_weight_sq_sum),
    # against the variance of the optimizer's recursion applied to unit-variance noise
    eta_rows = []
    for horizon in (2, 5, 20, 100, 1000):
        m = np.zeros(200_000)
        for _ in range(horizon):
            m = b1 * m + (1 - b1) * rng.standard_normal(m.shape)
        mc = float(np.var(m / (1 - b1**horizon)))
        pred = float(ema_sum_sq_weights(b1, np.array([horizon]))[0])
        eta_rows.append(
            {"T": horizon, "eta": pred, "mc_variance": mc, "rel_err": abs(mc - pred) / pred}
        )
    ok1 = all(r["rel_err"] <= 0.03 for r in eta_rows)
    # (f2) mean skew score at the true frame (rotated coordinates, U* = I by equivariance) with the
    # plug-in mean c M_{t-1} built from t - 1 = 29 past gradients of the same law
    m_, n_, past, reps = 6, 8, 29, 100_000
    d = make_variances(m_, n_, 1.0, 1.0, rng)
    a = 1.0 / d
    theta = 0.7 * np.sqrt(d) * rng.standard_normal((m_, n_))
    bias0 = theta @ (theta * a).T
    bias0 = bias0 - bias0.T
    mom = np.zeros((reps, m_, n_))
    for _ in range(past):
        mom = b1 * mom + (1 - b1) * (theta + np.sqrt(d) * rng.standard_normal((reps, m_, n_)))
    mom /= 1 - b1**past
    w = np.sqrt(d) * rng.standard_normal((reps, m_, n_))
    score_rows = []
    for c in (0.0, 0.5, 1.0):
        z = theta + w - c * mom
        s_l = z @ np.swapaxes(z * a, 1, 2)
        mean = (s_l - np.swapaxes(s_l, 1, 2)).mean(axis=0)
        pred = (1 - c) ** 2 * bias0
        score_rows.append(
            {
                "c": c,
                "rel_err": float(np.linalg.norm(mean - pred) / np.linalg.norm(bias0)),
                "pred_norm_rel": float(np.linalg.norm(pred) / np.linalg.norm(bias0)),
            }
        )
    ok2 = all(r["rel_err"] <= 0.05 for r in score_rows)
    # (f3) the optimizer's plug-in factor (Gimbal._mean_shrinkage on buffers built by the
    # optimizer's recursions) against the oracle, 32x48 Gaussian streams, factor read at t = 200
    m_, n_, t_read, runs = 32, 48, 200, 100
    d = make_variances(m_, n_, 1.0, 1.0, rng)
    eta = float(ema_sum_sq_weights(b1, np.array([t_read - 1]))[0])
    direction = rng.standard_normal((m_, n_))
    direction /= np.linalg.norm(direction)
    factor_rows = []
    for snr, decay in ((0.0, 1.0), (0.1, 1.0), (1.0, 1.0), (10.0, 1.0), (100.0, 1.0), (10.0, 0.99)):
        mu0 = np.sqrt(snr * eta * d.sum()) * direction
        cs = []
        for _ in range(runs):
            mb = torch.zeros(m_, n_, dtype=torch.float64)
            vb = torch.zeros(m_, n_, dtype=torch.float64)
            for t in range(1, t_read):
                g = torch.from_numpy(mu0 * decay**t + np.sqrt(d) * rng.standard_normal((m_, n_)))
                mb.mul_(b1).add_(g, alpha=1 - b1)
                vb.mul_(b2).addcmul_(g, g, value=1 - b2)
            cs.append(float(Gimbal._mean_shrinkage(mb, vb, b1, b2, t_read)))
        mu_now = np.linalg.norm(mu0 * decay ** (t_read - 1)) ** 2
        oracle = mu_now / (mu_now + eta * d.sum())
        factor_rows.append(
            {
                "snr": snr,
                "decay": decay,
                "c_mean": float(np.mean(cs)),
                "c_sd": float(np.std(cs)),
                "oracle": float(oracle),
                "gated": decay == 1.0,
            }
        )
    ok3 = all(abs(r["c_mean"] - r["oracle"]) <= 0.1 for r in factor_rows if r["gated"])
    return {
        "eta": eta_rows,
        "score": score_rows,
        "factor": factor_rows,
        "pass_eta": bool(ok1),
        "pass_score": bool(ok2),
        "pass_factor": bool(ok3),
        "pass": bool(ok1 and ok2 and ok3),
    }


def main() -> None:
    res = {
        "a": part_a(),
        "b_preregistered": part_b(61, "leading"),
        "b": part_b(62, "second_order"),
        "c": part_c(),
        "d": part_d(),
        "e": part_e(),
        "f": part_f(),
    }
    res["G2.5"] = all(res[k]["pass"] for k in "abcdef")
    (RESULTS / "e210_theory_vs_simulation.json").write_text(json.dumps(res, indent=1))
    lines = [
        "# E2.10 theory–simulation agreement (generated by e210_theory_vs_simulation.py)",
        "",
        "## (a) Predicted vs measured frame KL (second-half mean, E2.1 seeds 10–19)",
        "",
        "Prediction: J ≈ ½ Σ_p F_p Var(θ_p) with Var(θ_p) = V_p Σ w² (Theorem 3.1) for the "
        "factor methods; the Cramér–Rao value ½ (#pairs) Σ w² for Gimbal. Ratio = measured / "
        "predicted at each method's best memory; `nonlin.` is the share of the pooled "
        "prediction carried by pairs whose predicted angle s.d. exceeds 0.15 rad; cells with "
        "share ≤ 0.15 are in the gate (C-009).",
        "",
        "| cell | nonlin. | pooled_eigh | soap | soap_rt | kl_eigh | klsoap | gimbal (vs CR) "
        "| gimbal_k4 (vs CR) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in res["a"]["cells"]:
        lin = c["nonlinear_share_pooled"]
        vals = []
        for mth in ("pooled_eigh", "soap", "soap_rt", "kl_eigh", "klsoap", "gimbal", "gimbal_k4"):
            x = c["methods"][mth]
            b = x["by_memory"][x["best_memory"]]
            vals.append(f"{b['ratio']:.2f}")
        lines.append(
            f"| γ={c['gamma']:g} s={c['slope']:g} {c['shape'][0]}×{c['shape'][1]} | "
            f"{lin:.3f} | " + " | ".join(vals) + " |"
        )
    lines += [
        "",
        f"Gate (i): pooled_eigh ratio in [0.75, 1.33] in every linear cell: "
        f"**{res['a']['pass']}** ({len(res['a']['gate_cells'])} linear cells).",
        "",
        "## (b) Lemma 5.4.2 plug-in inflation",
        "",
        "Pre-registered test (seed 61, leading-order formula s K / F): "
        f"{res['b_preregistered']['n_gated']} gated (pair, ε²) combinations, median relative"
        f" error {res['b_preregistered']['median_rel_error_gated']:.3f}, max "
        f"{res['b_preregistered']['max_rel_error_gated']:.3f} (tolerance 0.15): "
        f"**{res['b_preregistered']['pass']}** (F-013). After the theory revision (second-"
        f"order formula, theory v0.5) on fresh pairs (seed 62): median "
        f"{res['b']['median_rel_error_gated']:.3f}, max {res['b']['max_rel_error_gated']:.3f}"
        f": **{res['b']['pass']}**.",
        "",
        "## (c) Proposition 5.5 split-sample noise and shrinkage",
        "",
        "| γ | noise | β | t | ν true | ν̂ | rel. bias | mean c | oracle c* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["c"]["records"]:
        lines.append(
            f"| {r['gamma']:g} | {r['dist']} | {r['beta']} | {r['t']} | "
            f"{r['nu_true']:.4f} | {r['nu_hat']:.4f} | {r['rel_bias']:+.3f} | "
            f"{r['c_mean']:.3f} | {r['c_oracle']:.3f} |"
        )
    lines += [
        "",
        f"Fidelity to the optimizer's code path (max relative difference of D): "
        f"{res['c']['fidelity_max_rel_diff_vs_gimbal_code']:.1e}. Gate (iii) (Gaussian "
        f"rows): **{res['c']['pass']}**.",
        "",
        "## (d) Theorem 4.2 one-step contraction (α = 0.5, δ = 0.003)",
        "",
        f"F spans {res['d']['F_span']:.0f}×. Gimbal: measured vs 1 − αF/(F + δn); SOAP's "
        "power iteration contracts by the eigenvalue ratio of the pooled factor.",
        "",
        "| γ | s | F | Gimbal measured | predicted | power iteration | eigenvalue ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in res["d"]["pairs"]:
        lines.append(
            f"| {r['gamma']:g} | {r['slope']:g} | {r['F']:.3g} | "
            f"{r['gimbal_measured']:.4f} | {r['gimbal_predicted']:.4f} | "
            f"{r['power_iteration']:.4f} | {r['eigenvalue_ratio']:.4f} |"
        )
    e = res["e"]
    lines += [
        "",
        f"Gate (iv): **{res['d']['pass']}**.",
        "",
        "## (e) Hessian of J at the optimum and gradient of J",
        "",
        f"Max relative error of the Hessian diagonal against F: "
        f"{e['max_rel_err_hessian_diag']:.1e}; largest mixed second derivative "
        f"{e['max_abs_cross']:.1e} (max F {e['max_F']:.3g}); max relative error of the "
        f"gradient against the expected score: {e['max_rel_err_gradient']:.1e}. Gate (v): "
        f"**{e['pass']}**.",
        "",
        "## (f) Proposition 5.7: centering the frame statistic",
        "",
        "Momentum noise factor η (β₁ = 0.9) against the Monte Carlo variance of the "
        "bias-corrected momentum (200,000 draws):",
        "",
        "| T | η | Monte Carlo | rel. error |",
        "|---|---|---|---|",
    ]
    f = res["f"]
    lines += [
        f"| {r['T']} | {r['eta']:.5f} | {r['mc_variance']:.5f} | {r['rel_err']:.4f} |"
        for r in f["eta"]
    ]
    lines += [
        "",
        "Mean skew score at the true frame with plug-in mean c·M̂ (6×8, 29 past "
        "gradients, 100,000 draws), error relative to the uncentred bias; prediction "
        "(1 − c)² × uncentred bias:",
        "",
        "| c | predicted norm (rel.) | rel. error |",
        "|---|---|---|",
    ]
    lines += [f"| {r['c']:g} | {r['pred_norm_rel']:.3f} | {r['rel_err']:.4f} |" for r in f["score"]]
    lines += [
        "",
        "The optimizer's plug-in factor against the oracle (32×48, t = 200, 100 runs; "
        "SNR = ‖μ‖²/(η tr Σ); the decaying mean μ_t = μ₀·0.99^t is reported, not gated):",
        "",
        "| SNR (at t = 0) | decay | mean ĉ | s.d. | oracle c* |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['snr']:g} | {r['decay']:g} | {r['c_mean']:.3f} | {r['c_sd']:.3f} | "
        f"{r['oracle']:.3f} |"
        for r in f["factor"]
    ]
    lines += [
        "",
        f"Gate (vi): η {f['pass_eta']}, score bias {f['pass_score']}, plug-in factor "
        f"{f['pass_factor']}: **{f['pass']}**.",
        "",
        f"**G2.5: {res['G2.5']}**",
    ]
    (RESULTS / "e210_report.md").write_text("\n".join(lines))
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
