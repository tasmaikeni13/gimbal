"""Shared utilities for the Phase 02 experiments.

* Gradient streams from the KRD model (theory §1) with controllable non-separability, ties,
  drift and heavy tails.
* The frame-KL metric J (Proposition 1) evaluated exactly against the true frame.
* Frame extraction from the *actual* optimizer states, so that the frames being scored are the
  ones the optimizers use.
* Paired statistics: bootstrap intervals, Wilcoxon signed-rank, Holm correction.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.stats
import torch

from gimbal.torch import KLSOAP, SOAP, Gimbal

# ---------------------------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------------------------


def rand_orth(n: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def make_variances(m: int, n: int, gamma: float, slope: float, rng: np.random.Generator,
                   tie: bool = False) -> np.ndarray:
    """Variance array with log D = a_i + b_j + gamma * c_ij (power-law margins, slope s).

    gamma = 0 is the separable (Kronecker-product) case. With ``tie=True`` rows come in pairs that
    share their row sums but have different profiles (Example 1 family), which pooled factors cannot
    separate.
    """
    a = -slope * np.log(np.arange(1, m + 1))
    b = -slope * np.log(np.arange(1, n + 1))
    a, b = rng.permutation(a), rng.permutation(b)
    log_d = a[:, None] + b[None, :] + gamma * rng.standard_normal((m, n))
    d = np.exp(log_d)
    if tie:
        for i in range(0, m - 1, 2):
            d[i + 1] = d[i][rng.permutation(n)]  # same multiset of values -> same row sum
    return d / d.mean()


def nonseparability_index(d: np.ndarray, n_samples: int | None = None) -> float:
    """Share of the variance of log D not explained by the additive model a_i + b_j.

    If ``n_samples`` is given, the estimation noise of log-variances from that many samples
    (≈ 2 / n_samples for Gaussian entries) is subtracted.
    """
    ld = np.log(d)
    resid = ld - ld.mean(1, keepdims=True) - ld.mean(0, keepdims=True) + ld.mean()
    total = ld.var()
    noise = 2.0 / n_samples if n_samples else 0.0
    return float(max(resid.var() - noise, 0.0) / max(total - noise, 1e-12))


@dataclass
class KRDStream:
    """Gradients G = Q_L (sqrt(D) * N) Q_R^T, optionally with drifting frames and t noise."""

    ql: np.ndarray
    qr: np.ndarray
    d: np.ndarray
    rng: np.random.Generator
    nu: float | None = None  # Student-t degrees of freedom (None: Gaussian)
    drift: float = 0.0  # rotation per step (radians, spectral norm of the generator)

    def __post_init__(self) -> None:
        self.step_l = self.step_r = None
        if self.drift > 0:
            self.step_l = self._drift_rotation(self.ql.shape[0])
            self.step_r = self._drift_rotation(self.qr.shape[0])

    def _drift_rotation(self, n: int) -> np.ndarray:
        k = self.rng.standard_normal((n, n))
        k = k - k.T
        k /= np.linalg.norm(k, 2)
        return scipy.linalg.expm(self.drift * k)

    def noise(self) -> np.ndarray:
        shape = self.d.shape
        if self.nu is None:
            return self.rng.standard_normal(shape)
        t = self.rng.standard_t(self.nu, size=shape)
        return t / math.sqrt(self.nu / (self.nu - 2.0))  # unit variance

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            yield self.ql @ (np.sqrt(self.d) * self.noise()) @ self.qr.T
            if self.step_l is not None:
                self.ql = self.ql @ self.step_l
                self.qr = self.qr @ self.step_r


# ---------------------------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------------------------


def frame_kl(ql_hat: np.ndarray | None, qr_hat: np.ndarray | None, ql: np.ndarray,
             qr: np.ndarray, d: np.ndarray) -> float:
    """J(U_hat) = 1/2 sum(log D_tilde - log D), D_tilde = (P_L o P_L) D (P_R o P_R)^T."""
    pl = (ql_hat.T @ ql) if ql_hat is not None else ql
    pr = (qr_hat.T @ qr) if qr_hat is not None else qr
    d_tilde = (pl * pl) @ d @ (pr * pr).T
    return 0.5 * float(np.sum(np.log(d_tilde) - np.log(d)))


# ---------------------------------------------------------------------------------------------
# Frame estimators = the real optimizers run with lr = 0
# ---------------------------------------------------------------------------------------------

FRAME_METHODS = {
    "soap": lambda p, mem: SOAP([p], lr=0.0, weight_decay=0.0, shampoo_beta=mem,
                                precondition_frequency=10),
    "soap_rt": lambda p, mem: SOAP([p], lr=0.0, weight_decay=0.0, shampoo_beta=mem, realtime=True),
    "klsoap": lambda p, mem: KLSOAP([p], lr=0.0, beta_kron=mem),
    # Gimbal with the current defaults (C-004, C-013): bias-corrected rate, damping 0.003, flow
    # variance tied to the frame memory with empirical-Bayes shrinkage, pooled warm start, frame
    # statistics on the empirical-Bayes innovation. "gimbal_k4" is the default (C-012);
    # "gimbal" pins frame_every = 1.
    "gimbal": lambda p, mem: Gimbal([p], lr=0.0, rot_rate=mem, frame_every=1),
    "gimbal_k4": lambda p, mem: Gimbal([p], lr=0.0, rot_rate=mem, frame_every=4),
    # Historical ablations, run before C-013 and pinned to its uncentered statistics: C-003
    # without shrinkage; v0.3 (C-002 only: Adam's V as flow variance, eigh initialization).
    "gimbal_noshrink": lambda p, mem: Gimbal([p], lr=0.0, rot_rate=mem, flow_shrink=False,
                                             frame_every=1, flow_center=False),
    "gimbal_v03": lambda p, mem: Gimbal([p], lr=0.0, rot_rate=mem, flow_beta=None, init="eigh",
                                        frame_every=1, flow_center=False),
}

# Memory grids. For EMA methods the value is the EMA coefficient (memory ~ 2/(1-beta) samples);
# for Gimbal it is the rotation rate alpha (memory ~ (2-alpha)/alpha samples, Theorem 4).
MEMORY_GRID = {
    "soap": [0.9, 0.95, 0.99, 0.995, 0.998, 0.999],
    "soap_rt": [0.9, 0.95, 0.99, 0.995, 0.998, 0.999],
    "klsoap": [0.9, 0.95, 0.99, 0.995, 0.998],
    "pooled_eigh": [0.9, 0.95, 0.99, 0.995, 0.998, 0.999],
    "kl_eigh": [0.9, 0.95, 0.99, 0.995, 0.998],
    "gimbal": [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1],
    "gimbal_k4": [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1],
    "gimbal_noshrink": [0.0025, 0.005, 0.01, 0.05],
    "gimbal_v03": [0.01],
}
# Grid extensions, applied to every method alike when best memories sit on a grid edge: longer
# memory for stationary suites, shorter for drift (experiments ledger, E2.1).
MEMORY_EXTENSION = {
    "long": {"soap": [0.9999], "soap_rt": [0.9999], "pooled_eigh": [0.9999], "klsoap": [0.999],
             "kl_eigh": [0.999], "gimbal": [1e-4], "gimbal_k4": [1e-4]},
    "short": {"soap": [0.8], "soap_rt": [0.8], "pooled_eigh": [0.8], "klsoap": [0.8],
              "kl_eigh": [0.8], "gimbal": [0.2], "gimbal_k4": [0.2]},
}


def full_grid(extension: str | None) -> dict[str, list[float]]:
    """MEMORY_GRID merged with one extension (``None``: the base grid)."""
    ext = MEMORY_EXTENSION.get(extension, {}) if extension else {}
    return {m: sorted(set(v) | set(ext.get(m, []))) for m, v in MEMORY_GRID.items()}


# Equal effective sample size: EMA beta=0.99 averages ~2/(1-beta)=200 samples; the flow with rate
# alpha behaves like ~(2-alpha)/alpha samples (Theorem 4), so alpha=0.01 is the matched setting.
MATCHED_MEMORY = {"soap": 0.99, "soap_rt": 0.99, "klsoap": 0.99, "pooled_eigh": 0.99,
                  "kl_eigh": 0.99, "gimbal": 0.01, "gimbal_k4": 0.01}


def eigh_desc_np(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, v = np.linalg.eigh(0.5 * (s + s.T))
    return v[:, ::-1], w[::-1]


def exact_factor_frames(kind: str, beta: float, grads: list[np.ndarray]) -> Iterator[tuple]:
    """Controls: exact eigenvectors of the pooled ("pooled_eigh") or KL ("kl_eigh") factors at
    every step. Not practical optimizers; they isolate statistical efficiency (Theorem 3) from the
    one-step power-iteration lag of SOAP's frame refresh."""
    m, n = grads[0].shape
    lf, rf = np.zeros((m, m)), np.zeros((n, n))
    ql, lam_l = np.eye(m), np.ones(m)
    qr, lam_r = np.eye(n), np.ones(n)
    if kind == "kl_eigh":  # identity factors at the first gradient's scale, as in KLSOAP (F-018)
        sigma = float(np.sqrt(np.mean(grads[0] ** 2)))
        lf, rf = sigma * np.eye(m), sigma * np.eye(n)
        lam_l, lam_r = sigma * np.ones(m), sigma * np.ones(n)
    for g in grads:
        if kind == "pooled_eigh":
            lf = beta * lf + (1 - beta) * g @ g.T
            rf = beta * rf + (1 - beta) * g.T @ g
        else:
            fl = lam_l.clip(min=0) + 1e-6 * lam_l.clip(min=0).mean() + 1e-300
            fr = lam_r.clip(min=0) + 1e-6 * lam_r.clip(min=0).mean() + 1e-300
            r_inv = (qr / fr) @ qr.T
            l_inv = (ql / fl) @ ql.T
            lf = beta * lf + (1 - beta) * (g @ r_inv @ g.T) / n
            rf = beta * rf + (1 - beta) * (g.T @ l_inv @ g) / m
        ql, lam_l = eigh_desc_np(lf)
        qr, lam_r = eigh_desc_np(rf)
        yield ql, qr


def get_frames(method: str, opt: torch.optim.Optimizer, p: torch.Tensor):
    st = opt.state[p]
    if method in ("soap", "soap_rt"):
        ql, qr = st["Q"]
    else:
        ql, qr = st["QL"], st["QR"]
    to_np = lambda q: None if q is None else q.detach().double().numpy()  # noqa: E731
    return to_np(ql), to_np(qr)


def frame_trace(method: str, mem: float, grads: list[np.ndarray], ql: np.ndarray,
                qr: np.ndarray, d: np.ndarray, every: int = 1,
                true_frames: list[tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    """Run one frame estimator on a fixed gradient list and return J after each step."""
    m, n = d.shape
    if method in ("pooled_eigh", "kl_eigh"):
        out = []
        for t, (ql_hat, qr_hat) in enumerate(exact_factor_frames(method, mem, grads)):
            if t % every == 0:
                tq = true_frames[t] if true_frames is not None else (ql, qr)
                out.append(frame_kl(ql_hat, qr_hat, tq[0], tq[1], d))
        return np.asarray(out)
    p = torch.nn.Parameter(torch.zeros(m, n, dtype=torch.float64))
    opt = FRAME_METHODS[method](p, mem)
    out = []
    for t, g in enumerate(grads):
        p.grad = torch.from_numpy(g)
        opt.step()
        if t % every == 0:
            ql_hat, qr_hat = get_frames(method, opt, p)
            tq = true_frames[t] if true_frames is not None else (ql, qr)
            out.append(frame_kl(ql_hat, qr_hat, tq[0], tq[1], d))
    return np.asarray(out)


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 20000, level: float = 0.95,
                        seed: int = 0) -> tuple[float, float, float]:
    """Mean of paired differences with a percentile bootstrap interval."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(diff.mean()), float(lo), float(hi)


def wilcoxon_less(x: np.ndarray, y: np.ndarray) -> float:
    """One-sided paired Wilcoxon p-value for H1: x < y."""
    diff = np.asarray(x) - np.asarray(y)
    if np.allclose(diff, 0):
        return 1.0
    return float(scipy.stats.wilcoxon(diff, alternative="less").pvalue)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    k = len(items)
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (k - rank) * p))
        adjusted[name] = running
    return adjusted
