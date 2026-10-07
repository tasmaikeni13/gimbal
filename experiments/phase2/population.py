"""Noise-free (population) quantities of the KRD model, used by E2.10 and E2.11.

The true frame is the identity, without loss of generality (equivariance, Theorem 6). For a frame
estimate U = (Q_L, Q_R) write P = Q_L^T and R = Q_R^T, so that the rotated gradient is
Z = P (sqrt(D) * N) R^T. Then

    E[Z_ij Z_kj] = sum_a P_ia P_ka M_L[a, j],   M_L = D (R o R)^T,
    E[Z_ij Z_il] = sum_b R_jb R_lb M_R[i, b],   M_R = (P o P) D,
    d_U = E[Z o Z] = (P o P) D (R o R)^T,

and the expected scores E[S_L - S_L^T], E[S_R - S_R^T] of the flow (Lemma A) with the profile
variances d_U have the closed forms below. They are also the Riemannian gradient of the frame KL
J(U) = 1/2 sum log d_U - 1/2 sum log D in the skew coordinates (checked in E2.10 (e)).
"""

from __future__ import annotations

import numpy as np


def frame_variances(ql: np.ndarray, qr: np.ndarray, d: np.ndarray) -> np.ndarray:
    p2, r2 = ql.T**2, qr.T**2
    return p2 @ d @ r2.T


def frame_kl(ql: np.ndarray, qr: np.ndarray, d: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.log(frame_variances(ql, qr, d)) - np.log(d)))


def expected_scores(ql: np.ndarray, qr: np.ndarray, d: np.ndarray):
    """Expected left/right skew scores at U with D_hat = d_U, plus d_U and A = 1/d_U."""
    p, r = ql.T, qr.T
    du = (p * p) @ d @ (r * r).T
    a = 1.0 / du
    b_l = (d @ (r * r).T) @ a.T  # B_L[a, k] = sum_j M_L[a, j] A_kj
    t_l = p @ (b_l * p.T)  # E[S_L]
    b_r = ((p * p) @ d).T @ a  # B_R[b, l] = sum_i M_R[i, b] A_il
    t_r = r @ (b_r * r.T)  # E[S_R]
    return t_l - t_l.T, t_r - t_r.T, du, a


def fisher(du: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = du.shape
    f_l = du @ a.T
    f_r = du.T @ a
    return np.clip(f_l + f_l.T - 2 * n, 0, None), np.clip(f_r + f_r.T - 2 * m, 0, None)


def generator(
    score: np.ndarray,
    fis: np.ndarray,
    groups: int,
    alpha: float,
    damping: float,
    max_angle: float = 0.25,
    max_rotation: float = 1.0,
) -> np.ndarray:
    """Same rule as ``Gimbal._generator``: damped natural gradient with the trust-region caps."""
    om = -alpha * score / (fis + damping * groups)
    np.fill_diagonal(om, 0.0)
    om = np.clip(om, -max_angle, max_angle)
    norm = np.linalg.norm(om, 2)
    if norm > max_rotation:
        om *= max_rotation / norm
    return om


def expm2(om: np.ndarray) -> np.ndarray:
    return np.eye(len(om)) + om + 0.5 * om @ om


def polish(q: np.ndarray, tol: float = 1e-14, max_iter: int = 6) -> np.ndarray:
    eye = np.eye(q.shape[1])
    for _ in range(max_iter):
        gram = q.T @ q
        if np.abs(gram - eye).max() < tol:
            break
        q = q @ (1.5 * eye - 0.5 * gram)
    return q


def population_step(
    ql: np.ndarray,
    qr: np.ndarray,
    d: np.ndarray,
    alpha: float,
    damping: float,
    max_angle: float = 0.25,
    max_rotation: float = 1.0,
):
    """One step of the noise-free flow; returns the new frame and the scores at the old one."""
    e_l, e_r, du, a = expected_scores(ql, qr, d)
    f_l, f_r = fisher(du, a)
    m, n = d.shape
    om_l = generator(e_l, f_l, n, alpha, damping, max_angle, max_rotation)
    om_r = generator(e_r, f_r, m, alpha, damping, max_angle, max_rotation)
    return polish(ql @ expm2(om_l)), polish(qr @ expm2(om_r)), e_l, e_r


def givens(n: int, i: int, k: int, theta: float) -> np.ndarray:
    """Rotation by ``theta`` in the (i, k) plane: G[k, i] = sin(theta)."""
    g = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    g[i, i] = g[k, k] = c
    g[k, i], g[i, k] = s, -s
    return g


def pair_quantities(d: np.ndarray, side: str = "left"):
    """Per-pair Fisher information F and pooled-factor variance V_1 (Theorems 2 and 3).

    Returns arrays over pairs (i < k) of rows (``left``) or columns (``right``), and the pairs.
    """
    x = d if side == "left" else d.T
    a, b = x[:, None, :], x[None, :, :]
    f = ((a - b) ** 2 / (a * b)).sum(-1)
    with np.errstate(divide="ignore"):
        v1 = (a * b).sum(-1) / ((a - b).sum(-1)) ** 2
    iu = np.triu_indices(len(x), 1)
    return f[iu], v1[iu], iu


def weighted_variance(d: np.ndarray, w_cols: np.ndarray, side: str = "left") -> np.ndarray:
    """V_w for every pair with fixed weights over the other mode (Theorem 3.1)."""
    x = d if side == "left" else d.T
    a, b = x[:, None, :], x[None, :, :]
    num = (w_cols**2 * a * b).sum(-1)
    den = (w_cols * (a - b)).sum(-1) ** 2
    iu = np.triu_indices(len(x), 1)
    with np.errstate(divide="ignore"):
        return (num / den)[iu]


def kl_fixed_point(d: np.ndarray, iters: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ell, r) of KL-Shampoo's population factors in the true frame:
    ell_i = sum_j D_ij / r_j / n, r_j = sum_i D_ij / ell_i / m (a Sinkhorn-type scaling)."""
    m, n = d.shape
    ell, r = np.ones(m), np.ones(n)
    for _ in range(iters):
        ell = (d / r).sum(1) / n
        r = (d / ell[:, None]).sum(0) / m
    return ell, r


def ema_sum_sq_weights(beta: float, t: np.ndarray) -> np.ndarray:
    """Sum of squared normalized weights of an EMA over t samples (weights ~ beta^(t-s))."""
    t = np.asarray(t, dtype=float)
    return (1 - beta) * (1 + beta**t) / ((1 + beta) * (1 - beta**t))
