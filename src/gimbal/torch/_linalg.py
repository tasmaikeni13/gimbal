"""Small linear-algebra helpers shared by the reference optimizers."""

from __future__ import annotations

import torch


def compute_dtype(g: torch.Tensor) -> torch.Tensor:
    """Gradient in a precision suitable for optimizer arithmetic (upcast half precision only)."""
    return g.float() if g.dtype in (torch.float16, torch.bfloat16) else g


def eye_like(n: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.eye(n, dtype=ref.dtype, device=ref.device)


def eigh_desc(sym: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvectors and eigenvalues of a symmetric matrix, sorted in descending order."""
    sym = 0.5 * (sym + sym.T)
    # Smallest normal number: no effect at ordinary scales, no scale dependence (F-014).
    jitter = torch.finfo(sym.dtype).tiny * eye_like(sym.shape[0], sym)
    evals, evecs = torch.linalg.eigh(sym + jitter)
    return evecs.flip(1), evals.flip(0)


def rotate(x: torch.Tensor, ql: torch.Tensor | None, qr: torch.Tensor | None) -> torch.Tensor:
    """Rotated coordinates ``Q_L^T X Q_R``; a missing side is the identity."""
    if ql is not None:
        x = ql.T @ x
    if qr is not None:
        x = x @ qr
    return x


def unrotate(x: torch.Tensor, ql: torch.Tensor | None, qr: torch.Tensor | None) -> torch.Tensor:
    """Inverse of :func:`rotate`: ``Q_L X Q_R^T``."""
    if ql is not None:
        x = ql @ x
    if qr is not None:
        x = x @ qr.T
    return x


def expm2(omega: torch.Tensor) -> torch.Tensor:
    """Second-order exponential retraction ``I + Ω + Ω²/2``.

    For skew Ω the orthogonality defect is exactly ``Ω⁴/4`` (Theorem 5.2).
    """
    return eye_like(omega.shape[0], omega) + omega + 0.5 * (omega @ omega)


def ns_polish(q: torch.Tensor) -> torch.Tensor:
    """One Newton–Schulz step towards the nearest orthogonal matrix.

    If ``QᵀQ = I + E`` the result satisfies ``YᵀY - I = -3/4 E² + 1/4 E³`` (Theorem 5.3).
    """
    return q @ (1.5 * eye_like(q.shape[1], q) - 0.5 * (q.T @ q))


def zeropower_ns5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Muon's quintic Newton–Schulz iteration (Jordan et al., 2024) in float32."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = compute_dtype(g)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


def sinkhorn_normalize(g: torch.Tensor, iters: int = 5, eps: float = 1e-12) -> torch.Tensor:
    """Simultaneous row/column l2 normalization used by ARO-Sinkhorn (arXiv:2602.09006, Alg. 1)."""
    x = g
    for _ in range(iters):
        row = x.norm(dim=1, keepdim=True).clamp_min(eps)
        col = x.norm(dim=0, keepdim=True).clamp_min(eps)
        x = x / row / col
    return x


def qr_orth(a: torch.Tensor) -> torch.Tensor:
    """Orthonormal factor of a QR decomposition."""
    q, _ = torch.linalg.qr(a)
    return q


def skew_spectral_norm(omega: torch.Tensor, iters: int = 8) -> float:
    """Power-iteration estimate of ``||Ω||_2`` (slightly inflated to be safe)."""
    n = omega.shape[0]
    idx = torch.arange(1, n + 1, dtype=omega.dtype, device=omega.device)
    v = torch.sin(1.618 * idx)  # fixed, deterministic start vector
    v = v / v.norm()
    sigma = torch.zeros((), dtype=omega.dtype)
    for _ in range(iters):
        w = omega.T @ (omega @ v)
        sigma = w.norm().sqrt()
        v = w / (w.norm() + 1e-30)
    return float(1.1 * sigma)


def polish_until_orthogonal(q: torch.Tensor, max_iters: int = 4) -> torch.Tensor:
    """Newton–Schulz polish, repeated while the orthogonality defect is above working precision."""
    tol = 1e-12 if q.dtype == torch.float64 else 1e-6
    eye = eye_like(q.shape[1], q)
    for _ in range(max_iters):
        gram = q.T @ q
        if float((gram - eye).abs().max()) < tol:
            break
        q = q @ (1.5 * eye - 0.5 * gram)
    return q
