"""Linear-algebra helpers for the JAX optimizers (mirrors ``gimbal.torch._linalg``).

Every helper acts on one matrix; the optimizers batch them over same-shaped layers with
``jax.vmap``. All products run at ``Precision.HIGHEST``: TPUs otherwise multiply float32 inputs in a
single bfloat16 pass, which is too coarse for frames that must stay orthogonal to ~1e-6
(Phase 04 precision policy, applied identically to every optimizer).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST
def tiny(dtype) -> float:
    """Smallest normal number of ``dtype`` (scale-free guard against log(0) and 0/0; F-014)."""
    return float(jnp.finfo(dtype).tiny)


def mm(a: jax.Array, b: jax.Array) -> jax.Array:
    """Float32 matrix product at full precision."""
    return jnp.matmul(a, b, precision=HIGHEST)


def rotate(x: jax.Array, ql: jax.Array, qr: jax.Array) -> jax.Array:
    """Rotated coordinates ``Q_Lᵀ X Q_R``."""
    return mm(mm(ql.T, x), qr)


def unrotate(x: jax.Array, ql: jax.Array, qr: jax.Array) -> jax.Array:
    """Inverse of :func:`rotate`: ``Q_L X Q_Rᵀ``."""
    return mm(mm(ql, x), qr.T)


def eigh_desc(sym: jax.Array) -> jax.Array:
    """Eigenvectors of a symmetric matrix, ordered by descending eigenvalue."""
    sym = 0.5 * (sym + sym.T)
    # Smallest normal number: no effect at ordinary scales and no scale dependence (F-014).
    sym = sym + tiny(sym.dtype) * jnp.eye(sym.shape[0], dtype=sym.dtype)
    _, vecs = jnp.linalg.eigh(sym)
    return vecs[:, ::-1]


def expm2(omega: jax.Array) -> jax.Array:
    """Second-order exponential retraction ``I + Ω + Ω²/2`` (defect ``Ω⁴/4``, Theorem 5.2)."""
    return jnp.eye(omega.shape[0], dtype=omega.dtype) + omega + 0.5 * mm(omega, omega)


def skew_spectral_norm(omega: jax.Array, iters: int = 8) -> jax.Array:
    """Power-iteration estimate of ``‖Ω‖₂``, inflated by 10% to be safe (as in the reference).

    Starts from the column of Ω with the largest norm, which makes the estimate invariant under
    ``Ω → S Ω S`` for diagonal sign matrices ``S`` (eigenvector sign gauge; F-024, C-015).
    """
    v = omega[:, jnp.argmax(jnp.linalg.norm(omega, axis=0))]
    v = v / (jnp.linalg.norm(v) + 1e-30)

    def body(_, carry):
        v, _ = carry
        w = jnp.matmul(omega.T, jnp.matmul(omega, v, precision=HIGHEST), precision=HIGHEST)
        wn = jnp.linalg.norm(w)
        return w / (wn + 1e-30), jnp.sqrt(wn)

    _, sigma = jax.lax.fori_loop(0, iters, body, (v, jnp.zeros((), omega.dtype)))
    return 1.1 * sigma


def polish_until_orthogonal(q: jax.Array, max_iters: int = 4) -> jax.Array:
    """Newton–Schulz polish, repeated while ``max|QᵀQ − I|`` exceeds working precision (Thm 5.3).

    Same rule as the reference: at most ``max_iters`` steps, each applied only while the defect is
    above 1e-6 (float32) or 1e-12 (float64). Under ``vmap`` the loop runs until every matrix is
    done, and a finished matrix is left unchanged.
    """
    tol = 1e-12 if q.dtype == jnp.float64 else 1e-6
    eye = jnp.eye(q.shape[1], dtype=q.dtype)

    def cond(carry):
        i, _, done = carry
        return jnp.logical_and(i < max_iters, jnp.logical_not(done))

    def body(carry):
        i, q, _ = carry
        gram = mm(q.T, q)
        done = jnp.max(jnp.abs(gram - eye)) < tol
        q = jnp.where(done, q, mm(q, 1.5 * eye - 0.5 * gram))
        return i + 1, q, done

    _, q, _ = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), q, jnp.array(False)))
    return q


def qr_orth(a: jax.Array) -> jax.Array:
    """Orthonormal factor of a Householder QR decomposition."""
    q, _ = jnp.linalg.qr(a)
    return q
