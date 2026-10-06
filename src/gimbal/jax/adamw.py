"""AdamW (Loshchilov & Hutter, 2019) as a pure per-tensor step, matching ``torch.optim.AdamW``.

Used for every parameter of the AdamW baseline and, with identical settings for every method, for
the embeddings, norms and output head of the matrix optimizers (Phase 03 routing).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AdamWConfig:
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0


def init_state(x: jax.Array) -> dict:
    dtype = jnp.float64 if x.dtype == jnp.float64 else jnp.float32
    return {"m": jnp.zeros_like(x, dtype), "v": jnp.zeros_like(x, dtype)}


def step(state: dict, g: jax.Array, p: jax.Array, lr: jax.Array, t: jax.Array,
         cfg: AdamWConfig) -> tuple[dict, jax.Array]:
    """One step; ``t`` is the 1-based step number. Returns the state and ``Δp``.

    Decoupled weight decay is applied as in PyTorch: ``p ← p (1 − lr·wd)`` before the Adam step.
    """
    tf = t.astype(g.dtype)
    m = cfg.b1 * state["m"] + (1.0 - cfg.b1) * g
    v = cfg.b2 * state["v"] + (1.0 - cfg.b2) * g * g
    m_hat = m / (1.0 - cfg.b1**tf)
    v_hat = v / (1.0 - cfg.b2**tf)
    delta = -lr * cfg.weight_decay * p - lr * m_hat / (jnp.sqrt(v_hat) + cfg.eps)
    return {"m": m, "v": v}, delta
