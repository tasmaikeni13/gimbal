"""SOAP (Vyas et al., arXiv:2409.11321) in JAX for one matrix, following the official code.

Same rules as ``gimbal.torch.SOAP`` (``realtime=False``) and the vendored official implementation:
the first call only initializes the Kronecker factors and the frame (``eigh``); afterwards Adam
runs in the frame, the factors are updated after the parameter step, and every
``precondition_frequency`` steps the frame is refreshed by one power-iteration step and a QR
decomposition, after sorting the frame by estimated eigenvalues and re-ordering the second moment
accordingly.

The official code re-expresses the momentum in the frame at every step (rotate back, then rotate
again). With an unchanged frame this is the identity up to rounding, so here it happens only at
refresh steps; the same economy is given to Gimbal (momentum transported only when its frame
moves). As for Gimbal, the step kind is resolved in Python (:func:`schedule`) so that the compiled
steady-state step has no QR and no branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._linalg import eigh_desc, mm, qr_orth, rotate, unrotate


@dataclass(frozen=True)
class SOAPConfig:
    """Hyper-parameters (official defaults; ``shampoo_beta < 0`` means ``b2``)."""

    b1: float = 0.95
    b2: float = 0.95
    shampoo_beta: float = -1.0
    eps: float = 1e-8
    weight_decay: float = 0.01
    precondition_frequency: int = 10

    @property
    def sb(self) -> float:
        """Factor memory: ``shampoo_beta``, or ``b2`` when it is negative (official convention)."""
        return self.shampoo_beta if self.shampoo_beta >= 0 else self.b2


class Kind(NamedTuple):
    """Static kind of a SOAP call, resolved outside ``jit`` (see :func:`schedule`)."""

    init: bool  # first call: initialize factors and frame, no parameter update
    refresh: bool  # refresh the frame after this step's factor update


def schedule(call: int, cfg: SOAPConfig) -> Kind:
    """Kind of the ``call``-th call (1-based). Adam's step counter is ``call - 1``."""
    t = call - 1
    return Kind(init=call == 1, refresh=t > 0 and t % cfg.precondition_frequency == 0)


def init_state(shape: tuple[int, int], dtype=jnp.float32) -> dict:
    """State before the first call for an ``m x n`` matrix: Adam's moments in the frame,
    the Kronecker factors and the identity frame.
    """
    m, n = shape
    z = jnp.zeros(shape, dtype)
    return {
        "exp_avg": z,
        "exp_avg_sq": z,
        "GG_L": jnp.zeros((m, m), dtype),
        "GG_R": jnp.zeros((n, n), dtype),
        "QL": jnp.eye(m, dtype=dtype),
        "QR": jnp.eye(n, dtype=dtype),
    }


def _refresh(gg: jax.Array, q: jax.Array, v: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """Sort by estimated eigenvalues, re-order ``V`` along ``axis``, one power step + QR."""
    est = jnp.sum(q * mm(gg, q), axis=0)  # diag(Qᵀ GG Q)
    order = jnp.argsort(-est)
    v = jnp.take(v, order, axis=axis)
    return qr_orth(mm(gg, q[:, order])), v


def step(
    state: dict,
    g: jax.Array,
    p: jax.Array,
    lr: jax.Array,
    t: jax.Array,
    cfg: SOAPConfig,
    kind: Kind,
) -> tuple[dict, jax.Array]:
    """One SOAP call for one matrix; ``t`` is Adam's step counter after this call (traced).

    Returns the new state and the parameter increment ``Δp``.
    """
    state = dict(state)
    sb = cfg.sb
    if kind.init:
        state["GG_L"] = (1.0 - sb) * mm(g, g.T)
        state["GG_R"] = (1.0 - sb) * mm(g.T, g)
        state["QL"], state["QR"] = eigh_desc(state["GG_L"]), eigh_desc(state["GG_R"])
        return state, jnp.zeros_like(p)
    b1, b2 = cfg.b1, cfg.b2
    tf = t.astype(g.dtype)
    ql, qr = state["QL"], state["QR"]
    g_rot = rotate(g, ql, qr)
    exp_avg = b1 * state["exp_avg"] + (1.0 - b1) * g_rot
    exp_avg_sq = b2 * state["exp_avg_sq"] + (1.0 - b2) * g_rot * g_rot
    step_size = lr * jnp.sqrt(1.0 - b2**tf) / (1.0 - b1**tf)
    update = unrotate(exp_avg / (jnp.sqrt(exp_avg_sq) + cfg.eps), ql, qr)
    p1 = p - step_size * update
    delta = (p1 - lr * cfg.weight_decay * p1) - p  # official order: weight decay after the step
    # Factors are updated after the step, with the current gradient.
    state["GG_L"] = state["GG_L"] + (1.0 - sb) * (mm(g, g.T) - state["GG_L"])
    state["GG_R"] = state["GG_R"] + (1.0 - sb) * (mm(g.T, g) - state["GG_R"])
    if kind.refresh:
        m_orig = unrotate(exp_avg, ql, qr)
        ql, exp_avg_sq = _refresh(state["GG_L"], ql, exp_avg_sq, axis=0)
        qr, exp_avg_sq = _refresh(state["GG_R"], qr, exp_avg_sq, axis=1)
        exp_avg = rotate(m_orig, ql, qr)
        state["QL"], state["QR"] = ql, qr
    state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq
    return state, delta
