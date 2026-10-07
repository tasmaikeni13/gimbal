"""Gimbal in JAX: the algorithm of ``gimbal.torch.Gimbal`` (defaults of theory §5) for one matrix.

The step is a pure function of one ``m x n`` matrix, so the training step can batch it over
same-shaped layers with ``jax.vmap`` and shard that batch across devices (Phase 04). Control flow
that depends only on the step number (first step, pooled warm start, frame moves every
``frame_every`` steps) is resolved in Python by :func:`schedule`, so the compiled steady-state step
contains no eigendecomposition and no branch. :mod:`gimbal.jax.optax_api` wraps the same functions
in a single-function Optax transformation.

Two layout choices differ from the reference only by rounding (``tests/test_jax_optimizers.py``
checks agreement):

* the momentum is kept in rotated coordinates of the current frame and re-expressed (unrotate with
  the old frame, rotate with the new one) when the frame moves. This saves one rotation per step;
  centering (Proposition 5.7) needs exactly this rotated previous momentum;
* the Fisher matrix is formed only at steps where the frame moves (the reference forms it every
  step and uses the last one).

Only the defaults of the reference are implemented (``frame_schedule`` both ways,
``rot_schedule="bias_corrected"``,
``flow_beta="tied"``, ``flow_shrink=True``, ``flow_center="adaptive"``, ``init="pooled"``,
``transport=False``, ``polish_every=1``); both sides must be preconditioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._linalg import (
    eigh_desc,
    expm2,
    mm,
    polish_until_orthogonal,
    rotate,
    skew_spectral_norm,
    tiny,
    unrotate,
)


@dataclass(frozen=True)
class GimbalConfig:
    """Hyper-parameters (names and defaults as in ``gimbal.torch.Gimbal``)."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    rot_rate: float = 0.05  # = 1 - b2: one window for frame, flow variances and V (C-018)
    rot_rate_max: float = 0.5
    damping: float = 0.003
    floor: float = 1e-8
    max_angle: float = 0.25
    max_rotation: float = 1.0
    frame_every: int = 4
    frame_schedule: str = "adaptive"  # k_t = clamp(round(K·α/α_t), 1, K); "fixed": K (C-018)
    warm_start_steps: int = 50
    polish_max_iters: int = 4


class Kind(NamedTuple):
    """What a step does besides the Adam update; a static (compile-time) choice."""

    first: bool  # t = 1: frames from the first gradient's factors
    factors: bool  # 1 < t <= T_w: accumulate the pooled warm-start factors
    restart: bool  # t = T_w: restart from the pooled factors, no flow move (Remark 5.6)
    move: bool  # the flow takes its (amortized) frame step


def scheduled_rate(t: int, cfg: GimbalConfig) -> float:
    """Bias-corrected rotation rate ``α_t`` (Theorem 4.4), as the reference computes it."""
    a = cfg.rot_rate
    return min(cfg.rot_rate_max, a / (1 - (1 - a) ** t)) if a > 0 else 0.0


def frames_per_move(t: int, cfg: GimbalConfig) -> int:
    """Scores accumulated per frame move at step ``t``: ``frame_every``, or with the adaptive
    schedule ``clamp(round(K·α/α_t), 1, K)``, which keeps ``k·α_t`` near its steady-state value
    (the amortized step matches the per-step flow to first order in ``k·α``)."""
    k = cfg.frame_every
    if cfg.frame_schedule == "adaptive" and k > 1 and cfg.rot_rate > 0:
        k = int(min(k, max(1, round(k * cfg.rot_rate / scheduled_rate(t, cfg)))))
    return k


_MOVES: dict = {}


def schedule(t: int, cfg: GimbalConfig) -> Kind:
    """Kind of step ``t`` (1-based), reproducing the reference's counters.

    The score accumulator moves the frame when it holds ``frames_per_move(t)`` scores; it is
    emptied by each move and by the warm start, whose step does not enter the flow.
    """
    tw = cfg.warm_start_steps
    warm = tw > 1
    restart = warm and t == tw
    moves = _MOVES.setdefault(cfg, [])  # moves[s - 1]: does step s move the frame?
    count = 0
    if moves:
        # resume the counter from the last cached step
        count = _MOVES[(cfg, "count")]
    while len(moves) < t:
        s = len(moves) + 1
        if warm and s == tw:
            count = 0
            moves.append(False)
            continue
        count += 1
        move = count >= frames_per_move(s, cfg)
        if move:
            count = 0
        moves.append(move)
    _MOVES[(cfg, "count")] = count
    return Kind(first=t == 1, factors=warm and 1 < t <= tw, restart=restart, move=moves[t - 1])


def init_state(shape: tuple[int, int], cfg: GimbalConfig, dtype=jnp.float32) -> dict:
    """State before the first step. ``L_acc``/``R_acc`` exist until the warm start."""
    m, n = shape
    z = jnp.zeros(shape, dtype)
    state = {
        "M": z,
        "V": z,
        "VF": z,
        "VF_odd": z,
        "w_odd": jnp.zeros((), dtype),
        "QL": jnp.eye(m, dtype=dtype),
        "QR": jnp.eye(n, dtype=dtype),
        "acc_L": jnp.zeros((m, m), dtype),
        "acc_R": jnp.zeros((n, n), dtype),
        "keep": jnp.ones((), dtype),
        "count": jnp.zeros((), dtype),
    }
    if cfg.warm_start_steps > 1:
        state["L_acc"] = jnp.zeros((m, m), dtype)
        state["R_acc"] = jnp.zeros((n, n), dtype)
    return state


def drop_warm_start_buffers(state: dict) -> dict:
    """State layout after the warm start (the pooled factors are freed)."""
    return {k: v for k, v in state.items() if k not in ("L_acc", "R_acc")}


def _mean_shrinkage(
    m_prev: jax.Array, v_prev: jax.Array, t: jax.Array, cfg: GimbalConfig
) -> jax.Array:
    """James–Stein factor of the previous momentum as an estimate of the mean (Prop. 5.7).

    ``||M||²`` and ``Σ V`` do not depend on the frame, so the rotated momentum gives the same
    factor as the reference's parameter-space momentum.
    """
    b1, b2 = cfg.b1, cfg.b2
    tm1 = (t - 1).astype(m_prev.dtype)
    bc1 = 1.0 - b1**tm1
    eta = (1.0 - b1) * (1.0 + b1**tm1) / ((1.0 + b1) * bc1)
    s_m = jnp.sum(m_prev * m_prev) / bc1**2
    s_v = jnp.sum(v_prev) / (1.0 - b2**tm1)
    separable = 1.0 - eta >= 1e-6  # one gradient (t = 2) or b1 = 0: mean and noise inseparable
    noise = eta * jnp.maximum(s_v - s_m, 0.0) / jnp.where(separable, 1.0 - eta, 1.0)
    # A zero momentum makes the factor irrelevant (c·M = 0). The reference divides by
    # max(S_M, tiny); under jit XLA may merge that division with the one above into a product that
    # flushes to zero, so the zero case is taken out explicitly.
    nonzero = s_m > 0.0
    c = jnp.clip(1.0 - noise / jnp.where(nonzero, s_m, 1.0), 0.0, 1.0)
    return jnp.where(jnp.logical_and(separable, nonzero), c, 0.0)


def _shrunk_variances(
    vf: jax.Array, vf_odd: jax.Array, w_full: jax.Array, w_odd: jax.Array, floor: float
) -> jax.Array:
    """Empirical-Bayes variances for the flow (Proposition 5.5), as in the reference."""
    m, n = vf.shape
    v_hat = vf / w_full
    fl = floor * jnp.mean(v_hat) + tiny(vf.dtype)
    log_v = jnp.log(v_hat + fl)
    additive = (
        jnp.mean(log_v, axis=1, keepdims=True)
        + jnp.mean(log_v, axis=0, keepdims=True)
        - jnp.mean(log_v)
    )
    resid = log_v - additive
    w_even = w_full - w_odd
    split = jnp.logical_and(w_odd > 0.0, w_even > 1e-12 * w_full)
    ratio = jnp.log(vf_odd / jnp.where(split, w_odd, 1.0) + fl) - jnp.log(
        jnp.maximum(vf - vf_odd, 0.0) / jnp.where(split, w_even, 1.0) + fl
    )
    noise = jnp.var(ratio) * (0.25 * (1.0 - 1.0 / m) * (1.0 - 1.0 / n))
    shrink = jnp.clip(1.0 - noise / jnp.maximum(jnp.mean(resid * resid), 1e-30), 0.0, 1.0)
    shrink = jnp.where(split, shrink, 0.0)
    # Algebraically the reference's exp(x)·mean/mean(exp(x)). XLA flushes subnormals to zero
    # (PyTorch on CPU keeps them), so the exponent is taken relative to its maximum and the result
    # is kept at or above the smallest normal number: for a zero gradient the variances are that
    # number, and 1/D must stay finite.
    x = additive + shrink * resid
    e = jnp.exp(x - jnp.max(x))
    return jnp.maximum(e * (jnp.mean(v_hat + fl) / jnp.mean(e)), tiny(vf.dtype))


def separability_stats(
    vf: jax.Array, vf_odd: jax.Array, w_full: jax.Array, w_odd: jax.Array, floor: float
) -> dict:
    """Diagnostics of the flow's variance average (not used by the update): the share of the
    variance of log V̂^F outside its additive fit (raw κ), the same share after subtracting the
    split-sample noise estimate (noise-corrected κ, the E3.1 statistic), and the shrinkage factor c
    of Proposition 5.5."""
    m, n = vf.shape
    v_hat = vf / w_full
    fl = floor * jnp.mean(v_hat) + tiny(vf.dtype)
    log_v = jnp.log(v_hat + fl)
    additive = (
        jnp.mean(log_v, axis=1, keepdims=True)
        + jnp.mean(log_v, axis=0, keepdims=True)
        - jnp.mean(log_v)
    )
    resid_sq = jnp.mean((log_v - additive) ** 2)
    ratio = jnp.log(vf_odd / w_odd + fl) - jnp.log(
        jnp.maximum(vf - vf_odd, 0.0) / (w_full - w_odd) + fl
    )
    noise = jnp.var(ratio) * (0.25 * (1.0 - 1.0 / m) * (1.0 - 1.0 / n))
    total = jnp.maximum(jnp.var(log_v), 1e-30)
    return {
        "kappa_raw": resid_sq / total,
        "kappa_noise_corrected": jnp.maximum(resid_sq - noise, 0.0) / total,
        "shrink_c": jnp.clip(1.0 - noise / jnp.maximum(resid_sq, 1e-30), 0.0, 1.0),
    }


def _skew_score(z: jax.Array, za: jax.Array, side: str) -> jax.Array:
    """Skew score ``S − Sᵀ`` of Lemma A, ``S_L = Z (Z∘A)ᵀ`` or ``S_R = Zᵀ (Z∘A)``."""
    s = mm(z, za.T) if side == "left" else mm(z.T, za)
    return s - s.T


def _fisher(d: jax.Array, a: jax.Array, side: str) -> tuple[jax.Array, int]:
    """Fisher information per pair (Theorem 2) and the number of groups it sums over."""
    if side == "left":
        groups, f = d.shape[1], mm(d, a.T)
    else:
        groups, f = d.shape[0], mm(d.T, a)
    return jnp.maximum(f + f.T - 2.0 * groups, 0.0), groups


def _generator(
    score: jax.Array, fisher: jax.Array, groups: int, rate: jax.Array, cfg: GimbalConfig
) -> jax.Array:
    """Damped natural-gradient generator with the entry clip and the spectral trust region."""
    omega = -rate * score / (fisher + cfg.damping * groups)
    omega = omega * (1.0 - jnp.eye(omega.shape[0], dtype=omega.dtype))
    omega = jnp.clip(omega, -cfg.max_angle, cfg.max_angle)
    norm = skew_spectral_norm(omega)
    return jnp.where(norm > cfg.max_rotation, omega * (cfg.max_rotation / norm), omega)


def step(
    state: dict,
    g: jax.Array,
    p: jax.Array,
    lr: jax.Array,
    t: jax.Array,
    cfg: GimbalConfig,
    kind: Kind,
) -> tuple[dict, jax.Array]:
    """One Gimbal step for one matrix.

    Parameters
    ----------
    state: from :func:`init_state` (with ``L_acc``/``R_acc`` while ``t <= warm_start_steps``).
    g, p: gradient and parameter, ``m x n`` float32.
    lr: learning rate of this step. ``t``: 1-based step number (traced).
    kind: :func:`schedule` ``(t, cfg)``.

    Returns the new state and the parameter increment ``Δp`` (``p ← p + Δp``).
    """
    b1, b2 = cfg.b1, cfg.b2
    tf = t.astype(g.dtype)
    state = dict(state)
    if kind.first:
        # Frames from the first gradient's factors; the pooled sums start with them.
        ggt, gtg = mm(g, g.T), mm(g.T, g)
        state["QL"], state["QR"] = eigh_desc(ggt), eigh_desc(gtg)
        if "L_acc" in state:
            state["L_acc"], state["R_acc"] = ggt, gtg
    ql, qr = state["QL"], state["QR"]
    z = rotate(g, ql, qr)
    m_prev = state["M"]

    # Adam in the current frame; the momentum lives in rotated coordinates.
    if kind.first:
        z_flow, g_flow = z, g
    else:
        # Innovation G_t − c·M_{t−1} (Proposition 5.7, C-013).
        scale = _mean_shrinkage(m_prev, state["V"], t, cfg) / (1.0 - b1 ** (tf - 1.0))
        z_flow = z - scale * m_prev
        g_flow = g - scale * unrotate(m_prev, ql, qr) if kind.factors else None
    m_new = b1 * m_prev + (1.0 - b1) * z
    v_new = b2 * state["V"] + (1.0 - b2) * z * z
    v_hat = v_new / (1.0 - b2**tf)
    n_rot = (m_new / (1.0 - b1**tf)) / (jnp.sqrt(v_hat) + cfg.eps)
    delta = -lr * cfg.weight_decay * p - lr * unrotate(n_rot, ql, qr)
    state["M"], state["V"] = m_new, v_new

    # Pooled warm start (Remark 5.6).
    if kind.factors:
        state["L_acc"] = state["L_acc"] + mm(g_flow, g_flow.T)
        state["R_acc"] = state["R_acc"] + mm(g_flow.T, g_flow)
    if kind.restart:
        new_l, new_r = eigh_desc(state["L_acc"]), eigh_desc(state["R_acc"])
        move_l, move_r = mm(ql.T, new_l), mm(qr.T, new_r)
        sq_l, sq_r = move_l * move_l, move_r * move_r
        for key in ("V", "VF", "VF_odd"):
            state[key] = mm(mm(sq_l.T, state[key]), sq_r)  # Theorem 7 transport
        state["M"] = mm(mm(move_l.T, m_new), move_r)
        state["QL"], state["QR"] = ql, qr = new_l, new_r
        z_flow = rotate(g_flow, ql, qr)
        state = drop_warm_start_buffers(state)
        state["acc_L"] = jnp.zeros_like(state["acc_L"])
        state["acc_R"] = jnp.zeros_like(state["acc_R"])
        state["keep"] = jnp.ones_like(state["keep"])
        state["count"] = jnp.zeros_like(state["count"])

    # Variances seen by the flow: tied memory, split into odd and even steps (Prop. 5.5).
    beta_d = 1.0 - cfg.rot_rate
    zf2 = z_flow * z_flow
    state["VF"] = beta_d * state["VF"] + (1.0 - beta_d) * zf2
    odd = (t % 2 == 1).astype(g.dtype)
    state["VF_odd"] = beta_d * state["VF_odd"] + odd * (1.0 - beta_d) * zf2
    state["w_odd"] = state["w_odd"] * beta_d + odd * (1.0 - beta_d)
    if kind.restart:
        return state, delta
    d = _shrunk_variances(state["VF"], state["VF_odd"], 1.0 - beta_d**tf, state["w_odd"], cfg.floor)

    # One accumulated natural-gradient step of the likelihood (Section 5, amortized flow).
    alpha = cfg.rot_rate
    if alpha > 0:
        alpha = jnp.minimum(cfg.rot_rate_max, alpha / (1.0 - (1.0 - alpha) ** tf))
    a = 1.0 / d
    za = z_flow * a
    state["keep"] = state["keep"] * (1.0 - alpha)
    state["count"] = state["count"] + 1.0
    acc_l = state["acc_L"] + _skew_score(z_flow, za, "left")
    acc_r = state["acc_R"] + _skew_score(z_flow, za, "right")
    if not kind.move:
        state["acc_L"], state["acc_R"] = acc_l, acc_r
        return state, delta
    rate = 1.0 - state["keep"]
    k = state["count"]  # scores in this block (frame_every, or k_t when adaptive)
    new_q = {}
    for key, side, acc in (("QL", "left", acc_l), ("QR", "right", acc_r)):
        fisher, groups = _fisher(d, a, side)
        omega = _generator(acc / k, fisher, groups, rate, cfg)
        new_q[key] = polish_until_orthogonal(mm(state[key], expm2(omega)), cfg.polish_max_iters)
    # Re-express the momentum in the new frame (it is a parameter-space quantity).
    state["M"] = rotate(unrotate(state["M"], ql, qr), new_q["QL"], new_q["QR"])
    state["QL"], state["QR"] = new_q["QL"], new_q["QR"]
    state["acc_L"], state["acc_R"] = jnp.zeros_like(acc_l), jnp.zeros_like(acc_r)
    state["keep"] = jnp.ones_like(state["keep"])
    state["count"] = jnp.zeros_like(state["count"])
    return state, delta
