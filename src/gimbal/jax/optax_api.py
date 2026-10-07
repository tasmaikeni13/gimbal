"""Optax ``GradientTransformation`` wrappers with a common schema (Phase 03).

``adamw(lr, cfg)``, ``soap(lr, cfg)`` and ``gimbal(lr, cfg)`` take a learning rate (a float or an
Optax schedule of the step count) and the optimizer's config dataclass, act on a pytree of 2-D
float32 matrices, and return parameter increments, so ``optax.apply_updates`` performs the step.

These wrappers trace the step number, so the step kind (first step, warm start, frame move, QR
refresh) is chosen with ``lax.switch`` and the state keeps one fixed layout (Gimbal's warm-start
factors are kept, as zeros, after the warm start). The training loop instead resolves the kind in
Python and frees those buffers (:mod:`gimbal.jax.distributed`); both call the same step functions.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from . import adamw as _adamw
from . import gimbal as _gimbal
from . import soap as _soap

LearningRate = float | Callable[[jax.Array], jax.Array]


def _lr(learning_rate: LearningRate, count: jax.Array) -> jax.Array:
    return learning_rate(count) if callable(learning_rate) else jnp.asarray(learning_rate)


def _f32(x: jax.Array) -> jax.Array:
    """Optimizer arithmetic in float32 (float64 kept for tests); bf16 inputs are upcast."""
    return x if x.dtype == jnp.float64 else x.astype(jnp.float32)


def _switch(index: jax.Array, kinds: list, fn: Callable) -> object:
    return jax.lax.switch(index, [lambda k=k: fn(k) for k in kinds])


def adamw(
    learning_rate: LearningRate, cfg: _adamw.AdamWConfig | None = None
) -> optax.GradientTransformation:
    cfg = cfg or _adamw.AdamWConfig()

    def init(params):
        return {"count": jnp.zeros((), jnp.int32), "inner": jax.tree.map(_adamw.init_state, params)}

    def update(grads, state, params):
        t = state["count"] + 1
        lr = _lr(learning_rate, state["count"])
        out = jax.tree.map(
            lambda s, g, p: _adamw.step(s, _f32(g), _f32(p), lr, t, cfg),
            state["inner"],
            grads,
            params,
            is_leaf=lambda x: isinstance(x, dict) and "m" in x,
        )
        inner = jax.tree.map(lambda o: o[0], out, is_leaf=lambda x: isinstance(x, tuple))
        deltas = jax.tree.map(lambda o: o[1], out, is_leaf=lambda x: isinstance(x, tuple))
        return deltas, {"count": t, "inner": inner}

    return optax.GradientTransformation(init, update)


def _matrix_transform(
    init_one: Callable,
    step_one: Callable,
    kinds: list,
    kind_index: Callable,
    learning_rate: LearningRate,
    t_of: Callable,
    fix_layout: Callable,
) -> optax.GradientTransformation:
    def init(params):
        return {
            "count": jnp.zeros((), jnp.int32),
            "inner": [init_one(p) for p in jax.tree.leaves(params)],
        }

    def update(grads, state, params):
        call = state["count"] + 1
        lr = _lr(learning_rate, state["count"])
        g_leaves, treedef = jax.tree.flatten(grads)
        p_leaves = jax.tree.leaves(params)

        def run(kind):
            outs = [
                step_one(s, _f32(g), _f32(p), lr, t_of(call), kind)
                for s, g, p in zip(state["inner"], g_leaves, p_leaves, strict=True)
            ]
            return [fix_layout(o[0], s) for o, s in zip(outs, state["inner"], strict=True)], [
                o[1] for o in outs
            ]

        inner, deltas = _switch(kind_index(call), kinds, run)
        return jax.tree.unflatten(treedef, deltas), {"count": call, "inner": inner}

    return optax.GradientTransformation(init, update)


def soap(
    learning_rate: LearningRate, cfg: _soap.SOAPConfig | None = None
) -> optax.GradientTransformation:
    cfg = cfg or _soap.SOAPConfig()
    kinds = [_soap.Kind(init=a, refresh=b) for a, b in itertools.product((False, True), repeat=2)]

    def kind_index(call):
        t = call - 1
        refresh = jnp.logical_and(t > 0, t % cfg.precondition_frequency == 0)
        return 2 * (call == 1).astype(jnp.int32) + refresh.astype(jnp.int32)

    return _matrix_transform(
        lambda p: _soap.init_state(p.shape),
        lambda s, g, p, lr, t, k: _soap.step(s, g, p, lr, t, cfg, k),
        kinds,
        kind_index,
        learning_rate,
        t_of=lambda call: jnp.maximum(call - 1, 1),
        fix_layout=lambda new, old: new,
    )


def gimbal(
    learning_rate: LearningRate, cfg: _gimbal.GimbalConfig | None = None
) -> optax.GradientTransformation:
    cfg = cfg or _gimbal.GimbalConfig()
    # The move pattern is tabulated until it is periodic (fixed period frame_every once the
    # adaptive k_t has reached it and the warm start is over) and continued periodically after.
    horizon = max(cfg.warm_start_steps, 1) + 2 * cfg.frame_every + 2
    while _gimbal.frames_per_move(horizon, cfg) < cfg.frame_every:
        horizon *= 2
    horizon += 2 * cfg.frame_every
    table = [_gimbal.schedule(t, cfg) for t in range(1, horizon + 1)]
    last_move = max(t for t, kind in enumerate(table, start=1) if kind.move)
    moves = jnp.asarray([kind.move for kind in table])
    # Only kinds that occur are compiled (some bit combinations would not even trace).
    kinds = sorted(set(table))
    code = jnp.zeros(16, jnp.int32)
    for i, kind in enumerate(kinds):
        code = code.at[sum(int(b) << (3 - j) for j, b in enumerate(kind))].set(i)

    def kind_index(t):
        tw = cfg.warm_start_steps
        warm = tw > 1
        first = t == 1
        factors = jnp.logical_and(warm, jnp.logical_and(t > 1, t <= tw))
        restart = jnp.logical_and(warm, t == tw)
        periodic = (t - last_move) % cfg.frame_every == 0
        move = jnp.where(t <= horizon, moves[jnp.minimum(t, horizon) - 1], periodic)
        bits = (first, factors, restart, move)
        return code[sum(b.astype(jnp.int32) << (3 - i) for i, b in enumerate(bits))]

    def fix_layout(new, old):
        # Keep the warm-start buffers (as zeros once freed) so every branch has one layout.
        return {**{k: jnp.zeros_like(v) for k, v in old.items() if k not in new}, **new}

    return _matrix_transform(
        lambda p: _gimbal.init_state(p.shape, cfg),
        lambda s, g, p, lr, t, k: _gimbal.step(s, g, p, lr, t, cfg, k),
        kinds,
        kind_index,
        learning_rate,
        t_of=lambda call: call,
        fix_layout=fix_layout,
    )
