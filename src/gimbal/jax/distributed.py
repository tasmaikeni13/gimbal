"""Shape-bucketed, sharded optimizer step shared by AdamW, SOAP and Gimbal (Phase 04).

The model keeps its hidden matrices in stacks of equal shape (``[L, 4, d, d]`` attention,
``[L, 3, d, f]`` MLP; :mod:`gimbal.train.model`). Here each stack is viewed as a bucket of
``count`` matrices, padded to a multiple of the device count, and the bucket axis is sharded across
the devices: each device runs the optimizer on its own few matrices with batched matrix products
(``vmap``), and only the parameter increments are all-gathered (ZeRO-1). Every optimizer gets the
same layout, batching and sharding. Padded slots receive zero gradients; the optimizers are
zero-gradient safe (F-026) and the slots' increments are discarded.

Embeddings and norms use AdamW for every optimizer (Phase 03 routing); the embedding's state is
sharded along the vocabulary, the norms' state is replicated.

Everything here runs inside ``shard_map`` over the mesh axis ``"data"``: functions receive the
local shard of the optimizer state and of the gradient, and the full (replicated) parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import adamw as _adamw
from . import gimbal as _gimbal
from . import soap as _soap

AXIS = "data"
MATRIX_KEYS = ("attn", "mlp")
OTHER_KEYS = ("embed", "norm_attn", "norm_mlp", "norm_f")


@dataclass(frozen=True)
class Bucket:
    key: str
    count: int  # matrices in the stack
    shape: tuple[int, int]  # (m, n) of each matrix
    padded: int  # count rounded up to a multiple of the device count
    per_device: int


def make_buckets(param_shapes: dict, n_dev: int) -> tuple[Bucket, ...]:
    out = []
    for key in MATRIX_KEYS:
        shape = param_shapes[key]
        count = 1
        for s in shape[:-2]:
            count *= s
        padded = -(-count // n_dev) * n_dev
        out.append(Bucket(key, count, tuple(shape[-2:]), padded, padded // n_dev))
    return tuple(out)


@dataclass(frozen=True)
class OptimizerSpec:
    """Matrix optimizer (``"adamw"``, ``"soap"`` or ``"gimbal"``) and the shared AdamW for the
    remaining parameters (no weight decay: decoupled decay applies to hidden matrices only)."""

    name: str
    matrix: object
    other: _adamw.AdamWConfig = field(default_factory=lambda: _adamw.AdamWConfig())

    def kind(self, t: int):
        if self.name == "soap":
            return _soap.schedule(t, self.matrix)
        if self.name == "gimbal":
            return _gimbal.schedule(t, self.matrix)
        return None


# ---------------------------------------------------------------------------------------------
# State layout
# ---------------------------------------------------------------------------------------------


def _matrix_state(spec: OptimizerSpec, bucket: Bucket, k: int, warm: bool = True) -> dict:
    """State of ``k`` matrices of one bucket (leading axis ``k``)."""
    m, n = bucket.shape
    if spec.name == "adamw":
        one = _adamw.init_state(jnp.zeros((m, n), jnp.float32))
    elif spec.name == "soap":
        one = _soap.init_state((m, n))
    else:
        one = _gimbal.init_state((m, n), spec.matrix)
        if not warm:
            one = _gimbal.drop_warm_start_buffers(one)
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (k,) + x.shape), one)


def init_state(spec: OptimizerSpec, buckets: tuple[Bucket, ...], params: dict) -> dict:
    """Global (unsharded) initial state; place it with :func:`state_specs`."""
    return {
        "matrix": {b.key: _matrix_state(spec, b, b.padded) for b in buckets},
        "other": {k: _adamw.init_state(params[k]) for k in OTHER_KEYS},
    }


def state_specs(state: dict) -> dict:
    """PartitionSpecs: bucket axis sharded; embedding state along the vocabulary; norms
    replicated."""
    return {
        "matrix": jax.tree.map(lambda x: P(AXIS), state["matrix"]),
        "other": {
            k: jax.tree.map(lambda x, k=k: P(AXIS) if k == "embed" else P(), v)
            for k, v in state["other"].items()
        },
    }


def drop_gimbal_warm_buffers(state: dict) -> dict:
    return {
        **state,
        "matrix": {k: _gimbal.drop_warm_start_buffers(v) for k, v in state["matrix"].items()},
    }


# ---------------------------------------------------------------------------------------------
# Inside shard_map
# ---------------------------------------------------------------------------------------------


def _bucket_view(x: jax.Array, bucket: Bucket) -> jax.Array:
    """``[..., m, n]`` stack -> ``[padded, m, n]`` (zero padding)."""
    x = x.reshape((bucket.count,) + bucket.shape)
    if bucket.padded > bucket.count:
        x = jnp.concatenate([x, jnp.zeros((bucket.padded - bucket.count,) + bucket.shape, x.dtype)])
    return x


def reduce_gradients(grads: dict, buckets: tuple[Bucket, ...], n_dev: int) -> dict:
    """Mean over devices, in float32, scattered to the optimizer layout (ZeRO-1)."""
    out = {}
    for b in buckets:
        g = _bucket_view(grads[b.key].astype(jnp.float32), b)
        out[b.key] = jax.lax.psum_scatter(g, AXIS, scatter_dimension=0, tiled=True) / n_dev
    out["embed"] = (
        jax.lax.psum_scatter(
            grads["embed"].astype(jnp.float32), AXIS, scatter_dimension=0, tiled=True
        )
        / n_dev
    )
    for k in OTHER_KEYS[1:]:
        out[k] = jax.lax.psum(grads[k].astype(jnp.float32), AXIS) / n_dev
    return out


def global_norm(local: dict) -> jax.Array:
    """Norm of the full gradient from its local shards (norms are replicated: counted once)."""
    sharded = sum(jnp.sum(local[k] * local[k]) for k in (*MATRIX_KEYS, "embed"))
    replicated = sum(jnp.sum(local[k] * local[k]) for k in OTHER_KEYS[1:])
    return jnp.sqrt(jax.lax.psum(sharded, AXIS) + replicated)


def _local_slice(x: jax.Array, size: int) -> jax.Array:
    idx = jax.lax.axis_index(AXIS)
    return jax.lax.dynamic_slice_in_dim(x, idx * size, size, axis=0)


def optimizer_update(
    spec: OptimizerSpec,
    kind,
    state: dict,
    grads: dict,
    params: dict,
    buckets: tuple[Bucket, ...],
    n_dev: int,
    lr: jax.Array,
    t: jax.Array,
) -> tuple:
    """One optimizer step on the local shards; returns the new local state and new params."""
    new_state = {"matrix": {}, "other": {}}
    new_params = dict(params)
    for b in buckets:
        p_local = _local_slice(_bucket_view(params[b.key], b), b.per_device)
        st, g = state["matrix"][b.key], grads[b.key]
        if spec.name == "adamw":
            st, delta = _adamw.step(st, g, p_local, lr, t, spec.matrix)
        elif spec.name == "soap":
            # Adam's counter lags the call count: the first call only initializes (official).
            step = partial(_soap.step, lr=lr, t=jnp.maximum(t - 1, 1), cfg=spec.matrix, kind=kind)
            st, delta = jax.vmap(step)(st, g, p_local)
        else:
            step = partial(_gimbal.step, lr=lr, t=t, cfg=spec.matrix, kind=kind)
            st, delta = jax.vmap(step)(st, g, p_local)
        new_state["matrix"][b.key] = st
        full = jax.lax.all_gather(delta, AXIS, axis=0, tiled=True)[: b.count]
        new_params[b.key] = params[b.key] + full.reshape(params[b.key].shape)
    for k in OTHER_KEYS:
        p_k = _local_slice(params[k], params[k].shape[0] // n_dev) if k == "embed" else params[k]
        st, delta = _adamw.step(state["other"][k], grads[k], p_k, lr, t, spec.other)
        new_state["other"][k] = st
        if k == "embed":
            delta = jax.lax.all_gather(delta, AXIS, axis=0, tiled=True)
        new_params[k] = params[k] + delta
    return new_state, new_params
