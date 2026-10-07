"""The training loop's distributed optimizer step against a direct per-matrix loop (float64).

The training loop (:class:`gimbal.train.train.Trainer`) stacks same-shaped matrices into padded
buckets sharded across devices (ZeRO-1), vmaps the per-matrix step over each device's share,
resolves the step kind in Python and frees Gimbal's warm-start buffers at the restart. This check
feeds a structured gradient stream for 120 steps (past Gimbal's warm start at step 50 and into
its adaptive schedule; across SOAP's refreshes) to that path and to a plain loop that calls the
same per-matrix step on one matrix at a time, both in float64, and reports the largest parameter
difference relative to the largest parameter change. (In float32 the two orderings of the same
arithmetic drift apart for Gimbal, whose first steps amplify rounding: F-025.) Runs on CPU with 4
simulated devices, so it is a separate process (the device count must be set before JAX starts);
``tests/test_distributed.py`` runs it.

Usage: python tests/check_distributed.py [--steps 120]
"""

from __future__ import annotations

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gimbal.jax import adamw as A  # noqa: E402
from gimbal.jax import distributed as dist  # noqa: E402
from gimbal.jax import gimbal as G  # noqa: E402
from gimbal.jax import soap as S  # noqa: E402
from gimbal.train.train import Trainer  # noqa: E402

LR = 3e-3


def config() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "base_125m.yaml").read_text())
    cfg["model"].update(
        vocab=64, d_model=16, n_layers=2, n_heads=2, d_ff=32, seq_len=8, attention="xla"
    )
    return cfg


def gradient_stream(trainer: Trainer, steps: int, seed: int) -> list[dict]:
    """Gradients with unequal variances and a non-zero mean, in the parameter layout."""
    rng = np.random.default_rng(seed)
    shapes = trainer.param_shapes
    scales = {k: rng.uniform(0.3, 3.0, size=shapes[k][-2:]) for k in dist.MATRIX_KEYS}
    out = []
    for _ in range(steps):
        g = {}
        for k, shape in shapes.items():
            x = rng.standard_normal(shape)
            if k in scales:
                x = x * scales[k] + 0.3 * rng.standard_normal(shape[-1])
            g[k] = 0.01 * x
        out.append(g)
    return out


def to_layout(trainer: Trainer, g: dict) -> dict:
    """Gradients in the training loop's layout: padded buckets and embedding rows sharded."""
    out = {}
    for b in trainer.buckets:
        flat = g[b.key].reshape((b.count,) + b.shape)
        pad = np.zeros((b.padded - b.count,) + b.shape)
        out[b.key] = jax.device_put(
            np.concatenate([flat, pad]), NamedSharding(trainer.mesh, P(dist.AXIS))
        )
    out["embed"] = jax.device_put(g["embed"], NamedSharding(trainer.mesh, P(dist.AXIS)))
    for k in dist.OTHER_KEYS[1:]:
        out[k] = jax.device_put(g[k], trainer.replicated)
    return out


def run(optimizer: str, steps: int) -> dict:
    trainer = Trainer(config(), optimizer)
    params, state = trainer.init(seed=0)

    def f64(x):
        return x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x

    params, state = jax.tree.map(f64, params), jax.tree.map(f64, state)
    p0 = {k: np.asarray(v) for k, v in params.items()}
    stream = gradient_stream(trainer, steps, seed=1)
    cfg = trainer.spec.matrix
    lr = jnp.float32(LR)  # as the training loop passes it (lr * weight_decay is float32)

    # Reference: one matrix at a time, plain loop.
    ref = {}
    for k in dist.MATRIX_KEYS:
        shape = trainer.param_shapes[k]
        n, mshape = int(np.prod(shape[:-2])), tuple(shape[-2:])
        if optimizer == "gimbal":
            states = [G.init_state(mshape, cfg, dtype=jnp.float64) for _ in range(n)]
        elif optimizer == "soap":
            states = [S.init_state(mshape, dtype=jnp.float64) for _ in range(n)]
        else:
            states = [A.init_state(jnp.zeros(mshape, jnp.float64)) for _ in range(n)]
        ref[k] = [states, [jnp.asarray(p0[k].reshape((n,) + mshape)[i]) for i in range(n)]]
    step_g = jax.jit(G.step, static_argnames=("cfg", "kind"))
    step_s = jax.jit(S.step, static_argnames=("cfg", "kind"))
    step_a = jax.jit(A.step, static_argnames=("cfg",))
    for t, g in enumerate(stream, start=1):
        state, params = trainer.update(state, to_layout(trainer, g), params, 0.0, LR, t)
        kind = trainer.spec.kind(t)
        for k in dist.MATRIX_KEYS:
            states, ps = ref[k]
            shape = trainer.param_shapes[k]
            gk = g[k].reshape((len(ps),) + tuple(shape[-2:]))
            for i in range(len(ps)):
                gi = jnp.asarray(gk[i])
                if optimizer == "gimbal":
                    st, d = step_g(states[i], gi, ps[i], lr, jnp.int32(t), cfg=cfg, kind=kind)
                    if kind.restart:
                        st = G.drop_warm_start_buffers(st)
                elif optimizer == "soap":
                    tt = jnp.int32(max(t - 1, 1))
                    st, d = step_s(states[i], gi, ps[i], lr, tt, cfg=cfg, kind=kind)
                else:
                    st, d = step_a(states[i], gi, ps[i], lr, jnp.int32(t), cfg=cfg)
                states[i], ps[i] = st, ps[i] + d
    out = {}
    for k in dist.MATRIX_KEYS:
        r = np.stack([np.asarray(p) for p in ref[k][1]]).reshape(trainer.param_shapes[k])
        moved = float(np.abs(r - p0[k]).max())
        diff = float(np.abs(np.asarray(params[k]) - r).max())
        out[k] = {"max_abs_diff": diff, "max_param_change": moved, "rel": diff / moved}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    assert len(jax.devices()) == 4, jax.devices()
    print(json.dumps({opt: run(opt, args.steps) for opt in ("adamw", "soap", "gimbal")}))


if __name__ == "__main__":
    main()
