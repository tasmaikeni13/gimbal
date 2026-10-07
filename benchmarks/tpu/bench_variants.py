"""Where does the gradient step spend its time? Ablations timed on the full slice (Phase 04).

Variants of the per-device computation inside ``shard_map`` (no optimizer):
forward only; forward + backward without gradient communication; + float32 reduce-scatter (the
training step); the same with bfloat16 logits; without the output head (loss on the hidden state).
"""

from __future__ import annotations

import json
import os
import pathlib

import jax
import jax.numpy as jnp
import yaml
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from bench_components import timeit

    from gimbal.jax import distributed as dist
    from gimbal.train import model as M
    from gimbal.train.data import TokenFile, global_batch, train_order
    from gimbal.train.train import Trainer

    cfg = yaml.safe_load((ROOT / "configs/base_125m.yaml").read_text())
    data = TokenFile(ROOT / cfg["data"]["train"])
    tr = Trainer(cfg, "adamw")
    params, _ = tr.init(0)
    batch = global_batch(data, train_order(data.n_seq, 0)[:240], tr.batch_sharding)
    mcfg = tr.model_cfg
    pspec = {k: P() for k in tr.param_shapes}

    def hidden_loss(p, b):
        x = M.forward_hidden(p, b[:, :-1], mcfg)
        return jnp.mean(x.astype(jnp.float32) ** 2)

    def bf16_logits_loss(p, b):
        x = M.forward_hidden(p, b[:, :-1], mcfg)
        logits = jnp.einsum("btd,vd->btv", x, p["embed"].astype(M.COMPUTE))
        logz = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
        gold = jnp.take_along_axis(logits, b[:, 1:, None], axis=-1)[..., 0]
        return jnp.mean(logz - gold.astype(jnp.float32))

    full = lambda p, b: M.loss_fn(p, b, mcfg)  # noqa: E731
    variants = {
        "forward_only": (lambda p, b: jax.lax.pmean(full(p, b), "data"), P()),
        "fwd_bwd_local": (lambda p, b: jax.grad(full)(p, b), pspec),
        "fwd_bwd_reduce_scatter": (
            lambda p, b: dist.reduce_gradients(jax.grad(full)(p, b), tr.buckets, tr.n_dev),
            tr._grad_specs(),
        ),
        "fwd_bwd_local_bf16_logits": (lambda p, b: jax.grad(bf16_logits_loss)(p, b), pspec),
        "fwd_bwd_local_no_head": (lambda p, b: jax.grad(hidden_loss)(p, b), pspec),
    }
    out = {}
    if os.environ.get("BENCH_MODEL_SET"):  # e.g. "attention=xla,scan_layers=false"
        import dataclasses

        sets = dict(kv.split("=") for kv in os.environ["BENCH_MODEL_SET"].split(","))
        mcfg = dataclasses.replace(mcfg, **{k: yaml.safe_load(v) for k, v in sets.items()})
        full = lambda p, b: M.loss_fn(p, b, mcfg)  # noqa: E731
        variants = {
            "forward_only": (lambda p, b: jax.lax.pmean(full(p, b), "data"), P()),
            "fwd_bwd_local": (lambda p, b: jax.grad(full)(p, b), pspec),
        }
    for name, (fn, ospec) in variants.items():
        f = jax.jit(
            shard_map(
                fn,
                mesh=tr.mesh,
                in_specs=(pspec, P("data", None)),
                out_specs=ospec,
                check_rep=False,
            )
        )
        out[name] = timeit(f, params, batch)
        print(name, out[name], flush=True)
    if os.environ.get("GIMBAL_WORKER", "0") == "0":
        tag = os.environ.get("BENCH_MODEL_SET", "").replace("=", "_").replace(",", "__")
        (ROOT / f"benchmarks/tpu/results/variants{('_' + tag) if tag else ''}.json").write_text(
            json.dumps(out, indent=1)
        )


if __name__ == "__main__":
    main()
