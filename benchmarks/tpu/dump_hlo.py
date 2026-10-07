"""Write the optimized HLO of the gradient and update programs and list their collectives
(Phase 04 task 2: no unintended all-gathers of optimizer state)."""

from __future__ import annotations

import collections
import os
import pathlib
import re
import sys

import jax
import jax.numpy as jnp
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "soap"
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from gimbal.train.data import TokenFile, global_batch, train_order
    from gimbal.train.train import Trainer

    cfg = yaml.safe_load((ROOT / "configs/base_125m.yaml").read_text())
    data = TokenFile(ROOT / cfg["data"]["train"])
    tr = Trainer(cfg, name)
    params, state = tr.init(0)
    batch = global_batch(data, train_order(data.n_seq, 0)[:240], tr.batch_sharding)
    out = ROOT / "runs" / "hlo"
    out.mkdir(parents=True, exist_ok=True)
    texts = {"grad": tr._grad_fn.lower(params, batch).compile().as_text()}
    loss, grads, gnorm = tr._grad_fn(params, batch)
    t = int(sys.argv[2]) if len(sys.argv) > 2 else {"adamw": 2, "soap": 2, "gimbal": 60}[name]
    kind = tr.spec.kind(t)
    if name == "gimbal":
        from gimbal.jax import distributed as dist

        state = jax.eval_shape(lambda: dist.drop_gimbal_warm_buffers(state))
        state = jax.tree.map(
            lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh),
            state,
            tr.state_shardings(state),
        )
    fn = tr._make_update_fn(kind, jax.eval_shape(lambda: state))
    texts["update"] = (
        fn.lower(state, grads, params, gnorm, jnp.float32(1e-3), jnp.int32(t)).compile().as_text()
    )
    if os.environ.get("GIMBAL_WORKER", "0") != "0":
        return
    for k, text in texts.items():
        (out / f"{name}_{k}.hlo.txt").write_text(text)
        ops = collections.Counter(
            re.findall(
                r"= \S+ (all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)"
                r"(?:-start)?\(",
                text,
            )
        )
        print(name, k, dict(ops), flush=True)
        pattern = re.compile(r"(all-reduce|all-gather|reduce-scatter)(-start)?\(")
        for line in text.splitlines():
            if pattern.search(line) and "=" in line:
                print("   ", line.strip()[:200])


if __name__ == "__main__":
    main()
