"""Component timings on the v4-32 slice (Phase 04): attention kernels, gradient step, updates.

Every timing is the median of repeated calls after warm-up, with the device synchronized around
each repetition. Run on all hosts: scripts/tpu/launch.sh <logdir> python
benchmarks/tpu/bench_components.py --out benchmarks/tpu/results/components.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def timeit(fn, *args, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/tpu/results/components.json")
    parser.add_argument("--attention", default="xla,flash,splash")
    args = parser.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from gimbal.train.data import TokenFile, global_batch, train_order
    from gimbal.train.train import Trainer

    cfg = yaml.safe_load((ROOT / "configs/base_125m.yaml").read_text())
    results = {"devices": jax.device_count(), "batch": cfg["train"]["batch"]}
    data = TokenFile(ROOT / cfg["data"]["train"])
    order = train_order(data.n_seq, 0)
    for attn in args.attention.split(","):
        cfg["model"]["attention"] = attn
        tr = Trainer(cfg, "adamw")
        params, state = tr.init(0)
        batch = global_batch(data, order[:cfg["train"]["batch"]], tr.batch_sharding)
        try:
            t = timeit(tr._grad_fn, params, batch)
            results[f"grad_step_{attn}_s"] = t
        except Exception as exc:  # record kernels that do not compile on this chip
            results[f"grad_step_{attn}_s"] = f"failed: {type(exc).__name__}: {exc}"[:500]
    for name in ("adamw", "soap", "gimbal"):
        cfg["model"]["attention"] = "xla"
        tr = Trainer(cfg, name)
        params, state = tr.init(0)
        batch = global_batch(data, order[:cfg["train"]["batch"]], tr.batch_sharding)
        loss, grads, gnorm = tr._grad_fn(params, batch)
        # Bring the state to its steady-state layout and contents first.
        warm = {"adamw": 1, "soap": 12, "gimbal": 56}[name]
        for t in range(1, warm + 1):
            state, params = tr.update(state, jax.tree.map(jnp.copy, grads), params, gnorm,
                                      1e-4, t)
        kinds = {}
        for t in range(warm + 1, warm + 21):
            kinds.setdefault(str(tr.spec.kind(t)), t)
        for kind, t in kinds.items():
            def run(state, params, t=t, tr=tr, grads=grads, gnorm=gnorm):
                return tr.update(state, jax.tree.map(jnp.copy, grads), params, gnorm, 1e-4, t)
            # donation consumes the inputs, so time with fresh copies each repetition
            times = []
            for i in range(8):
                s_in = jax.tree.map(jnp.copy, state)
                p_in = jax.tree.map(jnp.copy, params)
                jax.block_until_ready((s_in, p_in))
                t0 = time.perf_counter()
                out = run(s_in, p_in)
                jax.block_until_ready(out)
                if i >= 2:
                    times.append(time.perf_counter() - t0)
            results[f"update_{name}_{kind}_s"] = float(np.median(times))
    if os.environ.get("GIMBAL_WORKER", "0") == "0":
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
