"""Per-operation device time of one optimizer update kind, traced alone (Phase 04).

scripts/tpu/launch.sh <logdir> "cd benchmarks/tpu && python profile_update.py gimbal"
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import sys

import jax
import jax.numpy as jnp
import yaml
from xplane import device_op_times

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    name = sys.argv[1]
    reps = 10
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from gimbal.train.data import TokenFile, global_batch, train_order
    from gimbal.train.train import Trainer

    cfg = yaml.safe_load((ROOT / "configs/base_125m.yaml").read_text())
    data = TokenFile(ROOT / cfg["data"]["train"])
    tr = Trainer(cfg, name)
    params, state = tr.init(0)
    batch = global_batch(data, train_order(data.n_seq, 0)[:240], tr.batch_sharding)
    loss, grads, gnorm = tr._grad_fn(params, batch)
    warm = {"adamw": 1, "soap": 12, "gimbal": 56}[name]
    for t in range(1, warm + 1):
        state, params = tr.update(state, jax.tree.map(jnp.copy, grads), params, gnorm, 1e-4, t)
    kinds = {}
    for t in range(warm + 1, warm + 21):
        kinds.setdefault(tr.spec.kind(t), t)
    worker = os.environ.get("GIMBAL_WORKER", "0")
    results = {}
    for kind, t in kinds.items():
        jax.block_until_ready(state)
        trace = ROOT / "runs" / "profiles" / f"update_{name}_{t}_w{worker}"
        shutil.rmtree(trace, ignore_errors=True)
        copies = [
            (
                jax.tree.map(jnp.copy, state),
                jax.tree.map(jnp.copy, grads),
                jax.tree.map(jnp.copy, params),
            )
            for _ in range(reps)
        ]
        jax.block_until_ready(copies)
        jax.profiler.start_trace(str(trace))
        for s_in, g_in, p_in in copies:
            out = tr.update(s_in, g_in, p_in, gnorm, 1e-4, t)
        jax.block_until_ready(out)
        jax.profiler.stop_trace()
        if worker == "0":
            path = sorted(glob.glob(str(trace / "**" / "*.xplane.pb"), recursive=True))[-1]
            ops = device_op_times(path)
            total = sum(ops.values())
            results[str(kind)] = {
                "ms_per_call": total / 1e6 / reps,
                "ops_ms": {k: round(v / 1e6 / reps, 3) for k, v in ops.most_common(25)},
            }
    if worker == "0":
        dest = ROOT / "benchmarks/tpu/results" / f"profile_update_{name}.json"
        dest.write_text(json.dumps(results, indent=1))
        print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
