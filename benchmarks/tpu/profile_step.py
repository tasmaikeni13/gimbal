"""Device-time breakdown of one training step by XLA operation (Phase 04 profiling).

Captures a trace of a few gradient + update steps with ``jax.profiler`` and aggregates the TPU
events of worker 0's first chip by name. Run on all hosts:
scripts/tpu/launch.sh <logdir> "cd benchmarks/tpu && python profile_step.py --optimizer adamw"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib

import jax
import yaml
from xplane import device_op_times

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from gimbal.train.data import TokenFile, global_batch, train_order
    from gimbal.train.train import Trainer

    cfg = yaml.safe_load((ROOT / "configs/base_125m.yaml").read_text())
    data = TokenFile(ROOT / cfg["data"]["train"])
    order = train_order(data.n_seq, 0)
    tr = Trainer(cfg, args.optimizer)
    params, state = tr.init(0)
    batch = global_batch(data, order[: cfg["train"]["batch"]], tr.batch_sharding)
    warm = {"adamw": 3, "soap": 12, "gimbal": 56}[args.optimizer]
    for t in range(1, warm + 1):
        loss, grads, gnorm = tr._grad_fn(params, batch)
        state, params = tr.update(state, grads, params, gnorm, 1e-4, t)
    jax.block_until_ready(params)
    worker = os.environ.get("GIMBAL_WORKER", "0")
    trace_dir = ROOT / "runs" / "profiles" / f"{args.optimizer}_w{worker}"
    jax.profiler.start_trace(str(trace_dir))
    for t in range(warm + 1, warm + 1 + args.steps):
        loss, grads, gnorm = tr._grad_fn(params, batch)
        state, params = tr.update(state, grads, params, gnorm, 1e-4, t)
    jax.block_until_ready(params)
    jax.profiler.stop_trace()
    if worker != "0":
        return
    path = sorted(glob.glob(str(trace_dir / "**" / "*.xplane.pb"), recursive=True))[-1]
    totals = device_op_times(path)
    modules = device_op_times(path, line_names=("XLA Modules",))
    total = sum(totals.values())
    rows = [
        {"op": k, "ms_per_step": v / 1e6 / args.steps, "share": v / total}
        for k, v in totals.most_common(args.top)
    ]
    out = {
        "optimizer": args.optimizer,
        "device_ms_per_step": total / 1e6 / args.steps,
        "modules_ms_per_step": {k: v / 1e6 / args.steps for k, v in modules.most_common()},
        "top": rows,
    }
    dest = ROOT / "benchmarks/tpu/results" / f"profile_{args.optimizer}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
