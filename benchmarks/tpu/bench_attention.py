"""Gradient-step time for attention implementations and flash-attention tiles (Phase 04).

Run on all hosts: scripts/tpu/launch.sh <logdir> python benchmarks/tpu/bench_attention.py
"""

from __future__ import annotations

import json
import os
import pathlib

import jax
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
VARIANTS = [("xla", 0, 0, 1), ("flash", 0, 0, 1), ("flash", 256, 256, 1), ("flash", 512, 512, 1),
            ("flash", 1024, 512, 1), ("flash", 1024, 1024, 1), ("flash", 512, 512, 3),
            ("flash", 1024, 1024, 3)]


def main() -> None:
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from bench_components import timeit

    from gimbal.train.data import TokenFile, global_batch, train_order
    from gimbal.train.train import Trainer

    cfg = yaml.safe_load((ROOT / "configs/base_125m.yaml").read_text())
    data = TokenFile(ROOT / cfg["data"]["train"])
    order = train_order(data.n_seq, 0)
    out = {}
    for kind, bq, bk, bb in VARIANTS:
        cfg["model"].update(attention=kind, flash_block_q=bq, flash_block_k=bk,
                            flash_block_b=bb)
        name = f"{kind}_q{bq}_k{bk}_b{bb}"
        try:
            tr = Trainer(cfg, "adamw")
            params, _ = tr.init(0)
            batch = global_batch(data, order[:cfg["train"]["batch"]], tr.batch_sharding)
            out[name] = timeit(tr._grad_fn, params, batch)
        except Exception as exc:
            out[name] = f"failed: {type(exc).__name__}: {exc}"[:300]
        print(name, out[name], flush=True)
    if os.environ.get("GIMBAL_WORKER", "0") == "0":
        path = ROOT / "benchmarks/tpu/results/attention.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
