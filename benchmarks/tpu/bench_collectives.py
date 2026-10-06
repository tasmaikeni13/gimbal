"""Collective variants for the gradient reduction and parameter gather (Phase 04).

Times psum (all-reduce), psum_scatter (tiled and untiled) and all_gather in float32 and bfloat16
on a [48, 768, 2048] array over the 16-chip mesh, and records which HLO collective each became.
"""

from __future__ import annotations

import json
import os
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    jax.distributed.initialize()
    from bench_components import timeit

    mesh = Mesh(np.array(jax.devices()), ("data",))
    n = jax.device_count()
    out = {}
    for dtype in (jnp.float32, jnp.bfloat16):
        full = jax.device_put(jnp.ones((n, 48, 768, 2048), dtype),
                              NamedSharding(mesh, P("data")))  # one local copy per device
        local = jax.device_put(jnp.ones((48 // n * n, 768, 2048), dtype),
                               NamedSharding(mesh, P("data")))
        variants = {
            "psum": (lambda x: jax.lax.psum(x[0], "data"), P("data"), P()),
            "psum_scatter_tiled": (
                lambda x: jax.lax.psum_scatter(x[0], "data", scatter_dimension=0, tiled=True),
                P("data"), P("data")),
            "psum_scatter_untiled": (
                lambda x: jax.lax.psum_scatter(x[0].reshape(n, -1, 768, 2048), "data",
                                               scatter_dimension=0, tiled=False),
                P("data"), P("data")),
            "all_gather": (lambda x: jax.lax.all_gather(x, "data", axis=0, tiled=True),
                           P("data"), P()),
        }
        for name, (fn, ispec, ospec) in variants.items():
            f = jax.jit(shard_map(fn, mesh=mesh, in_specs=ispec, out_specs=ospec,
                                  check_rep=False))
            arg = local if name == "all_gather" else full
            hlo = f.lower(arg).compile().as_text()
            kinds = sorted(set(re.findall(r"(all-reduce|reduce-scatter|all-gather)(?:-start)?\(",
                                          hlo)))
            key = f"{name}_{jnp.dtype(dtype).name}"
            out[key] = {"seconds": timeit(f, arg), "hlo": kinds}
            print(key, out[key], flush=True)
    if os.environ.get("GIMBAL_WORKER", "0") == "0":
        (ROOT / "benchmarks/tpu/results/collectives.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
