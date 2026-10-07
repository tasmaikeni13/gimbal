"""Record the TPU environment (Phase 04: verify hardware facts, do not assume them).

Run on every host at once (``scripts/tpu/launch.sh python scripts/tpu/env_check.py``). Process 0
writes ``runs/env.json`` with the device topology, software versions and XLA flags.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import subprocess

import jax
import jax.numpy as jnp


def main() -> None:
    jax.distributed.initialize()
    devices = jax.devices()
    local = jax.local_devices()
    # One all-reduce across every chip proves the slice is wired up (ICI between hosts).
    x = jax.device_put(
        jnp.ones((len(devices),)),
        jax.sharding.NamedSharding(
            jax.sharding.Mesh(devices, ("d",)), jax.sharding.PartitionSpec("d")
        ),
    )
    total = float(jax.jit(jnp.sum)(x))
    info = {
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "device_kind": devices[0].device_kind,
        "platform": devices[0].platform,
        "coords": [list(getattr(d, "coords", [])) for d in local],
        "allreduce_check": total,
        "hbm_bytes_per_device": local[0].memory_stats().get("bytes_limit"),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__ if hasattr(jax.lib, "__version__") else None,
        "python": platform.python_version(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    try:
        from importlib.metadata import version

        info["libtpu"] = version("libtpu")
    except Exception as exc:  # pragma: no cover - environment probe
        info["libtpu"] = f"unavailable: {exc}"
    try:
        info["tpu_env"] = pathlib.Path("/tmp/tpu-env").read_text().splitlines()[:5]
    except OSError:
        info["tpu_env"] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        info["git_commit"] = None
    print(json.dumps(info), flush=True)
    # JAX numbers processes by chip coordinates, not by worker id, so the file is written by
    # worker 0 (where results are collected), whatever its process index.
    info["worker"] = int(os.environ.get("GIMBAL_WORKER", "-1"))
    if info["worker"] == 0:
        out = pathlib.Path(__file__).resolve().parents[2] / "runs" / "env.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
