"""Per-host checkpoints on the hosts' local disks (Phase 05).

Every host writes the shards it holds (replicated arrays once) to
``<dir>/step_<n>/process_<k>/shards.npz`` and restores them with
``jax.make_array_from_callback``, so no array is ever gathered to one host. Only the latest
checkpoint of a run is kept. JAX numbers processes by chip coordinates, which are fixed for the
slice, so a host finds its own files again on restart.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import jax
import numpy as np
from jax.experimental import multihost_utils


def _key(index: tuple, shape: tuple) -> str:
    return ",".join(f"{s.start or 0}:{s.stop if s.stop is not None else n}"
                    for s, n in zip(index, shape, strict=True))


def latest(ckpt_dir: pathlib.Path) -> int | None:
    steps = sorted(int(p.name.split("_")[1]) for p in ckpt_dir.glob("step_*")
                   if (p / f"process_{jax.process_index()}" / "done").exists())
    return steps[-1] if steps else None


def save(ckpt_dir: pathlib.Path, step: int, params: dict, state: dict,
         params_only: bool = False) -> None:
    tree = {"params": params} if params_only else {"params": params, "state": state}
    leaves, _ = jax.tree.flatten(tree)
    out = ckpt_dir / f"step_{step:06d}" / f"process_{jax.process_index()}"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, x in enumerate(leaves):
        for sh in x.addressable_shards:
            name = f"{i}|{_key(sh.index, x.shape)}"
            if name not in arrays:
                arrays[name] = np.asarray(sh.data)
    np.savez(out / "shards.npz", **arrays)
    (out / "meta.json").write_text(json.dumps({"step": step, "leaves": len(leaves),
                                               "params_only": params_only}))
    (out / "done").touch()
    multihost_utils.sync_global_devices(f"ckpt_{step}")
    for old in ckpt_dir.glob("step_*"):
        if old.name != out.parent.name:
            shutil.rmtree(old / f"process_{jax.process_index()}", ignore_errors=True)
            if not any(old.iterdir()):
                old.rmdir()


def restore(ckpt_dir: pathlib.Path, step: int, trainer, seed: int) -> tuple[dict, dict]:
    """Parameters and optimizer state saved at ``step`` (layouts from ``trainer``)."""
    del seed  # the data order is recomputed from the seed by the caller
    params_abs = trainer.init_params_abstract()
    state_abs = trainer.abstract_state(step)
    tree_abs = {"params": params_abs, "state": state_abs}
    shard = {"params": jax.tree.map(lambda _: trainer.replicated, params_abs),
             "state": trainer.state_shardings(state_abs)}
    leaves, treedef = jax.tree.flatten(tree_abs)
    shard_leaves = jax.tree.leaves(shard)
    data = np.load(ckpt_dir / f"step_{step:06d}" / f"process_{jax.process_index()}"
                   / "shards.npz")
    arrays = []
    for i, (x, sh) in enumerate(zip(leaves, shard_leaves, strict=True)):
        arrays.append(jax.make_array_from_callback(
            x.shape, sh, lambda index, i=i, x=x: data[f"{i}|{_key(index, x.shape)}"]))
    tree = jax.tree.unflatten(treedef, arrays)
    return tree["params"], tree["state"]
