"""Token data for the TPU runs (Phase 05): memory-mapped uint16 files, seeded order, sharding.

A sequence ``i`` of a split is ``tokens[1024 i : 1024 i + 1025]`` (inputs and shifted targets), so
the 2.5B-token train split holds 2,441,406 sequences. A seed fixes a permutation of them, which
every optimizer shares (paired design); step ``s`` uses permutation entries
``[240 s, 240 (s + 1))``, so no sequence is seen twice in a run. Validation and test sequences are
taken in file order. Each host reads only the rows of its own devices.
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Iterator
from queue import Queue

import jax
import numpy as np
from jax.sharding import NamedSharding


class TokenFile:
    def __init__(self, path: str | pathlib.Path, seq_len: int = 1024) -> None:
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_seq = (len(self.tokens) - 1) // seq_len

    def rows(self, idx: np.ndarray) -> np.ndarray:
        out = np.empty((len(idx), self.seq_len + 1), dtype=np.int32)
        for r, i in enumerate(idx):
            start = int(i) * self.seq_len
            out[r] = self.tokens[start:start + self.seq_len + 1]
        return out


def train_order(n_seq: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_seq)


def global_batch(data: TokenFile, idx: np.ndarray, sharding: NamedSharding) -> jax.Array:
    """Batch ``[len(idx), T + 1]`` sharded along the batch axis; each host reads its own rows."""
    shape = (len(idx), data.seq_len + 1)

    def callback(index):
        rows = idx[index[0]]
        return data.rows(rows)[:, index[1]]

    return jax.make_array_from_callback(shape, sharding, callback)


def train_batches(data: TokenFile, order: np.ndarray, batch: int, start_step: int,
                  steps: int, sharding: NamedSharding, prefetch: int = 2) -> Iterator[jax.Array]:
    """Batches for steps ``start_step .. steps - 1``, prepared by a background thread."""
    if steps * batch > len(order):
        raise ValueError(f"{steps} steps of {batch} need {steps * batch} sequences, "
                         f"have {len(order)}")
    queue: Queue = Queue(maxsize=prefetch)

    def worker():
        for s in range(start_step, steps):
            queue.put(global_batch(data, order[s * batch:(s + 1) * batch], sharding))
        queue.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (item := queue.get()) is not None:
        yield item


def eval_batches(data: TokenFile, n_seq: int, batch: int,
                 sharding: NamedSharding) -> Iterator[tuple[jax.Array, int]]:
    """The first ``n_seq`` sequences in order; the last batch is padded (count of real rows)."""
    for start in range(0, n_seq, batch):
        real = min(batch, n_seq - start)
        idx = np.arange(start, start + batch)
        idx[real:] = start  # padding rows repeat a real row and are dropped by the caller
        yield global_batch(data, idx, sharding), real
