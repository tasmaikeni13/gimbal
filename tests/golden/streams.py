"""Fixed gradient sequences for the cross-framework golden tests (Phase 03).

Sequences are generated from fixed seeds with NumPy's PCG64 generator, whose output is stable across
NumPy versions, so no binary fixtures are stored. Each gradient is a KRD draw (independent entries
in a random Kronecker frame) plus a fixed mean, scaled to typical language-model gradient sizes.

* ``identified``: row and column variance profiles spread geometrically, so every pair of rows and
  columns has Fisher information well above the damping (Theorem 2): the likelihood frame is
  well-defined and trajectories are not dominated by weakly identified pairs.
* ``random``: log-normal variances; some pairs are nearly degenerate.
"""

from __future__ import annotations

import numpy as np


def stream(
    shape: tuple[int, int],
    steps: int,
    seed: int,
    kind: str = "identified",
    scale: float = 1e-3,
    mean: float = 0.3,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    m, n = shape
    ql = np.linalg.qr(rng.standard_normal((m, m)))[0]
    qr = np.linalg.qr(rng.standard_normal((n, n)))[0]
    if kind == "identified":
        d = np.outer(1.6 ** np.arange(m), 1.5 ** np.arange(n))
        d = d * np.exp(0.3 * rng.standard_normal(shape))
    elif kind == "random":
        d = np.exp(rng.standard_normal(shape))
    else:
        raise ValueError(kind)
    d = d / d.mean()
    mu = mean * rng.standard_normal(shape)
    return [
        (scale * (mu + ql @ (np.sqrt(d) * rng.standard_normal(shape)) @ qr.T)) for _ in range(steps)
    ]


def initial_params(shape: tuple[int, int], seed: int) -> np.ndarray:
    return 0.02 * np.random.default_rng(seed + 1000).standard_normal(shape)
