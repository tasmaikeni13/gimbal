"""E2.1–E2.4: frame-estimation efficiency, ties, drift and heavy tails (Phase 02).

Every frame estimator is the real optimizer run with lr = 0 on the *same* gradient stream
(paired design). The score is the time-averaged frame KL J over the second half of the run
(Proposition 1), against the true (possibly drifting) frame.

Usage:
  python experiments/phase2/e21_frame_efficiency.py --suite main  --seeds 10
  python experiments/phase2/e21_frame_efficiency.py --suite tie   --seeds 10
  python experiments/phase2/e21_frame_efficiency.py --suite drift --seeds 10
  python experiments/phase2/e21_frame_efficiency.py --suite tails --seeds 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import pathlib
import time
import zlib

import numpy as np
import torch
from common import (  # noqa: E402  (run as a script)
    MEMORY_EXTENSION,
    MEMORY_GRID,
    KRDStream,
    frame_trace,
    make_variances,
    nonseparability_index,
    rand_orth,
)

RESULTS = pathlib.Path(__file__).parent / "results"

SUITES = {
    # (gamma, slope, shape, tie, nu, drift)
    "main": [
        dict(gamma=g, slope=s, shape=sh, tie=False, nu=None, drift=0.0)
        for g, s, sh in itertools.product(
            [0.0, 0.5, 1.0, 2.0], [0.5, 1.0, 1.5], [(32, 48), (64, 64)]
        )
    ],
    "tie": [
        dict(gamma=g, slope=1.0, shape=(32, 48), tie=True, nu=None, drift=0.0)
        for g in [0.0, 0.5, 1.0]
    ],
    "drift": [
        dict(gamma=1.0, slope=1.0, shape=(32, 48), tie=False, nu=None, drift=w)
        for w in [1e-3, 3e-3, 1e-2]
    ],
    "tails": [
        dict(gamma=1.0, slope=1.0, shape=(32, 48), tie=False, nu=nu, drift=0.0)
        for nu in [3.0, 5.0, None]
    ],
}


# Which grid extension each suite uses (``--extend``): drift favours short memories.
EXTENSION = {"main": "long", "tie": "long", "tails": "long", "drift": "short"}


def run_cell(args: tuple[dict, int, int, dict]) -> list[dict]:
    cfg, seed, steps, grid = args
    torch.set_num_threads(1)
    # Deterministic per-cell seed (Python's hash() is salted per process).
    cell_id = zlib.crc32(json.dumps(cfg, default=str, sort_keys=True).encode()) % 100_003
    rng = np.random.default_rng(1000 * seed + cell_id)
    m, n = cfg["shape"]
    d = make_variances(m, n, cfg["gamma"], cfg["slope"], rng, tie=cfg["tie"])
    ql, qr = rand_orth(m, rng), rand_orth(n, rng)
    stream = KRDStream(ql.copy(), qr.copy(), d, rng, nu=cfg["nu"], drift=cfg["drift"])
    grads, frames = [], []
    for _, g in zip(range(steps), stream, strict=False):
        grads.append(g)
        frames.append((stream.ql.copy(), stream.qr.copy()))
    # The frame valid for gradient t is the one before the stream advanced.
    true_frames = [(ql, qr)] + frames[:-1] if cfg["drift"] > 0 else None
    rows = []
    for method, mems in grid.items():
        for mem in mems:
            t0 = time.perf_counter()
            trace = frame_trace(method, mem, grads, ql, qr, d, true_frames=true_frames)
            rows.append(
                {
                    **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                    "seed": seed,
                    "method": method,
                    "memory": mem,
                    "kl_second_half": float(trace[steps // 2 :].mean()),
                    "kl_final": float(trace[-1]),
                    "kl_trace_every10": trace[::10].tolist(),
                    "kappa": nonseparability_index(d),
                    "seconds": time.perf_counter() - t0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="main")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=10,
        help="confirmatory runs use seeds 10.. (seeds 0-9 were exploratory)",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="run only the grid extension of this suite (same seeds and streams)",
    )
    parser.add_argument(
        "--methods", default="", help="comma-separated subset of methods (re-runs after a peer fix)"
    )
    parser.add_argument("--tag", default="", help="suffix of the output file")
    args = parser.parse_args()
    grid = MEMORY_EXTENSION[EXTENSION[args.suite]] if args.extend else MEMORY_GRID
    if args.methods:
        keep = set(args.methods.split(","))
        grid = {m: v for m, v in grid.items() if m in keep}
    jobs = [
        (cfg, args.seed_offset + seed, args.steps, grid)
        for cfg in SUITES[args.suite]
        for seed in range(args.seeds)
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"e21_{args.suite}{'_ext' if args.extend else ''}{args.tag}.jsonl"
    t0 = time.time()
    with mp.Pool(args.workers) as pool, out.open("w") as fh:
        for i, rows in enumerate(pool.imap_unordered(run_cell, jobs)):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} cells done, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
