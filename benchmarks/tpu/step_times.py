"""Steady-state step time (C-017): median over the 20-step blocks of steps 200-400 of the mean
step time in each block, from the per-step logs of short runs at the confirmatory configuration.

Usage: python benchmarks/tpu/step_times.py runs/timing/<run> [...]  -> benchmarks/tpu/results/
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def steady_state(log: pathlib.Path, lo: int = 200, hi: int = 400, block: int = 20) -> dict:
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    t = {r["step"]: r["step_time"] for r in rows}
    blocks = [np.mean([t[s] for s in range(b, b + block)]) for b in range(lo, hi, block)
              if all(s in t for s in range(b, b + block))]
    return {"steady_state_s": float(np.median(blocks)), "blocks": len(blocks),
            "block_means_s": [float(b) for b in blocks],
            "per_step_median_s": float(np.median([t[s] for s in range(lo, hi) if s in t]))}


def main() -> None:
    out = {}
    for run in sys.argv[1:]:
        path = ROOT / run
        cfg = json.loads((path / "config.json").read_text())
        out[path.name] = {"optimizer": cfg["optimizer"], "matrix_config": cfg["matrix_config"],
                          **steady_state(path / "log.jsonl")}
        print(path.name, round(out[path.name]["steady_state_s"] * 1e3, 2), "ms", flush=True)
    dest = ROOT / "benchmarks/tpu/results/step_times.json"
    old = json.loads(dest.read_text()) if dest.exists() else {}
    dest.write_text(json.dumps({**old, **out}, indent=1))


if __name__ == "__main__":
    main()
