"""Driver for E2.7: equal-budget LR sweep then confirmation seeds for every optimizer.

Stage A: every optimizer gets the same number of learning rates (factor-2 grid around a
literature-based centre) on seed 0. Stage B: the selected LR is re-run on fresh seeds. Runs are
skipped if their summary already exists, so the sweep can be resumed.

Runs execute in parallel (``--jobs`` processes, ``--threads`` BLAS threads each; one thread per
process gives the best throughput for this model size on CPU).

Usage: python experiments/phase2/run_e27_sweep.py --stage A
       python experiments/phase2/run_e27_sweep.py --stage B --seeds 1 2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results" / "lm"

GRIDS = {
    "adamw": [1e-3, 2e-3, 4e-3, 8e-3],
    "soap": [1e-3, 2e-3, 4e-3, 8e-3],
    "soap_rt": [1e-3, 2e-3, 4e-3, 8e-3],
    "klsoap": [1e-3, 2e-3, 4e-3, 8e-3],
    "muon": [5e-3, 1e-2, 2e-2, 4e-2],
    "normuon": [2.5e-3, 5e-3, 1e-2, 2e-2],
    "splus": [0.1, 0.2, 0.4, 0.8],
    "aro": [1e-3, 2e-3, 4e-3, 8e-3],
    "gimbal": [1e-3, 2e-3, 4e-3, 8e-3],
    "gimbal_k4": [1e-3, 2e-3, 4e-3, 8e-3],
}


def summary_path(method: str, lr: float, seed: int, tag: str) -> pathlib.Path:
    return RESULTS / f"{method}_lr{lr:g}_s{seed}{tag}.summary.json"


def run(method: str, lr: float, seed: int, steps: int, threads: int, tag: str) -> dict:
    path = summary_path(method, lr, seed, tag)
    if path.exists():
        return json.loads(path.read_text())
    cmd = [sys.executable, str(HERE / "e27_small_lm.py"), "--method", method, "--lr", str(lr),
           "--seed", str(seed), "--steps", str(steps), "--threads", str(threads), "--tag", tag]
    env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads)}
    subprocess.run(cmd, check=True, cwd=HERE, env={**__import__("os").environ, **env},
                   stdout=subprocess.DEVNULL)
    return json.loads(path.read_text())


def best_lr(method: str, tag: str) -> float:
    rows = [json.loads(summary_path(method, lr, 0, tag).read_text()) for lr in GRIDS[method]]
    ok = [r for r in rows if not r["diverged"]]
    return min(ok, key=lambda r: r["final_val_loss"])["lr"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--methods", nargs="*", default=list(GRIDS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.stage == "A":
        todo = [(m, lr, 0) for m in args.methods for lr in GRIDS[m]]
    else:
        todo = [(m, best_lr(m, args.tag), seed) for m in args.methods for seed in args.seeds]

    def one(job):
        method, lr, seed = job
        r = run(method, lr, seed, args.steps, args.threads, args.tag)
        print(f"{args.stage} {method:9s} lr={lr:<8g} seed={seed} val={r['final_val_loss']:.4f} "
              f"opt_s={r['optimizer_seconds']:.0f}", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    main()
