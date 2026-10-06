"""Phase 06 tuning driver (protocol after D-003): equal-budget sweep of AdamW, SOAP and Gimbal.

Every run: 125M model, 2,500 steps of 240 x 1024 tokens, seeds 0 and 1, selection on the mean over
the two seeds of the final loss on the full validation split. Stage A: 7 learning rates (factor 2)
around each reference implementation's default; if any optimizer's best rate is on an edge, every
optimizer gets two more rates (on its edge side, or on the side of its better neighbour), so the
budgets stay equal. Stage C: the secondary knob of each optimizer (3 values; the default is the
Stage A run at the best rate). All runs use one frozen code snapshot; runs are launched one at a
time on all 16 chips, interleaved across optimizers, and skipped if already finished, so the
driver can be restarted. Every trial is recorded in ``runs/tuning/manifest.csv``.

Usage (on worker 0): nohup python scripts/tpu/tune.py > runs/logs/tune.log 2>&1 &
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "tuning"
STEPS = 2500
SEEDS = (0, 1)
CENTRE = {"adamw": 1e-3, "soap": 3e-3, "gimbal": 3e-3}  # reference implementations' defaults
SECONDARY = {  # (config key, values; the middle one is the default)
    "adamw": ("b2", (0.95, 0.98, 0.999)),
    "soap": ("shampoo_beta", (0.9, 0.95, 0.99)),
    "gimbal": ("rot_rate", (0.025, 0.05, 0.1)),
}
DEFAULT_SECONDARY = {"adamw": 0.95, "soap": 0.95, "gimbal": 0.05}


def run_name(opt: str, lr: float, seed: int, knob: str | None = None) -> str:
    return f"{opt}_lr{lr:.4g}" + (f"_{knob}" if knob else "") + f"_s{seed}"


def result(name: str) -> dict | None:
    path = RUNS / name / "final.json"
    return json.loads(path.read_text()) if path.exists() else None


def launch(snapshot: str, opt: str, lr: float, seed: int, sets: list[str], name: str) -> dict:
    done = result(name)
    if done is not None:
        return done
    cmd = (f"python -m gimbal.train.train --optimizer {opt} --lr {lr} --seed {seed} "
           f"--steps {STEPS} --run-dir runs/tuning/{name} "
           + " ".join(f"--set {s}" for s in sets))
    for attempt in range(2):  # one retry for infrastructure failures (not for divergence)
        env = dict(os.environ, GIMBAL_CODE=snapshot)
        t0 = time.time()
        proc = subprocess.run([str(ROOT / "scripts/tpu/launch.sh"),
                               str(ROOT / "runs/logs/tuning" / name), cmd], env=env)
        done = result(name)
        print(f"{time.strftime('%H:%M:%S')} {name} exit={proc.returncode} "
              f"attempt={attempt} {time.time() - t0:.0f}s {done}", flush=True)
        if done is not None:
            return done
    return {"diverged": None, "failed": True}


def score(opt: str, lr: float, knob: str | None = None) -> float:
    """Mean final validation loss over the seeds; inf if any seed diverged or failed."""
    vals = []
    for s in SEEDS:
        r = result(run_name(opt, lr, s, knob))
        if r is None or r.get("diverged") or "val_loss" not in r:
            return math.inf
        vals.append(r["val_loss"])
    return sum(vals) / len(vals)


def best_lr(opt: str, lrs: list[float]) -> float:
    """Lowest mean loss; ties within 0.002 nats go to the smaller learning rate."""
    scores = {lr: score(opt, lr) for lr in lrs}
    top = min(scores.values())
    return min(lr for lr, v in scores.items() if v <= top + 0.002)


def write_manifest(grids: dict) -> None:
    rows = []
    for opt in CENTRE:
        for lr in sorted(grids[opt]):
            for s in SEEDS:
                rows.append(("A", opt, lr, "", s, run_name(opt, lr, s)))
        key, values = SECONDARY[opt]
        for v in values:
            if v == DEFAULT_SECONDARY[opt]:
                continue
            for lr in grids.get(f"{opt}_best", []):
                for s in SEEDS:
                    rows.append(("C", opt, lr, f"{key}={v}", s, run_name(opt, lr, s, f"{key}{v}")))
    with (RUNS / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "optimizer", "lr", "secondary", "seed", "run", "status",
                    "val_loss", "train_seconds"])
        for stage, opt, lr, sec, s, name in rows:
            r = result(name)
            status = ("pending" if r is None else "diverged" if r.get("diverged")
                      else "failed" if r.get("failed") else "done")
            w.writerow([stage, opt, f"{lr:.4g}", sec, s, name, status,
                        "" if r is None else r.get("val_loss", ""),
                        "" if r is None else round(r.get("train_seconds", 0))])


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    snap_file = RUNS / "SNAPSHOT"
    if snap_file.exists():
        snapshot = snap_file.read_text().strip()
    else:
        snapshot = subprocess.check_output(
            [str(ROOT / "scripts/tpu/snapshot.sh"), "tuning"], text=True).strip().splitlines()[-1]
        snap_file.write_text(snapshot + "\n")
    print("snapshot", snapshot, flush=True)
    grids = {opt: [c * 2.0**k for k in range(-3, 4)] for opt, c in CENTRE.items()}
    # Stage A, interleaved: (rate index, seed) outer, optimizer inner.
    for k in range(7):
        for s in SEEDS:
            for opt in CENTRE:
                launch(snapshot, opt, grids[opt][k], s, [], run_name(opt, grids[opt][k], s))
                write_manifest(grids)
    # Edge extension, offered to every optimizer alike (at most twice).
    for _ in range(2):
        best = {opt: best_lr(opt, grids[opt]) for opt in CENTRE}
        edge = {opt: best[opt] in (min(grids[opt]), max(grids[opt])) for opt in CENTRE}
        if not any(edge.values()):
            break
        for opt in CENTRE:
            lrs = sorted(grids[opt])
            if best[opt] == lrs[-1] or (not edge[opt] and score(opt, lrs[lrs.index(best[opt]) + 1])
                                        < score(opt, lrs[lrs.index(best[opt]) - 1])):
                new = [lrs[-1] * 2, lrs[-1] * 4]
            else:
                new = [lrs[0] / 2, lrs[0] / 4]
            grids[opt] += new
        for lr_idx in range(2):
            for s in SEEDS:
                for opt in CENTRE:
                    lr = grids[opt][-2 + lr_idx]
                    launch(snapshot, opt, lr, s, [], run_name(opt, lr, s))
                    write_manifest(grids)
    best = {opt: best_lr(opt, grids[opt]) for opt in CENTRE}
    for opt in CENTRE:
        grids[f"{opt}_best"] = [best[opt]]
    print("stage A selected", best, flush=True)
    # Stage C at the selected rate.
    for s in SEEDS:
        for idx in (0, 2):
            for opt in CENTRE:
                key, values = SECONDARY[opt]
                v = values[idx]
                launch(snapshot, opt, best[opt], s, [f"optimizers.{opt}.{key}={v}"],
                       run_name(opt, best[opt], s, f"{key}{v}"))
                write_manifest(grids)
    final = {}
    for opt in CENTRE:
        key, values = SECONDARY[opt]
        scores = {v: (score(opt, best[opt]) if v == DEFAULT_SECONDARY[opt]
                      else score(opt, best[opt], f"{key}{v}")) for v in values}
        top = min(scores.values())
        # ties within 0.002 nats keep the default
        chosen = (DEFAULT_SECONDARY[opt] if scores[DEFAULT_SECONDARY[opt]] <= top + 0.002
                  else min(scores, key=scores.get))
        final[opt] = {"lr": best[opt], key: chosen, "stage_c_scores": scores,
                      "stage_a_scores": {f"{lr:.4g}": score(opt, lr) for lr in sorted(grids[opt])}}
    (RUNS / "selection.json").write_text(json.dumps(final, indent=1))
    print("selection", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
