"""Phase 07 driver: confirmatory runs of AdamW, SOAP and Gimbal at 125M / 2.5B tokens, 2 seeds.

Runs the frozen configurations (``configs/frozen/<optimizer>.yaml``, tag ``tuning-frozen``) on
seeds 2 and 3 (not used in Phase 06), interleaved (seed 2 of every optimizer, then seed 3), each on
all 16 chips from one frozen code snapshot, 10,172 steps of 240 x 1024 tokens. Checkpoints every
1,000 steps on the hosts' local disks. Procedure (Phase 07): on a crash, resume from the last
checkpoint with the identical configuration; on divergence, restart once from the last checkpoint
before it; a second divergence is a recorded failure. ``runs/main/manifest.csv`` has one row per
planned run.

Usage (on worker 0): nohup python scripts/tpu/main_runs.py > runs/logs/main_runs.log 2>&1 &
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "main"
OPTIMIZERS = ("adamw", "soap", "gimbal")
SEEDS = (2, 3)
STEPS = 10172


def final(name: str) -> dict | None:
    path = RUNS / name / "final.json"
    return json.loads(path.read_text()) if path.exists() else None


def frozen_args(opt: str) -> str:
    """Command-line overrides from the frozen configuration of one optimizer."""
    cfg = yaml.safe_load((ROOT / "configs" / "frozen" / f"{opt}.yaml").read_text())
    sets = [f"--set optimizers.{opt}.{k}={v}" for k, v in cfg["optimizer"].items() if k != "lr"]
    return f"--lr {cfg['optimizer']['lr']} " + " ".join(sets)


def run(snapshot: str, opt: str, seed: int) -> dict:
    name = f"{opt}/seed{seed}"
    done = final(name)
    if done is not None and not done.get("diverged"):
        return done
    base = (
        f"python -m gimbal.train.train --optimizer {opt} --seed {seed} --steps {STEPS} "
        f"--run-dir runs/main/{name} --set log.ckpt_every=1000 {frozen_args(opt)}"
    )
    env = dict(os.environ, GIMBAL_CODE=snapshot)
    divergences = 0
    resume_flag = ""
    for attempt in range(4):
        resume = resume_flag or ("--resume" if (RUNS / name / "log.jsonl").exists() else "")
        t0 = time.time()
        proc = subprocess.run(
            [
                str(ROOT / "scripts/tpu/launch.sh"),
                str(ROOT / "runs/logs/main" / f"{opt}_seed{seed}_a{attempt}"),
                f"{base} {resume}",
            ],
            env=env,
        )
        done = final(name)
        print(
            f"{time.strftime('%H:%M:%S')} {name} attempt={attempt} exit={proc.returncode} "
            f"{time.time() - t0:.0f}s {done}",
            flush=True,
        )
        if done is not None and not done.get("diverged"):
            return done
        if done is not None and done.get("diverged"):
            divergences += 1
            if divergences >= 2:
                return done  # twice-reproduced divergence: a valid, reported failure
            (RUNS / name / "final.json").rename(RUNS / name / f"final_diverged_{divergences}.json")
            # restart from a checkpoint taken before the divergence began (it is detected after
            # 50 steps above twice the 100-step minimum)
            last = json.loads((RUNS / name / "log.jsonl").read_text().splitlines()[-1])["step"]
            resume_flag = f"--resume-before {last - 50}"
        else:
            resume_flag = ""
    return done or {"failed": True}


def write_manifest() -> None:
    with (RUNS / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "optimizer",
                "seed",
                "run_dir",
                "status",
                "val_loss",
                "train_seconds",
                "config_commit",
                "final_checkpoint",
            ]
        )
        for seed in SEEDS:
            for opt in OPTIMIZERS:
                name = f"{opt}/seed{seed}"
                r = final(name)
                cfg = RUNS / name / "config.json"
                commit = json.loads(cfg.read_text()).get("git", "") if cfg.exists() else ""
                status = (
                    "planned"
                    if r is None
                    else "diverged"
                    if r.get("diverged")
                    else "done"
                    if "val_loss" in r
                    else "running"
                )
                # one shard file per host (process_<k>/shards.npz on worker k's local disk)
                ckpt = RUNS / name / "ckpt" / f"step_{STEPS:06d}"
                w.writerow(
                    [
                        opt,
                        seed,
                        f"runs/main/{name}",
                        status,
                        "" if r is None else r.get("val_loss", ""),
                        "" if r is None else round(r.get("train_seconds", 0)),
                        commit,
                        str(ckpt.relative_to(ROOT)) if ckpt.exists() else "",
                    ]
                )


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    snap_file = RUNS / "SNAPSHOT"
    if snap_file.exists():
        snapshot = snap_file.read_text().strip()
    else:
        snapshot = (
            subprocess.check_output([str(ROOT / "scripts/tpu/snapshot.sh"), "main"], text=True)
            .strip()
            .splitlines()[-1]
        )
        snap_file.write_text(snapshot + "\n")
    print("snapshot", snapshot, flush=True)
    # D-009: keep the step-1000 checkpoints of seed 2's SOAP and Gimbal runs for the mid-training
    # frame probe (the trainer keeps only the two newest periodic checkpoints).
    for opt in ("soap", "gimbal"):
        subprocess.Popen(
            [str(ROOT / "scripts/tpu/keep_checkpoint.sh"), f"runs/main/{opt}/seed2", "1000"],
            stdout=open(ROOT / "runs/logs" / f"keep_ckpt_{opt}.log", "a"),
            stderr=subprocess.STDOUT,
        )
    write_manifest()
    for seed in SEEDS:
        for opt in OPTIMIZERS:
            run(snapshot, opt, seed)
            write_manifest()
    print("all runs finished", flush=True)


if __name__ == "__main__":
    main()
