"""E3.2 (formerly E2.7; also E3.1 snapshots): byte-level LM on a FineWeb-Edu sample (Phase 03).

Every optimizer trains the same model from the same initialization on the same sequence of
batches for a given seed (paired design). Hidden matrices use the optimizer under test; the
embedding, output head and norms use one AdamW configuration shared by all methods.

Usage:
  python experiments/phase3/e27_small_lm.py --method gimbal --lr 3e-3 --seed 0 --steps 800
  python experiments/phase3/e27_small_lm.py --method adamw --lr 3e-3 --seed 0 \
      --snapshots 100,400,800
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
from lm.model import Config, TinyLM

from gimbal.torch import build

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
RESULTS = pathlib.Path(__file__).parent / "results" / "lm"

MATRIX_KW = {
    "adamw": dict(betas=(0.9, 0.95), weight_decay=0.0),
    "soap": dict(betas=(0.9, 0.95), weight_decay=0.0),
    "soap_rt": dict(betas=(0.9, 0.95), weight_decay=0.0),
    "klsoap": dict(betas=(0.9, 0.95)),
    "muon": dict(),
    "normuon": dict(),
    "splus": dict(weight_decay=0.0),
    "aro": dict(),
    "gimbal": dict(betas=(0.9, 0.95)),
    "gimbal_k4": dict(betas=(0.9, 0.95), frame_every=4),
}
# Variants that are a registered optimizer with different settings.
ALIASES = {"gimbal_k4": "gimbal"}
OTHER_KW = dict(lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)


def load_bytes(split: str) -> torch.Tensor:
    raw = np.frombuffer((DATA / f"fineweb_edu_{split}.txt").read_bytes(), dtype=np.uint8)
    return torch.from_numpy(raw.astype(np.int64))


def batches(data: torch.Tensor, batch: int, seq: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    while True:
        starts = torch.randint(0, len(data) - seq - 1, (batch,), generator=gen)
        idx = torch.stack([data[s : s + seq + 1] for s in starts.tolist()])
        yield idx[:, :-1], idx[:, 1:]


def lr_scale(t: int, total: int, warmup: int) -> float:
    if t < warmup:
        return (t + 1) / warmup
    prog = (t - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model: TinyLM, val: torch.Tensor, n_batches: int, batch: int) -> float:
    model.eval()
    it = batches(val, batch, model.cfg.seq_len, seed=12345)
    losses = [float(model(*next(it))) for _ in range(n_batches)]
    model.train()
    return float(np.mean(losses))


def gradient_snapshot(model, train, n_samples, batch, seed, names):
    """Independent minibatch gradients of selected matrices at fixed weights (E3.1)."""
    params = dict(model.named_parameters())
    it = batches(train, batch, model.cfg.seq_len, seed=seed)
    out = {n: [] for n in names}
    for _ in range(n_samples):
        model.zero_grad(set_to_none=True)
        model(*next(it)).backward()
        for n in names:
            out[n].append(params[n].grad.detach().double().numpy().copy())
    model.zero_grad(set_to_none=True)
    return {n: np.stack(v) for n, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--snapshots", default="")
    parser.add_argument("--snapshot-samples", type=int, default=96)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg = Config(d_model=args.d_model, n_layers=args.layers,
                 mlp_hidden=int(round(8 * args.d_model / 3 / 32)) * 32)
    model = TinyLM(cfg)
    train, val = load_bytes("train"), load_bytes("val")
    opt = build(ALIASES.get(args.method, args.method), model.hidden_matrices(),
                model.other_parameters(),
                matrix_kwargs=dict(lr=args.lr, **MATRIX_KW[args.method]), other_kwargs=OTHER_KW)
    snapshots = {int(s) for s in args.snapshots.split(",") if s}
    snap_names = [f"blocks.{i}.{w}.weight" for i in (0, cfg.n_layers - 1)
                  for w in ("wq", "wo", "w_up", "w_down")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.method}_lr{args.lr:g}_s{args.seed}{args.tag}"
    log_path = RESULTS / f"{run_id}.jsonl"
    log = log_path.open("w")
    it = batches(train, args.batch, cfg.seq_len, seed=args.seed)
    warmup = max(1, args.steps // 20)
    opt_time = 0.0
    t_start = time.perf_counter()
    for step in range(args.steps):
        opt.set_lr_scale(lr_scale(step, args.steps, warmup))
        x, y = next(it)
        loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        t0 = time.perf_counter()
        opt.step()
        opt_time += time.perf_counter() - t0
        rec = {"step": step, "loss": float(loss)}
        if not math.isfinite(rec["loss"]):
            rec["diverged"] = True
            log.write(json.dumps(rec) + "\n")
            break
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            rec["val_loss"] = evaluate(model, val, args.eval_batches, args.batch)
            rec["wall"] = time.perf_counter() - t_start
            rec["opt_time"] = opt_time
        if step + 1 in snapshots:
            snap = gradient_snapshot(model, train, args.snapshot_samples, args.batch,
                                     seed=777 + step, names=snap_names)
            np.savez_compressed(RESULTS / f"snap_{run_id}_step{step + 1}.npz", **snap)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    summary = {"run_id": run_id, "method": args.method, "lr": args.lr, "seed": args.seed,
               "steps": args.steps, "final_val_loss": rec.get("val_loss", float("nan")),
               "wall_seconds": time.perf_counter() - t_start, "optimizer_seconds": opt_time,
               "diverged": bool(rec.get("diverged", False))}
    (RESULTS / f"{run_id}.summary.json").write_text(json.dumps(summary))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
