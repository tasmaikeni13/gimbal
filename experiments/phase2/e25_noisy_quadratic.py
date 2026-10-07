"""E2.5: optimization on noisy quadratics with KRD curvature and noise (Phase 02).

Loss L(W) = 1/2 sum_ij H_ij X_ij^2 with X = Q_L*^T (W - W*) Q_R*, and gradient noise whose
covariance is proportional to H in the same Kronecker frame (Fisher ~ Hessian). The curvature
array H has log H = a_i + b_j + gamma c_ij (gamma = 0: separable).

Protocol (phases/02, G2.4): each method gets the same 7-point learning-rate grid (factor 2 around a
method-specific centre), tuned on seeds 0-2; the selected LR is evaluated on fresh seeds. All
methods see identical problems and noise per seed (paired). Schedule: 5% linear warm-up, cosine
decay to 10%. Evaluation seeds: 100-111 for the run that found F-020 (outputs ``*_full``), 300-311
for the confirmation of C-013 (``--tag _c013``, outputs ``*_full_c013``); seeds 200-203 were
pilots and are not reported.

Usage: python experiments/phase2/e25_noisy_quadratic.py [--quick] [--eval-offset 300] [--tag _c013]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import time

import numpy as np
import torch
from common import make_variances, rand_orth

from gimbal.torch import KLSOAP, SOAP, AROSinkhorn, Gimbal, Muon, NorMuon, SPlus

RESULTS = pathlib.Path(__file__).parent / "results"

# name -> (constructor, LR grid centre)
METHODS = {
    "adamw": (
        lambda p, lr: torch.optim.AdamW([p], lr=lr, betas=(0.9, 0.95), weight_decay=0.0),
        3e-2,
    ),
    "soap": (lambda p, lr: SOAP([p], lr=lr, betas=(0.9, 0.95), weight_decay=0.0), 3e-2),
    "soap_rt": (
        lambda p, lr: SOAP([p], lr=lr, betas=(0.9, 0.95), weight_decay=0.0, realtime=True),
        3e-2,
    ),
    "klsoap": (lambda p, lr: KLSOAP([p], lr=lr, betas=(0.9, 0.95)), 3e-2),
    "muon": (lambda p, lr: Muon([p], lr=lr), 0.1),
    "normuon": (lambda p, lr: NorMuon([p], lr=lr), 3e-2),
    "splus": (lambda p, lr: SPlus([p], lr=lr, weight_decay=0.0), 1.0),
    "aro": (lambda p, lr: AROSinkhorn([p], lr=lr), 3e-2),
    # Ours: the default ("gimbal_k4", frame_every = 4 since C-012) and frame_every = 1.
    "gimbal": (lambda p, lr: Gimbal([p], lr=lr, betas=(0.9, 0.95), frame_every=1), 3e-2),
    "gimbal_k4": (lambda p, lr: Gimbal([p], lr=lr, betas=(0.9, 0.95), frame_every=4), 3e-2),
    # Ablations of ours (not peers): no variance shrinkage; frame statistics on the raw gradient
    # (the default before C-013) and on the fully centered innovation (c = 1).
    "gimbal_noshrink": (
        lambda p, lr: Gimbal([p], lr=lr, betas=(0.9, 0.95), flow_shrink=False, frame_every=1),
        3e-2,
    ),
    "gimbal_k4_nocenter": (
        lambda p, lr: Gimbal([p], lr=lr, betas=(0.9, 0.95), frame_every=4, flow_center=False),
        3e-2,
    ),
    "gimbal_k4_center": (
        lambda p, lr: Gimbal([p], lr=lr, betas=(0.9, 0.95), frame_every=4, flow_center=True),
        3e-2,
    ),
}
ORACLE = "oracle"  # Adam in the true frame: the ceiling for "Adam in a Kronecker frame" methods

CONFIGS = [
    dict(gamma=g, noise=s, shape=(32, 48), slope=1.0)
    for g, s in itertools.product([0.0, 1.0, 2.0], [0.1, 1.0])
]


def lr_grid(centre: float) -> list[float]:
    return [centre * 2.0**k for k in range(-3, 4)]


def schedule(t: int, total: int) -> float:
    warm = max(1, total // 20)
    if t < warm:
        return (t + 1) / warm
    prog = (t - warm) / max(1, total - warm)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))


def run(method: str, lr: float, cfg: dict, seed: int, steps: int) -> dict:
    torch.set_num_threads(1)
    rng = np.random.default_rng(10_000 + seed * 31 + int(cfg["gamma"] * 7) + int(cfg["noise"] * 13))
    m, n = cfg["shape"]
    h = make_variances(m, n, cfg["gamma"], cfg["slope"], rng)
    h = h / h.max()
    ql, qr = rand_orth(m, rng), rand_orth(n, rng)
    x0 = rng.standard_normal((m, n))
    noise_rng = np.random.default_rng(99_000 + seed)
    noises = noise_rng.standard_normal((steps, m, n))
    H, QL, QR = (torch.from_numpy(a) for a in (h, ql, qr))
    sqrt_h = H.sqrt()
    w = torch.nn.Parameter(QL @ torch.from_numpy(x0) @ QR.T)  # W* = 0
    if method == ORACLE:
        opt = SOAP([w], lr=lr, betas=(0.9, 0.95), weight_decay=0.0, precondition_frequency=10**9)
        # Freeze SOAP's frame at the truth: initialise state with the true frame.
        st = opt.state[w]
        st.update(
            step=0,
            exp_avg=torch.zeros_like(w),
            exp_avg_sq=torch.zeros_like(w),
            GG=[None, None],
            Q=[QL.clone(), QR.clone()],
        )
    else:
        opt = METHODS[method][0](w, lr)
    losses = []
    for t in range(steps):
        for group in opt.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * schedule(t, steps)
        with torch.no_grad():
            x = QL.T @ w @ QR
            losses.append(0.5 * float((H * x * x).sum()))
            g_rot = H * x + cfg["noise"] * sqrt_h * torch.from_numpy(noises[t])
            w.grad = QL @ g_rot @ QR.T
        opt.step()
        if not math.isfinite(losses[-1]) or losses[-1] > 1e8:
            losses += [float("inf")] * (steps - t - 1)
            break
    tail = np.asarray(losses[-max(1, steps // 20) :])
    return {
        "method": method,
        "lr": lr,
        "seed": seed,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "final_loss": float(tail.mean()),
        "initial_loss": losses[0],
    }


def _job(args):
    return run(*args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--eval-offset",
        type=int,
        default=300,
        help="first evaluation seed (100 for the F-020 run, 300 since C-013)",
    )
    parser.add_argument("--tag", default="", help="suffix of the output files")
    args = parser.parse_args()
    steps = 150 if args.quick else args.steps
    tune_seeds = [0, 1, 2]
    eval_seeds = list(range(args.eval_offset, args.eval_offset + (4 if args.quick else 12)))
    methods = list(METHODS) + [ORACLE]
    centres = {**{k: v[1] for k, v in METHODS.items()}, ORACLE: 3e-2}
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Stage 1: tuning
    jobs = [
        (mth, lr, cfg, s, steps)
        for cfg in CONFIGS
        for mth in methods
        for lr in lr_grid(centres[mth])
        for s in tune_seeds
    ]
    with mp.Pool(args.workers) as pool:
        tune = pool.map(_job, jobs, chunksize=8)
    best = {}
    for cfg_i, cfg in enumerate(CONFIGS):
        for mth in methods:
            rows = [
                r
                for r in tune
                if r["method"] == mth and r["gamma"] == cfg["gamma"] and r["noise"] == cfg["noise"]
            ]
            by_lr = {}
            for r in rows:
                by_lr.setdefault(r["lr"], []).append(r["final_loss"])
            lr_best = min(by_lr, key=lambda k: np.mean(by_lr[k]))
            grid = lr_grid(centres[mth])
            best[(cfg_i, mth)] = (lr_best, lr_best in (grid[0], grid[-1]))
    print(f"tuning done in {time.time() - t0:.0f}s", flush=True)
    # Stage 2: evaluation on fresh seeds
    jobs = [
        (mth, best[(ci, mth)][0], cfg, s, steps)
        for ci, cfg in enumerate(CONFIGS)
        for mth in methods
        for s in eval_seeds
    ]
    with mp.Pool(args.workers) as pool:
        evals = pool.map(_job, jobs, chunksize=8)
    tag = ("quick" if args.quick else "full") + args.tag
    (RESULTS / f"e25_tune_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in tune))
    (RESULTS / f"e25_eval_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in evals))
    (RESULTS / f"e25_selected_lr_{tag}.json").write_text(
        json.dumps(
            {
                f"{CONFIGS[ci]['gamma']}|{CONFIGS[ci]['noise']}|{m}": {
                    "lr": v[0],
                    "at_grid_edge": v[1],
                }
                for (ci, m), v in best.items()
            },
            indent=1,
        )
    )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
