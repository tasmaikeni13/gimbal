"""G3.3: Phase 02's E2.5 noisy quadratics reproduced with the JAX optimizers (Phase 03).

Same problems, noise, schedule, evaluation seeds (300-311) and learning rates as the PyTorch run
of the current default (``experiments/phase2/results/e25_*_full_c018*``), for the three optimizers
of the TPU study (AdamW, SOAP, Gimbal's default ``gimbal_k4``), in float64 on the CPU. Criterion:
for every configuration and optimizer the mean paired difference (JAX - PyTorch) of the final
loss is within 0.1 seed standard deviations, i.e. far inside seed-level noise.

Usage: python experiments/phase3/e25_jax.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2"))
from common import make_variances, rand_orth  # noqa: E402
from e25_noisy_quadratic import CONFIGS, schedule  # noqa: E402

from gimbal.jax import adamw as jadamw  # noqa: E402
from gimbal.jax import gimbal as jgimbal  # noqa: E402
from gimbal.jax import soap as jsoap  # noqa: E402

RESULTS = ROOT / "experiments" / "phase2" / "results"
OUT = pathlib.Path(__file__).parent / "results"
TAG = "full_c018"
METHODS = ("adamw", "soap", "gimbal_k4")
_JIT: dict = {}


def jitted(name: str, kind):
    """One compiled step per (optimizer, step kind); shapes are fixed, so every run reuses it."""
    key = (name, kind)
    if key not in _JIT:
        if name == "adamw":
            cfg = jadamw.AdamWConfig(b1=0.9, b2=0.95)
            _JIT[key] = jax.jit(lambda s, g, p, lr, t: jadamw.step(s, g, p, lr, t, cfg))
        elif name == "soap":
            cfg = jsoap.SOAPConfig(b1=0.9, b2=0.95, weight_decay=0.0)
            _JIT[key] = jax.jit(lambda s, g, p, lr, t: jsoap.step(s, g, p, lr, t, cfg, kind))
        else:
            cfg = jgimbal.GimbalConfig(b1=0.9, b2=0.95, frame_every=4)
            _JIT[key] = jax.jit(lambda s, g, p, lr, t: jgimbal.step(s, g, p, lr, t, cfg, kind))
    return _JIT[key]


def run(method: str, lr: float, cfg: dict, seed: int, steps: int = 400) -> float:
    """Final loss (mean of the last 5% of steps), as in ``e25_noisy_quadratic.run``."""
    rng = np.random.default_rng(10_000 + seed * 31 + int(cfg["gamma"] * 7)
                                + int(cfg["noise"] * 13))
    m, n = cfg["shape"]
    h = make_variances(m, n, cfg["gamma"], cfg["slope"], rng)
    h = h / h.max()
    ql, qr = rand_orth(m, rng), rand_orth(n, rng)
    x0 = rng.standard_normal((m, n))
    noises = np.random.default_rng(99_000 + seed).standard_normal((steps, m, n))
    w = jnp.asarray(ql @ x0 @ qr.T)
    if method == "adamw":
        state = jadamw.init_state(w)
    elif method == "soap":
        state = jsoap.init_state((m, n), jnp.float64)
    else:
        state = jgimbal.init_state((m, n), jgimbal.GimbalConfig(frame_every=4), jnp.float64)
    losses = []
    gcfg = jgimbal.GimbalConfig(b1=0.9, b2=0.95, frame_every=4)
    scfg = jsoap.SOAPConfig(b1=0.9, b2=0.95, weight_decay=0.0)
    for t in range(steps):
        x = ql.T @ np.asarray(w) @ qr
        losses.append(0.5 * float((h * x * x).sum()))
        g = jnp.asarray(ql @ (h * x + cfg["noise"] * np.sqrt(h) * noises[t]) @ qr.T)
        lr_t = jnp.float64(lr * schedule(t, steps))
        if method == "adamw":
            state, d = jitted("adamw", None)(state, g, w, lr_t, jnp.int32(t + 1))
        elif method == "soap":
            kind = jsoap.schedule(t + 1, scfg)
            state, d = jitted("soap", kind)(state, g, w, lr_t, jnp.int32(max(t, 1)))
        else:
            kind = jgimbal.schedule(t + 1, gcfg)
            if kind.restart or kind.first:
                state, d = jgimbal.step(state, g, w, lr_t, jnp.int32(t + 1), gcfg, kind)
            else:
                state, d = jitted("gimbal", kind)(state, g, w, lr_t, jnp.int32(t + 1))
        w = w + d
        if not math.isfinite(losses[-1]) or losses[-1] > 1e8:
            losses += [float("inf")] * (steps - t - 1)
            break
    return float(np.mean(losses[-max(1, steps // 20):]))


def main() -> None:
    lrs = json.loads((RESULTS / f"e25_selected_lr_{TAG}.json").read_text())
    ref = [json.loads(x) for x in gzip.decompress(
        (RESULTS / f"e25_eval_{TAG}.jsonl.gz").read_bytes()).decode().splitlines() if x]
    rows, checks = [], {}
    for cfg in CONFIGS:
        for method in METHODS:
            lr = lrs[f"{cfg['gamma']}|{cfg['noise']}|{method}"]["lr"]
            torch_rows = {r["seed"]: r["final_loss"] for r in ref if r["method"] == method
                          and r["gamma"] == cfg["gamma"] and r["noise"] == cfg["noise"]}
            diffs = []
            for seed in sorted(torch_rows):
                ours = run(method, lr, cfg, seed)
                rows.append({"method": method, "gamma": cfg["gamma"], "noise": cfg["noise"],
                             "seed": seed, "lr": lr, "final_loss_jax": ours,
                             "final_loss_torch": torch_rows[seed]})
                diffs.append(ours - torch_rows[seed])
            sd = float(np.std(list(torch_rows.values()), ddof=1))
            key = f"gamma={cfg['gamma']},noise={cfg['noise']},{method}"
            checks[key] = {"mean_diff": float(np.mean(diffs)), "max_abs_diff":
                           float(np.max(np.abs(diffs))), "seed_sd": sd,
                           "pass": bool(abs(np.mean(diffs)) <= 0.1 * sd)}
            print(key, checks[key], flush=True)
    gate = {"G3.3": all(c["pass"] for c in checks.values()), "checks": checks}
    OUT.mkdir(exist_ok=True)
    (OUT / "e25_jax.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (OUT / "e25_jax_gate.json").write_text(json.dumps(gate, indent=1))
    print("G3.3:", gate["G3.3"])


if __name__ == "__main__":
    main()
