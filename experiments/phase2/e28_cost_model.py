"""E2.8: optimizer cost model for the 125M configuration (Phase 02, gate G2.6).

1. Analytical multiply–accumulate (MAC) counts per step and optimizer-state memory for every
   optimizer, summed over the hidden matrices of the Phase 05 model (12 layers; fused QKV
   2304x768, output 768x768, SwiGLU gate/up 2048x768, down 768x2048). Non-matmul linear algebra
   (QR, eigh) is reported separately and also converted to "matmul-equivalent" MACs with a penalty
   factor c (QR/eigh on accelerators run far below matmul throughput; c = 1 is optimistic for SOAP).
2. CPU microbenchmarks of one optimizer step per matrix shape (float32, median over steps).

Usage: python experiments/phase2/e28_cost_model.py [--bench]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from gimbal.torch import KLSOAP, SOAP, AROSinkhorn, Gimbal, Muon, NorMuon, SPlus

RESULTS = pathlib.Path(__file__).parent / "results"
LAYERS = 12
SHAPES = [(2304, 768), (768, 768), (2048, 768), (2048, 768), (768, 2048)]  # per layer
TOKENS_PER_STEP = 512 * 1024
PARAMS = 124e6


def unit(m, n):
    return m * m * n + m * n * n


def cube(m, n):
    return m**3 + n**3


def qr_macs(m, n):
    return (4 / 3) * (m**3 + n**3)  # Householder QR of the two square factors


def optimizer_costs(m: int, n: int) -> dict[str, dict]:
    """Per-step MACs (matmul), per-step non-matmul MACs (QR/eigh), and state floats."""
    a, b = min(m, n), max(m, n)  # Muon iterates on the short side
    eigh = 9 * (m**3 + n**3)  # rough MACs of a symmetric eigendecomposition
    costs = {
        "adamw": dict(matmul=0, nonmatmul=0, state=2 * m * n),
        "soap": dict(matmul=5 * unit(m, n) + 3 * cube(m, n) / 10, nonmatmul=qr_macs(m, n) / 10,
                     state=2 * (m * m + n * n) + 2 * m * n),
        "soap_rt": dict(matmul=4 * unit(m, n) + 3 * cube(m, n), nonmatmul=qr_macs(m, n),
                        state=2 * (m * m + n * n) + 2 * m * n),
        "klsoap": dict(matmul=5 * unit(m, n) + 4 * cube(m, n), nonmatmul=qr_macs(m, n),
                       state=2 * (m * m + n * n) + 2 * m * n + m + n),
        "muon": dict(matmul=5 * (2 * a * a * b + a**3), nonmatmul=0, state=m * n),
        "normuon": dict(matmul=5 * (2 * a * a * b + a**3), nonmatmul=0, state=m * n + m),
        "splus": dict(matmul=3 * unit(m, n), nonmatmul=eigh / 100,
                      state=2 * (m * m + n * n) + 2 * m * n),
        "aro": dict(matmul=4 * m * m * n, nonmatmul=(4 / 3) * m**3, state=m * m + m * n),
    }
    for k in (1, 4, 10):
        # 4 units per step (rotate G, rotate M, unrotate update, score); every k steps the Fisher
        # matrix (1 unit), Ω² and Q·P (2 cubes) and one polish (2 cubes). State: Q_L, Q_R, M, V and
        # the flow's variance averages (full and odd-step, Proposition 5.5); score accumulators for
        # k > 1. The 50-step warm start adds m² + n² temporarily and two eigh in total (ignored).
        acc_state = (m * m + n * n) if k > 1 else 0
        costs[f"gimbal_k{k}"] = dict(matmul=4 * unit(m, n) + (unit(m, n) + 4 * cube(m, n)) / k,
                                     nonmatmul=0, state=(m * m + n * n) + 4 * m * n + acc_state)
    return costs


def analytic_table() -> dict:
    model_macs = 3 * PARAMS * TOKENS_PER_STEP  # forward + backward ≈ 6 N T FLOPs = 3 N T MACs
    totals: dict[str, dict] = {}
    for m, n in SHAPES:
        for name, c in optimizer_costs(m, n).items():
            t = totals.setdefault(name, dict(matmul=0.0, nonmatmul=0.0, state=0.0))
            for key in t:
                t[key] += LAYERS * c[key]
    table = {}
    for name, t in totals.items():
        table[name] = {
            "matmul_GMAC": t["matmul"] / 1e9,
            "nonmatmul_GMAC": t["nonmatmul"] / 1e9,
            "pct_of_model_c1": 100 * (t["matmul"] + t["nonmatmul"]) / model_macs,
            "pct_of_model_c10": 100 * (t["matmul"] + 10 * t["nonmatmul"]) / model_macs,
            "state_Mfloats": t["state"] / 1e6,
        }
    return {"model_GMAC_per_step": model_macs / 1e9, "optimizers": table}


BENCH = {
    "adamw": lambda p: torch.optim.AdamW([p], lr=1e-3),
    "soap": lambda p: SOAP([p], lr=1e-3),
    "soap_rt": lambda p: SOAP([p], lr=1e-3, realtime=True),
    "klsoap": lambda p: KLSOAP([p], lr=1e-3),
    "muon": lambda p: Muon([p], lr=1e-3),
    "normuon": lambda p: NorMuon([p], lr=1e-3),
    "splus": lambda p: SPlus([p], lr=1e-3),
    "aro": lambda p: AROSinkhorn([p], lr=1e-3),
    "gimbal_k1": lambda p: Gimbal([p], lr=1e-3),
    "gimbal_k4": lambda p: Gimbal([p], lr=1e-3, frame_every=4),
    "gimbal_k10": lambda p: Gimbal([p], lr=1e-3, frame_every=10),
}


def bench(threads: int = 4, steps: int = 20) -> dict:
    torch.set_num_threads(threads)
    out = {}
    for m, n in sorted(set(SHAPES)):
        gen = torch.Generator().manual_seed(0)
        grads = [torch.randn(m, n, generator=gen) for _ in range(4)]
        for name, ctor in BENCH.items():
            p = torch.nn.Parameter(torch.zeros(m, n))
            opt = ctor(p)
            times = []
            for t in range(steps + 3):
                p.grad = grads[t % 4]
                t0 = time.perf_counter()
                opt.step()
                if t >= 3:
                    times.append(time.perf_counter() - t0)
            out.setdefault(f"{m}x{n}", {})[name] = {"mean_ms": 1e3 * sum(times) / len(times)}
        print(f"benchmarked {m}x{n}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = {"analytic": analytic_table()}
    if args.bench:
        result["cpu_benchmark_ms_per_step"] = bench()
    (RESULTS / "e28_cost_model.json").write_text(json.dumps(result, indent=1))
    a = result["analytic"]
    print(f"model forward+backward: {a['model_GMAC_per_step']:.0f} GMAC/step")
    print(f"{'optimizer':12s} {'matmul GMAC':>12s} {'QR/eigh GMAC':>13s} {'%model c=1':>11s} "
          f"{'%model c=10':>12s} {'state Mfl':>10s}")
    for name, r in a["optimizers"].items():
        print(f"{name:12s} {r['matmul_GMAC']:12.1f} {r['nonmatmul_GMAC']:13.2f} "
              f"{r['pct_of_model_c1']:11.2f} {r['pct_of_model_c10']:12.2f} "
              f"{r['state_Mfloats']:10.1f}")


if __name__ == "__main__":
    main()
