"""E2.11: landscape of the frame KL and global convergence of the flow (Phase 02, gate G2.6 (i)).

Conjecture 4.1 says that, for arrays whose variance profiles are pairwise distinct, the minimizers
of J on O(m) x O(n) are the true frame up to signed permutations and every other critical point is
a strict saddle. This experiment probes it numerically with the noise-free (population) flow of
`population.py` — exactly Gimbal's update with the expected score in place of the sampled one:

* arrays: separable, non-separable (gamma = 1 and 2), tied row sums (distinct profiles), and a
  near-degenerate pair of rows and of columns; shapes 6x8, 12x16, 24x32; 3 arrays per type;
* 100 Haar-random starting frames per array; flow with alpha = 0.5, damping 1e-6 and Gimbal's trust
  region, until J < 1e-10, the scores vanish, or 4000 steps;
* every end point with J >= 1e-8 is classified by the eigenvalues of the Hessian of J in the skew
  coordinates (finite differences of the exact gradient): strict saddle, local minimum, or not
  converged.

Usage: python experiments/phase2/e211_landscape.py [--starts 100] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import time
from collections import Counter, defaultdict

import numpy as np
from common import make_variances, rand_orth
from population import expected_scores, frame_kl, population_step

RESULTS = pathlib.Path(__file__).parent / "results"
SHAPES = [(6, 8), (12, 16), (24, 32)]
TYPES = ["separable", "nonsep_g1", "nonsep_g2", "tie", "near_degenerate"]


def make_array(kind: str, m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "separable":
        return make_variances(m, n, 0.0, 1.0, rng)
    if kind == "nonsep_g1":
        return make_variances(m, n, 1.0, 1.0, rng)
    if kind == "nonsep_g2":
        return make_variances(m, n, 2.0, 0.5, rng)
    if kind == "tie":
        return make_variances(m, n, 0.5, 1.0, rng, tie=True)
    d = make_variances(m, n, 0.0, 1.0, rng)  # near-degenerate: one row pair, one column pair
    d[1] = d[0] * (1 + 0.05 * rng.standard_normal(n))
    d[:, 1] = d[:, 0] * (1 + 0.05 * rng.standard_normal(m))
    return np.abs(d)


def skew_basis(k: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def gradient_vector(ql, qr, d):
    e_l, e_r, _, _ = expected_scores(ql, qr, d)
    m, n = d.shape
    return np.concatenate([e_l[np.triu_indices(m, 1)], e_r[np.triu_indices(n, 1)]])


def hessian_eigs(ql, qr, d, h: float = 1e-5) -> np.ndarray:
    """Eigenvalues of the Hessian of J in left-trivialized skew coordinates at a critical point."""
    m, n = d.shape
    coords = [("L", i, j) for i, j in skew_basis(m)] + [("R", i, j) for i, j in skew_basis(n)]
    cols = []
    for side, i, j in coords:
        grads = []
        for sgn in (1.0, -1.0):
            if side == "L":
                om = np.zeros((m, m))
                om[i, j], om[j, i] = sgn * h, -sgn * h
                qlp, qrp = ql @ (np.eye(m) + om + om @ om / 2), qr
            else:
                om = np.zeros((n, n))
                om[i, j], om[j, i] = sgn * h, -sgn * h
                qlp, qrp = ql, qr @ (np.eye(n) + om + om @ om / 2)
            grads.append(gradient_vector(qlp, qrp, d))
        cols.append((grads[0] - grads[1]) / (2 * h))
    hess = np.array(cols).T
    return np.linalg.eigvalsh(0.5 * (hess + hess.T))


def is_signed_permutation(ql: np.ndarray, qr: np.ndarray, tol: float = 1e-4) -> bool:
    return all(np.allclose(np.sort(np.abs(q), axis=1)[:, -1], 1.0, atol=tol) for q in (ql, qr))


def run_start(args) -> dict:
    kind, shape, array_id, start_id, max_steps = args
    m, n = shape
    rng = np.random.default_rng(10_000 * array_id + 97 * SHAPES.index(shape)
                                + 7919 * TYPES.index(kind))
    d = make_array(kind, m, n, rng)
    d = d / d.mean()
    srng = np.random.default_rng(1_000_003 * array_id + start_id + 31 * TYPES.index(kind)
                                 + 13 * SHAPES.index(shape))
    ql, qr = rand_orth(m, srng), rand_orth(n, srng)
    j0 = frame_kl(ql, qr, d)
    steps = 0
    status = "max_steps"
    for steps in range(1, max_steps + 1):
        ql, qr, e_l, e_r = population_step(ql, qr, d, alpha=0.5, damping=1e-6)
        if steps % 10 == 0:
            j = frame_kl(ql, qr, d)
            gnorm = max(np.abs(e_l).max(), np.abs(e_r).max())
            if j < 1e-10:
                status = "global"
                break
            if gnorm < 1e-11:
                status = "critical"
                break
    j = frame_kl(ql, qr, d)
    rec = {"type": kind, "shape": list(shape), "array": array_id, "start": start_id,
           "J0": j0, "J": j, "steps": steps, "status": status,
           "signed_permutation": bool(is_signed_permutation(ql, qr))}
    if j >= 1e-8:
        eigs = hessian_eigs(ql, qr, d)
        scale = max(abs(eigs).max(), 1e-12)
        rec["hessian_min"], rec["hessian_max"] = float(eigs.min()), float(eigs.max())
        gnorm = float(np.abs(gradient_vector(ql, qr, d)).max())
        rec["grad_max"] = gnorm
        if gnorm > 1e-6:
            rec["class"] = "not_converged"
        elif eigs.min() < -1e-7 * scale:
            rec["class"] = "strict_saddle"
        else:
            rec["class"] = "local_minimum"
    else:
        rec["class"] = "global_minimum" if rec["signed_permutation"] else "zero_J_not_permutation"
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--arrays", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(kind, shape, a, s, args.max_steps) for shape in SHAPES for kind in TYPES
            for a in range(args.arrays) for s in range(args.starts)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        rows = pool.map(run_start, jobs, chunksize=4)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e211_landscape.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    table = defaultdict(Counter)
    steps = defaultdict(list)
    for r in rows:
        key = (r["type"], f"{r['shape'][0]}x{r['shape'][1]}")
        table[key][r["class"]] += 1
        if r["class"] == "global_minimum":
            steps[key].append(r["steps"])
    classes = ["global_minimum", "strict_saddle", "local_minimum", "not_converged",
               "zero_J_not_permutation"]
    lines = ["# E2.11 landscape and global convergence (generated by e211_landscape.py)", "",
             f"Noise-free flow from {args.starts} Haar-random frames per array, {args.arrays} "
             f"arrays per type and shape; α = 0.5, δ = 1e-6, Gimbal's trust region; "
             f"{time.time() - t0:.0f} s.",
             "", "| type | shape | " + " | ".join(classes) + " | median steps to J < 1e-10 |",
             "|---" * (len(classes) + 3) + "|"]
    ok = True
    for key in sorted(table):
        c = table[key]
        total = sum(c.values())
        frac = c["global_minimum"] / total
        ok &= frac >= 0.99 and c["local_minimum"] == 0 and c["zero_J_not_permutation"] == 0
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(str(c[k]) for k in classes)
                     + f" | {np.median(steps[key]) if steps[key] else float('nan'):.0f} |")
    lines += ["", f"Gate G2.6 (i) (≥ 99% global, no local minima): **{ok}**"]
    (RESULTS / "e211_report.md").write_text("\n".join(lines))
    (RESULTS / "e211_gate.json").write_text(json.dumps({"G2.6_i": bool(ok)}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
