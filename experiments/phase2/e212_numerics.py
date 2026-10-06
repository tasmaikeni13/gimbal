"""E2.12: numerical behaviour of Gimbal's frame estimation (Phase 02, gate G2.6 (iii)).

1. float32 vs float64 on four E2.1 cells (32x48), seeds 50-52, rates 0.005 and 0.02: ratio of the
   second-half mean frame KL, and the largest orthogonality defect seen.
2. 10^4 steps on a 64x64 non-separable stream in float32 and float64: largest ||Q^T Q - I||_max.
3. Scale invariance (Theorem 8.2): gradients multiplied by 1e-30 ... 1e30 (float64) and 1e-15 ...
   1e15 (float32); largest frame difference to scale 1 after 300 steps.
4. Degenerate inputs in both precisions (500 steps, 16x24): zero gradients first, all-zero
   gradients, a dead row, a dead column, rank-one gradients, variances spanning 1e12. Every state
   tensor must stay finite and the frames orthogonal.

The optimizer runs with lr = 0 (frame estimation only), as in E2.1.

Usage: python experiments/phase2/e212_numerics.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from common import KRDStream, frame_kl, make_variances, rand_orth

from gimbal.torch import Gimbal

RESULTS = pathlib.Path(__file__).parent / "results"


def stream(m, n, gamma, slope, seed, steps):
    rng = np.random.default_rng(seed)
    d = make_variances(m, n, gamma, slope, rng)
    ql, qr = rand_orth(m, rng), rand_orth(n, rng)
    s = KRDStream(ql.copy(), qr.copy(), d, rng)
    return [g for _, g in zip(range(steps), s, strict=False)], ql, qr, d


def run(grads, dtype, rate=0.02, scale=1.0, check_every=1, frame_every=1):
    """Run frame estimation; return the frame trace (every ``check_every`` steps) and the max
    orthogonality defect and finiteness over all steps."""
    m, n = grads[0].shape
    p = torch.nn.Parameter(torch.zeros(m, n, dtype=dtype))
    opt = Gimbal([p], lr=0.0, rot_rate=rate, frame_every=frame_every)
    frames, defect, finite = [], 0.0, True
    for t, g in enumerate(grads):
        p.grad = torch.from_numpy(g * scale).to(dtype)
        opt.step()
        st = opt.state[p]
        for key in ("QL", "QR", "M", "V", "VF", "VF_odd"):
            if key in st and st[key] is not None:
                finite &= bool(torch.isfinite(st[key]).all())
        if t % check_every == 0 or t == len(grads) - 1:
            ql, qr = st["QL"].double(), st["QR"].double()
            for q in (ql, qr):
                eye = torch.eye(len(q), dtype=q.dtype)
                defect = max(defect, float((q.T @ q - eye).abs().max()))
            frames.append((ql.numpy().copy(), qr.numpy().copy()))
    return frames, defect, finite


def part1() -> dict:
    out, ok = [], True
    for gamma, slope in ((0.0, 0.5), (0.0, 1.5), (1.0, 1.0), (2.0, 0.5)):
        for rate in (0.005, 0.02):
            for k in (1, 4):  # k = 4 is the default since C-012
                ratios = []
                for seed in (50, 51, 52):
                    grads, ql, qr, d = stream(32, 48, gamma, slope, seed, 800)
                    j = {}
                    for dtype in (torch.float32, torch.float64):
                        frames, defect, finite = run(grads, dtype, rate, frame_every=k)
                        trace = [frame_kl(a, b, ql, qr, d) for a, b in frames]
                        j[dtype] = float(np.mean(trace[400:]))
                        ok &= finite
                        if dtype == torch.float32:
                            out.append({"cell": [gamma, slope], "rate": rate, "k": k,
                                        "seed": seed, "fp32_defect": defect})
                    ratios.append(j[torch.float32] / j[torch.float64])
                r = float(np.mean(ratios))
                ok &= 0.9 <= r <= 1.1
                out.append({"cell": [gamma, slope], "rate": rate, "k": k,
                            "J_ratio_fp32_fp64": r})
    return {"records": out, "pass": bool(ok)}


def part2() -> dict:
    grads, _, _, _ = stream(64, 64, 1.0, 1.0, 53, 10_000)
    res, ok = {}, True
    for k in (1, 4):
        for dtype, name in ((torch.float32, "fp32"), (torch.float64, "fp64")):
            _, defect, finite = run(grads, dtype, 0.02, check_every=10, frame_every=k)
            res[f"{name}_k{k}"] = {"max_defect": defect, "finite": finite}
            ok &= finite and (name != "fp32" or defect <= 1e-5)
    return {**res, "pass": bool(ok)}


def mod_signed_permutation(q1: np.ndarray, q2: np.ndarray) -> float:
    """Distance of two frames modulo signed permutations of their columns (0 iff equivalent)."""
    return float(np.max(1 - np.abs(q1.T @ q2).max(axis=1)))


def run_init(grads, dtype, scale, init):
    m, n = grads[0].shape
    p = torch.nn.Parameter(torch.zeros(m, n, dtype=dtype))
    opt = Gimbal([p], lr=0.0, rot_rate=0.02, init=init, frame_every=1)
    for g in grads:
        p.grad = torch.from_numpy(g * scale).to(dtype)
        opt.step()
    st = opt.state[p]
    finite = all(bool(torch.isfinite(st[k]).all()) for k in ("QL", "QR", "V", "VF", "VF_odd"))
    return st["QL"].double().numpy(), st["QR"].double().numpy(), finite


def part3() -> dict:
    """Scale invariance of the frame flow (Theorem 8.2) from gauge-free initial frames.

    Gated (G2.6 (iii) as re-specified by F-015): the default configuration on a square stream (the
    eigh initialization of a full-rank factor is unique up to signs) and ``init="identity"`` on a
    32x48 stream; frames after 300 steps at scales 1e+-15 and 1e+-30 must equal those at scale 1
    to 1e-8 (float64). Reported: the default configuration on the 32x48 stream, where the first
    gradient's G^T G is rank-deficient and eigh picks an arbitrary basis of its null space (the
    pre-registered implementation of this test), and float32 at 1e+-15.
    """
    res, ok = {}, True
    cases = (("square_default", (32, 32), "pooled", True),
             ("rect_identity", (32, 48), "identity", True),
             ("rect_default", (32, 48), "pooled", False))
    for name, shape, init, gated in cases:
        grads, _, _, _ = stream(*shape, 1.0, 1.0, 54, 300)
        for dtype, scales, tag in ((torch.float64, (1e-30, 1e-15, 1e15, 1e30), "fp64"),
                                   (torch.float32, (1e-15, 1e15), "fp32")):
            base = run_init(grads, dtype, 1.0, init)
            for c in scales:
                ql, qr, finite = run_init(grads, dtype, c, init)
                raw = max(float(np.abs(ql - base[0]).max()), float(np.abs(qr - base[1]).max()))
                msp = max(mod_signed_permutation(ql, base[0]), mod_signed_permutation(qr, base[1]))
                is_gate = gated and tag == "fp64"
                res[f"{name}_{tag}_{c:g}"] = {"raw_deviation": raw, "mod_signed_perm": msp,
                                             "finite": finite, "gated": is_gate}
                ok &= finite and (raw <= 1e-8 or not is_gate)
    return {**res, "pass": bool(ok)}


def part4() -> dict:
    rng = np.random.default_rng(55)
    m, n, steps = 16, 24, 500
    d = make_variances(m, n, 1.0, 1.0, rng)
    ql, qr = rand_orth(m, rng), rand_orth(n, rng)
    base = [ql @ (np.sqrt(d) * rng.standard_normal((m, n))) @ qr.T for _ in range(steps)]
    cases = {
        "zeros_first": [np.zeros((m, n))] * 20 + base[20:],
        "all_zero": [np.zeros((m, n))] * steps,
        "dead_row": [ql @ (np.sqrt(np.vstack([np.zeros((1, n)), d[1:]]))
                           * rng.standard_normal((m, n))) @ qr.T for _ in range(steps)],
        "dead_col": [ql @ (np.sqrt(np.hstack([np.zeros((m, 1)), d[:, 1:]]))
                           * rng.standard_normal((m, n))) @ qr.T for _ in range(steps)],
        "rank_one": [np.outer(rng.standard_normal(m), rng.standard_normal(n))
                     for _ in range(steps)],
        "range_1e12": [ql @ (np.sqrt(10.0 ** rng.uniform(-6, 6, (m, n)))
                             * rng.standard_normal((m, n))) @ qr.T for _ in range(steps)],
        # noise-free: the momentum equals the gradient, so the centered statistic (C-013) is
        # rounding residue
        "constant": [base[0]] * steps,
    }
    res, ok = {}, True
    for name, grads in cases.items():
        for dtype, tag in ((torch.float32, "fp32"), (torch.float64, "fp64")):
            _, defect, finite = run(grads, dtype, 0.02)
            res[f"{name}_{tag}"] = {"finite": finite, "max_defect": defect}
            ok &= finite and defect <= (1e-5 if tag == "fp32" else 1e-10)
    return {**res, "pass": bool(ok)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="suffix of the output files (re-runs)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    res = {"1_precision": part1(), "2_long_run": part2(), "3_scale": part3(),
           "4_degenerate": part4()}
    res["G2.6_iii"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"e212_numerics{args.tag}.json").write_text(json.dumps(res, indent=1))
    p1 = [r for r in res["1_precision"]["records"] if "J_ratio_fp32_fp64" in r]
    lines = ["# E2.12 numerical behaviour (generated by e212_numerics.py)", "",
             "## 1. float32 vs float64 (frame KL ratio, mean over seeds 50–52)", "",
             "| cell (γ, s) | rate | frame_every | J fp32 / J fp64 |", "|---|---|---|---|"]
    lines += [f"| {r['cell']} | {r['rate']} | {r['k']} | {r['J_ratio_fp32_fp64']:.4f} |"
              for r in p1]
    d32 = max(r["fp32_defect"] for r in res["1_precision"]["records"] if "fp32_defect" in r)
    lines += ["", f"Largest float32 orthogonality defect in these runs: {d32:.1e}.", "",
              "## 2. 10⁴ steps, 64×64", "",
              ", ".join(f"{k}: max defect {v['max_defect']:.1e}"
                        for k, v in res["2_long_run"].items() if isinstance(v, dict)) + ".", "",
              "## 3. Gradient scale (frame deviation from scale 1 after 300 steps)", "",
              "Gated rows start from gauge-free frames (square stream with the default "
              "initialization; identity initialization on 32×48). `rect_default` is the "
              "pre-registered implementation: the first gradient's GᵀG is rank-deficient, so "
              "eigh's basis of its null space is arbitrary and rounding differences select "
              "different "
              "bases (F-015); the flow then re-converges, as the distance modulo signed "
              "permutations shows.", "",
              "| case, precision, scale | raw deviation | modulo signed permutations | finite "
              "| gated |", "|---|---|---|---|---|"]
    for k, v in res["3_scale"].items():
        if isinstance(v, dict):
            lines.append(f"| {k} | {v['raw_deviation']:.1e} | {v['mod_signed_perm']:.1e} | "
                         f"{v['finite']} | {v['gated']} |")
    lines += ["", "## 4. Degenerate inputs (500 steps)", "", "| case | finite | max defect |",
              "|---|---|---|"]
    for k, v in res["4_degenerate"].items():
        if isinstance(v, dict):
            lines.append(f"| {k} | {v['finite']} | {v['max_defect']:.1e} |")
    lines += ["", "Pass by part: " + ", ".join(f"{k}: {v['pass']}" for k, v in res.items()
                                               if isinstance(v, dict)),
              "", f"**G2.6 (iii): {res['G2.6_iii']}**"]
    (RESULTS / f"e212_report{args.tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
