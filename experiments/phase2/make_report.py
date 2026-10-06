"""Assemble experiments/phase2/report.md from the generated per-experiment reports and gates.

Every number in the report comes from a file in ``results/`` written by an analysis or experiment
script; this script only collects them and evaluates the gate summary of phases/02. Gates are
evaluated on the outputs of the current default (change C-015: files tagged ``_c015`` where an
experiment executes the optimizer, re-run with the seeds of C-013); outputs of earlier defaults
stay in ``results/`` and the report names them where they decided something (G2.4's failure,
F-020). The cost model (E2.8) does not depend on C-015 and stays at its C-013 run.

Usage: python experiments/phase2/make_report.py
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"


def load_json(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def section(name: str) -> list[str]:
    path = RESULTS / name
    if not path.exists():
        return [f"*`{name}` not generated yet.*", ""]
    text = path.read_text().splitlines()
    # demote headings by one level so they nest under the experiment heading
    return [("#" + line) if line.startswith("#") else line for line in text] + [""]


def verdict(ok) -> str:
    return "pass" if ok else ("fail" if ok is not None else "pending")


def gate_rows() -> list[tuple[str, str, str]]:
    rows = []
    e21 = load_json("e21_gate.json") or {}
    if e21:
        worst = e21.get("G2.1_k4_separable_worst_ratio_vs_best_peer", float("nan"))
        upper = e21.get("G2.1_k4_separable_worst_upper_ratio", float("nan"))
        g21 = e21.get("G2.1_k4_pass")
        k1 = (e21.get("G2.1_nonseparable_all_wins_best") and e21.get("G2.1_separable_within_25pct")
              and e21.get("G2.1_separable_upper_within_25pct", False))
        rows.append(("G2.1 efficiency", verdict(g21),
                     f"default (frame_every = 4): non-separable cells, below every peer at best "
                     f"memory: {e21.get('G2.1_k4_nonseparable_all_wins_best')}, at matched memory: "
                     f"{e21.get('G2.1_k4_nonseparable_all_wins_matched')}; separable cells, worst "
                     f"ratio to the best peer {worst:.3f} (bootstrap upper bound {upper:.3f}); "
                     f"frame_every = 1: {verdict(k1)}"))
        e22b = load_json("e22b_gate_c015.json")
        cons = e22b["G2.2_consistency"] if e22b else None
        g22 = None if cons is None else bool(e21.get("tie_k4_all_wins_best") and cons)
        rows.append(("G2.2 identifiability", verdict(g22),
                     f"tie suite, default below every peer in every cell: "
                     f"{e21.get('tie_k4_all_wins_best')} (frame_every = 1: "
                     f"{e21.get('G2.2_tie_all_wins')}); consistency over the horizon (E2.2b, both "
                     f"configurations): "
                     f"{verdict(cons)}; the C-006 wording (monotone in memory at 800 steps) "
                     f"failed, F-016"))
        rows.append(("G2.3 tracking", verdict(e21.get("drift_k4_all_wins_best")),
                     "drift suite, default: best-memory error below every peer's; frame_every = 1:"
                     f" {verdict(e21.get('G2.3_drift_all_wins'))}"))
        rows.append(("(E2.4 heavy tails)", verdict(e21.get("tails_k4_all_wins_best")),
                     "Student-t suite, default; reported, not a gate item; frame_every = 1: "
                     f"{verdict(e21.get('G2.4h_tails_all_wins'))}"))
    e25 = load_json("e25_gate_full_c015.json")
    before = load_json("e25_gate_full.json")
    history = (f"; before C-013 (evaluation seeds 100–111): {verdict(before['G2.4'])} (F-020)"
               if before else "")
    rows.append(("G2.4 optimization", verdict(e25["G2.4"] if e25 else None),
                 ("default, evaluation seeds 300–311: "
                  + "; ".join(f"{k}: {'pass' if v['pass'] else 'fail'}"
                              for k, v in e25["configs"].items())
                  + f"; frame_every = 1: {verdict(e25.get('G2.4_k1'))}{history}") if e25
                 else "E2.5 pending"))
    e210 = load_json("e210_theory_vs_simulation.json")
    if e210:
        parts = ", ".join(f"({k}) {verdict(e210[k]['pass'])}" for k in "abcdef" if k in e210)
        pre = e210.get("b_preregistered", {}).get("pass")
        rows.append(("G2.5 theory–simulation", verdict(e210["G2.5"]),
                     f"{parts}; pre-registered (b) with the leading-order formula: "
                     f"{verdict(pre)} (F-013, re-run after the theory revision)"))
    else:
        rows.append(("G2.5 theory–simulation", "pending", "E2.10 pending"))
    e211 = load_json("e211_gate.json")
    e212 = load_json("e212_numerics_c015.json")
    e28 = load_json("e28_cost_model_c013.json")
    cost_ok, cost_txt = None, "E2.8 pending"
    if e28:
        c = e28["analytic"]["optimizers"]
        cost_ok = c["gimbal_k4"]["pct_of_model_c10"] <= c["soap"]["pct_of_model_c10"]
        cost_txt = (f"analytic cost with QR/eigh at 10× matmul, % of model compute: default "
                    f"(k=4) {c['gimbal_k4']['pct_of_model_c10']:.2f}, SOAP "
                    f"{c['soap']['pct_of_model_c10']:.2f}, k=1 "
                    f"{c['gimbal_k1']['pct_of_model_c10']:.2f} (fails: F-019, hence C-012); "
                    f"optimizer state of the default {c['gimbal_k4']['state_Mfloats']:.0f}M "
                    f"floats vs SOAP {c['soap']['state_Mfloats']:.0f}M (F-023, not a gate item); "
                    f"steady-state CPU timing not completed in Phase 02")
    land = e211["G2.6_i"] if e211 else None
    num = e212["G2.6_iii"] if e212 else None
    all6 = None if None in (land, cost_ok, num) else bool(land and cost_ok and num)
    rows.append(("G2.6 landscape, cost, numerics", verdict(all6),
                 f"(i) landscape {verdict(land)}; (ii) cost {verdict(cost_ok)}: "
                 f"{cost_txt}; (iii) numerics {verdict(num)}"))
    return rows


def main() -> None:
    lines = ["# Phase 02 report (generated by make_report.py)", "",
             "Algorithm: Gimbal with the defaults of change C-004, the numerical fix C-008, "
             "`frame_every = 4` (C-012; rows `gimbal_k4`; rows `gimbal` are `frame_every = 1`) "
             "and frame statistics on the empirical-Bayes innovation (C-013), with the "
             "sign-equivariant spectral-norm estimate (C-015) "
             "(`src/gimbal/torch/gimbal.py`, `theory/gimbal_theory.md` v0.8). Rows `*_c013` are "
             "the default before C-015 on the same seeds; rows `*_c012` and "
             "`*_nocenter` are the same configurations before C-013. Seeds: E2.1–E2.4 10–19; E2.5 "
             "tuning 0–2, evaluation 300–311 (100–111 in the run before C-013); E2.9 40–47; "
             "E2.10–E2.12 fixed seeds in the scripts (50+). Seeds 0–9 (E2.1 exploratory), 21–35 "
             "and 200–203 (pilots) are not reported here. Phase 02 trains no language model "
             "(C-006).", "",
             "## Gate summary", "", "| item | result | evidence |", "|---|---|---|"]
    lines += [f"| {a} | **{b}** | {c} |" for a, b, c in gate_rows()]
    lines += [""]
    for title, name in (("E2.1–E2.4 frame estimation", "e21_report.md"),
                        ("E2.2b consistency in a tied plane", "e22b_report_c015.md"),
                        ("E2.5 noisy quadratics", "e25_report_full_c015.md"),
                        ("E2.9 ablations", "e29_report_c015.md"),
                        ("E2.10 theory–simulation agreement", "e210_report.md"),
                        ("E2.11 landscape and global convergence", "e211_report.md"),
                        ("E2.12 numerical behaviour", "e212_report_c015.md")):
        lines += [f"## {title}", ""] + section(name)
    lines += ["Outputs of the default before C-015 (same seeds), kept for the record: E2.2b "
              "`e22b_report_c013.md`, E2.5 `e25_report_full_c013.md`, E2.9 `e29_report_c013.md`, "
              "E2.12 `e212_report_c013.md`; the paired effect of C-015 on E2.1–E2.4 is tabulated "
              "in the E2.1 report.", "",
              "Outputs of the default before C-013, kept for the record: E2.2b "
              "`e22b_report.md`, E2.5 `e25_report_full.md` (G2.4 failed, F-020), E2.9 "
              "`e29_report.md` (baseline `frame_every = 1`), E2.12 `e212_report.md`, E2.8 "
              "`e28_cost_model.json` (CPU timings inside the warm start; the steady-state "
              "re-run was stopped unfinished and is not reported).", ""]
    e28 = load_json("e28_cost_model_c013.json")
    lines += ["## E2.8 cost model", ""]
    if e28:
        a = e28["analytic"]
        lines += [f"Model forward+backward: {a['model_GMAC_per_step']:.0f} GMAC per step "
                  "(125M parameters, 0.5M tokens).", "",
                  "| optimizer | matmul GMAC | QR/eigh GMAC | % of model (QR = 1x) | "
                  "% of model (QR = 10x) | state (M floats) |", "|---|---|---|---|---|---|"]
        for name, r in a["optimizers"].items():
            lines.append(f"| {name} | {r['matmul_GMAC']:.0f} | {r['nonmatmul_GMAC']:.1f} | "
                         f"{r['pct_of_model_c1']:.2f} | {r['pct_of_model_c10']:.2f} | "
                         f"{r['state_Mfloats']:.0f} |")
        bench = e28.get("cpu_benchmark_ms_per_step")
        if bench:
            names = list(next(iter(bench.values())))
            lines += ["", "CPU microbenchmark, ms per optimizer step at steady state (60 untimed "
                      "steps, then 40 timed; float32, 4 threads):", "",
                      "| shape | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
            for shape, v in bench.items():
                lines.append(f"| {shape} | " + " | ".join(f"{v[k]['mean_ms']:.1f}" for k in names)
                             + " |")
        lines.append("")
    (HERE / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
