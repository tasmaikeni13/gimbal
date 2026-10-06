"""Analysis of E2.1–E2.4 (frame efficiency, ties, drift, tails) and gate decisions G2.1–G2.3.

For every cell (problem configuration) and seed, each method's score is the time-averaged frame KL
over the second half of the run. Two comparisons:

* best memory: each method at the memory value with the lowest mean over seeds (per cell);
* matched memory: equal effective sample size (``MATCHED_MEMORY`` in common.py).

Gimbal is compared with each peer by a one-sided paired Wilcoxon test over seeds (Holm-corrected
across peers within a cell) and a paired bootstrap interval of the log-ratio.

Usage: python experiments/phase2/analyze_e21.py
"""

from __future__ import annotations

import gzip
import json
import pathlib
from collections import defaultdict

import numpy as np
from common import MATCHED_MEMORY, holm, paired_bootstrap_ci, wilcoxon_less

RESULTS = pathlib.Path(__file__).parent / "results"
PEERS = ["soap", "soap_rt", "klsoap", "pooled_eigh", "kl_eigh"]
METHODS = ["gimbal", "soap", "soap_rt", "klsoap", "pooled_eigh", "kl_eigh", "gimbal_k4",
           "gimbal_noshrink", "gimbal_v03"]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    """Rows of a ``.jsonl`` file, or of its gzipped copy ``.jsonl.gz`` (how results are stored)."""
    if path.exists():
        text = path.read_text()
    elif path.with_suffix(".jsonl.gz").exists():
        text = gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes()).decode()
    else:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load(suite: str) -> list[dict]:
    # base grid + equal-treatment extension (run_e21_ext.sh)
    return (read_jsonl(RESULTS / f"e21_{suite}.jsonl")
            + read_jsonl(RESULTS / f"e21_{suite}_ext.jsonl"))


def cell_key(r: dict) -> tuple:
    return (r["gamma"], r["slope"], tuple(r["shape"]), r["tie"], r["nu"], r["drift"])


def summarize(rows: list[dict]) -> list[dict]:
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[cell_key(r)].append(r)
    out = []
    for key, rs in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
        # score[method][memory][seed]
        score = defaultdict(lambda: defaultdict(dict))
        for r in rs:
            score[r["method"]][r["memory"]][r["seed"]] = r["kl_second_half"]
        seeds = sorted(next(iter(score["gimbal"].values())).keys())
        best = {}
        for method, mems in score.items():
            mem = min(mems, key=lambda m: np.mean([mems[m][s] for s in seeds]))
            best[method] = (mem, np.array([mems[mem][s] for s in seeds]))
        matched = {m: np.array([score[m][MATCHED_MEMORY[m]][s] for s in seeds])
                   for m in MATCHED_MEMORY if m in score}
        row = {"gamma": key[0], "slope": key[1], "shape": key[2], "tie": key[3], "nu": key[4],
               "drift": key[5], "kappa": float(np.mean([r["kappa"] for r in rs])),
               "n_seeds": len(seeds),
               # best memory on the edge of the grid actually run (more than one value)
               "edge": sorted(m for m, (mem, _) in best.items()
                              if len(score[m]) > 1 and mem in (min(score[m]), max(score[m]))),
               "by_memory": {m: {str(mem): float(np.mean([v[s] for s in seeds]))
                                 for mem, v in sorted(mems.items())}
                             for m, mems in score.items()}}
        for label, table in (("best", {m: v[1] for m, v in best.items()}), ("matched", matched)):
            row[label] = {
                "mean_kl": {m: float(v.mean()) for m, v in table.items()},
                "memory": {m: best[m][0] for m in best} if label == "best" else MATCHED_MEMORY,
            }
            # Gimbal (defaults) and, for the record, the amortized variant against every peer.
            for ours in ("gimbal", "gimbal_k4"):
                if ours not in table:
                    continue
                g = table[ours]
                pvals, ratios = {}, {}
                for peer in PEERS:
                    if peer not in table:
                        continue
                    pvals[peer] = wilcoxon_less(g, table[peer])
                    ratios[peer] = paired_bootstrap_ci(np.log(table[peer]) - np.log(g))
                adj = holm(pvals)
                suffix = "" if ours == "gimbal" else "_k4"
                row[label].update({
                    f"gimbal{suffix}_wins_all": bool(all(adj[p] < 0.05 for p in adj)),
                    f"holm_p{suffix}": adj,
                    # exp(mean log ratio): geometric-mean factor by which a peer's KL exceeds ours
                    f"kl_factor_vs_gimbal{suffix}": {
                        p: [float(np.exp(x)) for x in v] for p, v in ratios.items()
                    },
                })
        out.append(row)
    return out


def per_seed_best(rows: list[dict], cell: dict, method: str) -> dict:
    """Per-seed scores of a method at its best memory (lowest mean over seeds) in one cell."""
    by_mem = defaultdict(dict)
    for r in rows:
        if (cell_key(r) == (cell["gamma"], cell["slope"], tuple(cell["shape"]), cell["tie"],
                            cell["nu"], cell["drift"]) and r["method"] == method):
            by_mem[r["memory"]][r["seed"]] = r["kl_second_half"]
    best = min(by_mem, key=lambda mem: np.mean(list(by_mem[mem].values())))
    return by_mem[best]


def separable_upper_ratio(rows: list[dict], cell: dict) -> float:
    """Bootstrap 95% upper bound of the geometric-mean ratio Gimbal / best peer (paired)."""
    best_peer = min(PEERS, key=lambda p: cell["best"]["mean_kl"][p])
    g = per_seed_best(rows, cell, "gimbal")
    q = per_seed_best(rows, cell, best_peer)
    diff = np.array([np.log(g[s]) - np.log(q[s]) for s in sorted(g)])
    return float(np.exp(paired_bootstrap_ci(diff)[2]))


def random_effects(summary: list[dict]) -> dict:
    """DerSimonian–Laird pooling over cells of the mean paired log-ratio peer / Gimbal."""
    out = {}
    for peer in PEERS:
        means, variances = [], []
        for r in summary:
            lo, hi = r["best"]["kl_factor_vs_gimbal"][peer][1:]
            mean = np.log(r["best"]["kl_factor_vs_gimbal"][peer][0])
            se = (np.log(hi) - np.log(lo)) / (2 * 1.96)  # from the bootstrap interval
            means.append(mean)
            variances.append(max(se**2, 1e-12))
        y, v = np.array(means), np.array(variances)
        w = 1 / v
        fixed = np.sum(w * y) / np.sum(w)
        q = float(np.sum(w * (y - fixed) ** 2))
        k = len(y)
        tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))
        w_re = 1 / (v + tau2)
        mu = np.sum(w_re * y) / np.sum(w_re)
        se_mu = np.sqrt(1 / np.sum(w_re))
        out[peer] = {"factor": float(np.exp(mu)), "lo": float(np.exp(mu - 1.96 * se_mu)),
                     "hi": float(np.exp(mu + 1.96 * se_mu)),
                     "i2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0, "k": k}
    return out


def fmt_cell(r: dict) -> str:
    parts = [f"γ={r['gamma']}", f"s={r['slope']}", f"{r['shape'][0]}x{r['shape'][1]}"]
    if r["tie"]:
        parts.append("tie")
    if r["nu"]:
        parts.append(f"t(ν={r['nu']:g})")
    if r["drift"]:
        parts.append(f"drift={r['drift']:g}")
    return " ".join(parts)


def markdown(summary: list[dict], label: str) -> str:
    methods = [m for m in METHODS if label == "best" or m in MATCHED_MEMORY]
    lines = ["| cell | κ | " + " | ".join(methods) + " | Gimbal better than all (Holm p<0.05) |"
             + (" grid edge |" if label == "best" else ""),
             "|---" * (len(methods) + 3 + (label == "best")) + "|"]
    for r in summary:
        t = r[label]
        vals = " | ".join(f"{t['mean_kl'].get(m, float('nan')):.2f}" for m in methods)
        edge = f" {', '.join(r['edge']) or '—'} |" if label == "best" else ""
        lines.append(f"| {fmt_cell(r)} | {r['kappa']:.2f} | {vals} | "
                     f"{'yes' if t['gimbal_wins_all'] else 'no'} |{edge}")
    return "\n".join(lines)


def memory_table(summary: list[dict], methods: list[str]) -> str:
    """Mean KL per memory value: an identifiable frame keeps improving with longer memory."""
    lines = []
    for r in summary:
        lines += [f"**{fmt_cell(r)}**", ""]
        for m in methods:
            vals = ", ".join(f"{mem}: {v:.2f}" for mem, v in r["by_memory"][m].items())
            lines.append(f"* {m}: {vals}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = ["# E2.1–E2.4 frame efficiency (generated by analyze_e21.py)", ""]
    gates = {}
    for suite in ("main", "tie", "drift", "tails"):
        rows = load(suite)
        if not rows:
            continue
        summary = summarize(rows)
        (RESULTS / f"e21_{suite}_summary.json").write_text(json.dumps(summary, indent=1,
                                                                       default=str))
        for label in ("best", "matched"):
            report += [f"## Suite `{suite}` — {label} memory (mean frame KL, lower is better)", "",
                       markdown(summary, label), ""]
        gates[f"{suite}_k4_all_wins_best"] = all(r["best"].get("gimbal_k4_wins_all", False)
                                                 for r in summary)
        if suite == "main":
            sep = [r for r in summary if r["gamma"] == 0.0]
            nonsep = [r for r in summary if r["gamma"] > 0.0]
            gates["G2.1_nonseparable_all_wins_best"] = all(r["best"]["gimbal_wins_all"]
                                                           for r in nonsep)
            gates["G2.1_nonseparable_all_wins_matched"] = all(r["matched"]["gimbal_wins_all"]
                                                              for r in nonsep)
            worst = max(
                r["best"]["mean_kl"]["gimbal"] / min(r["best"]["mean_kl"][p] for p in PEERS)
                for r in sep
            )
            gates["G2.1_separable_worst_ratio_vs_best_peer"] = worst
            gates["G2.1_separable_within_25pct"] = bool(worst <= 1.25)
            # C-006: the paired bootstrap 95% upper bound of the geometric-mean ratio Gimbal / best
            # peer must also be below 1.25 in every separable cell.
            uppers = [separable_upper_ratio(rows, r) for r in sep]
            gates["G2.1_separable_worst_upper_ratio"] = max(uppers)
            gates["G2.1_separable_upper_within_25pct"] = bool(max(uppers) <= 1.25)
            pooled = random_effects(summary)
            report += ["## Random-effects pooling across main-suite cells", "",
                       "Paired log-ratio log(KL_peer / KL_Gimbal) per seed at best memories; "
                       "DerSimonian–Laird pooling of the per-cell means (cells as studies).", "",
                       "| peer | pooled factor | 95% CI | I² | cells |", "|---|---|---|---|---|"]
            for peer, v in pooled.items():
                report.append(f"| {peer} | {v['factor']:.2f} | [{v['lo']:.2f}, {v['hi']:.2f}] |"
                              f" {v['i2']:.2f} | {v['k']} |")
            report.append("")
            gates["random_effects_factor_vs_gimbal"] = {k: v["factor"] for k, v in pooled.items()}
        if suite == "tie":
            gates["G2.2_tie_all_wins"] = all(r["best"]["gimbal_wins_all"] for r in summary)
            report += ["### Tie suite: mean frame KL by memory (longer memory to the right for "
                       "EMA methods, to the left for Gimbal)", "",
                       memory_table(summary, ["gimbal", "soap", "pooled_eigh", "klsoap"]), ""]
        if suite == "drift":
            gates["G2.3_drift_all_wins"] = all(r["best"]["gimbal_wins_all"] for r in summary)
        if suite == "tails":
            gates["G2.4h_tails_all_wins"] = all(r["best"]["gimbal_wins_all"] for r in summary)
    report += ["## Gate items", "", "```", json.dumps(gates, indent=1), "```"]
    (RESULTS / "e21_report.md").write_text("\n".join(report))
    (RESULTS / "e21_gate.json").write_text(json.dumps(gates, indent=1))
    print("\n".join(report[-6:]))


if __name__ == "__main__":
    main()
