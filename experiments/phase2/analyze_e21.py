"""Analysis of E2.1–E2.4 (frame efficiency, ties, drift, tails) and gate decisions G2.1–G2.3.

For every cell (problem configuration) and seed, each method's score is the time-averaged frame KL
over the second half of the run. Two comparisons:

* best memory: each method at the memory value with the lowest mean over seeds (per cell);
* matched memory: equal effective sample size (``MATCHED_MEMORY`` in common.py).

Gimbal is compared with each peer by a one-sided paired Wilcoxon test over seeds (Holm-corrected
across peers within a cell) and a paired bootstrap interval of the log-ratio.

Rows of ``gimbal`` and ``gimbal_k4`` re-run after change C-013 (files ``*_c013``) replace the
earlier ones; those stay in the analysis as the ablation ``*_c012`` (the same configuration with
frame statistics on the raw gradient), compared with the default on the same streams. Rows re-run
after change C-015 (sign-equivariant spectral-norm estimate, files ``*_c015``) and after C-018
(adaptive amortization, files ``*_c018``; same seeds throughout) replace their predecessors, which
stay as ``*_c013`` and ``*_c015`` and give the paired effect of each change.

Usage: python experiments/phase2/analyze_e21.py
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from common import MATCHED_MEMORY, holm, paired_bootstrap_ci, wilcoxon_less

RESULTS = pathlib.Path(__file__).parent / "results"
PEERS = ["soap", "soap_rt", "klsoap", "pooled_eigh", "kl_eigh"]
METHODS = [
    "gimbal",
    "soap",
    "soap_rt",
    "klsoap",
    "pooled_eigh",
    "kl_eigh",
    "gimbal_k4",
    "gimbal_k4_c012",
    "gimbal_noshrink",
    "gimbal_v03",
]
RERUN = ("gimbal", "gimbal_k4")  # re-run after C-013 and again after C-015
# Optimizers a practitioner would run; the exact-eigenvector controls (pooled_eigh, kl_eigh)
# recompute an eigendecomposition every step and are idealized references.
PRACTICAL = ("soap", "soap_rt", "klsoap")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    """Rows of a ``.jsonl`` file, or of its gzipped copy ``.jsonl.gz`` (how results are stored)."""
    if path.exists():
        text = path.read_text()
    elif path.with_suffix(".jsonl.gz").exists():
        text = gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes()).decode()
    else:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# Re-runs that replace the default's rows, in order. ``--c019`` adds the 125M study's momentum
# (beta_1 = 0.95, C-019) as a robustness check with suffixed outputs; the library default is _c018.
CHANGES = [("c013", "c012"), ("c015", "c013"), ("c018", "c015")]
SUFFIX = ""


def load(suite: str) -> list[dict]:
    """Base grid + equal-treatment extension; rows re-run after a peer fix (``*_peerfix``, C-011)
    replace the rows of the same cell, seed, method and memory. After C-013 the re-run rows of
    ``RERUN`` replace the earlier ones, which are kept under ``<method>_c012``."""
    rows = {}
    for name in (
        f"e21_{suite}.jsonl",
        f"e21_{suite}_ext.jsonl",
        f"e21_{suite}_peerfix.jsonl",
        f"e21_{suite}_ext_peerfix.jsonl",
    ):
        for r in read_jsonl(RESULTS / name):
            rows[(cell_key(r), r["seed"], r["method"], r["memory"])] = r
    for change, previous in CHANGES:
        rerun = read_jsonl(RESULTS / f"e21_{suite}_{change}.jsonl") + read_jsonl(
            RESULTS / f"e21_{suite}_ext_{change}.jsonl"
        )
        if not rerun:
            continue
        for key in [k for k in rows if k[2] in RERUN]:
            old = rows.pop(key)
            old = {**old, "method": f"{old['method']}_{previous}"}
            rows[(key[0], key[1], old["method"], key[3])] = old
        for r in rerun:
            rows[(cell_key(r), r["seed"], r["method"], r["memory"])] = r
    return list(rows.values())


def centering_effect(
    rows: list[dict],
    summary: list[dict],
    after: str = "_c013",
    before: str = "_c012",
    change: str = "C-013",
) -> list[str]:
    """Paired effect of a change on frame KL: ``<method><after>`` / ``<method><before>`` at each
    one's best memory (``after = ""`` is the current default). C-013's effect is measured on the
    rows before C-015, so that the two changes are not mixed."""
    lines = [
        f"| cell | k = 4: with / before {change} | k = 1: with / before {change} |",
        "|---|---|---|",
    ]
    for cell in summary:
        cells = []
        for ours in RERUN[::-1]:
            have = cell["best"]["mean_kl"]
            num = f"{ours}{after}" if after and f"{ours}{after}" in have else ours
            if f"{ours}{before}" not in have:
                cells.append("—")
                continue
            a = per_seed_best(rows, cell, num)
            b = per_seed_best(rows, cell, f"{ours}{before}")
            diff = np.array([np.log(a[s]) - np.log(b[s]) for s in sorted(a)])
            mean, lo, hi = (float(np.exp(x)) for x in paired_bootstrap_ci(diff))
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {fmt_cell(cell)} | " + " | ".join(cells) + " |")
    return lines


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
        matched = {
            m: np.array([score[m][MATCHED_MEMORY[m]][s] for s in seeds])
            for m in MATCHED_MEMORY
            if m in score
        }
        row = {
            "gamma": key[0],
            "slope": key[1],
            "shape": key[2],
            "tie": key[3],
            "nu": key[4],
            "drift": key[5],
            "kappa": float(np.mean([r["kappa"] for r in rs])),
            "n_seeds": len(seeds),
            # best memory on the edge of the grid actually run (more than one value)
            "edge": sorted(
                m
                for m, (mem, _) in best.items()
                if len(score[m]) > 1 and mem in (min(score[m]), max(score[m]))
            ),
            "by_memory": {
                m: {
                    str(mem): float(np.mean([v[s] for s in seeds]))
                    for mem, v in sorted(mems.items())
                }
                for m, mems in score.items()
            },
        }
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
                row[label].update(
                    {
                        f"gimbal{suffix}_wins_all": bool(all(adj[p] < 0.05 for p in adj)),
                        f"holm_p{suffix}": adj,
                        # exp(mean log ratio): geometric-mean factor by which a peer's KL
                        # exceeds ours
                        f"kl_factor_vs_gimbal{suffix}": {
                            p: [float(np.exp(x)) for x in v] for p, v in ratios.items()
                        },
                    }
                )
        out.append(row)
    return out


def per_seed_best(rows: list[dict], cell: dict, method: str) -> dict:
    """Per-seed scores of a method at its best memory (lowest mean over seeds) in one cell."""
    by_mem = defaultdict(dict)
    for r in rows:
        if (
            cell_key(r)
            == (
                cell["gamma"],
                cell["slope"],
                tuple(cell["shape"]),
                cell["tie"],
                cell["nu"],
                cell["drift"],
            )
            and r["method"] == method
        ):
            by_mem[r["memory"]][r["seed"]] = r["kl_second_half"]
    best = min(by_mem, key=lambda mem: np.mean(list(by_mem[mem].values())))
    return by_mem[best]


def separable_upper_ratio(rows: list[dict], cell: dict, ours: str = "gimbal") -> float:
    """Bootstrap 95% upper bound of the geometric-mean ratio ours / best peer (paired)."""
    best_peer = min(PEERS, key=lambda p: cell["best"]["mean_kl"][p])
    g = per_seed_best(rows, cell, ours)
    q = per_seed_best(rows, cell, best_peer)
    diff = np.array([np.log(g[s]) - np.log(q[s]) for s in sorted(g)])
    return float(np.exp(paired_bootstrap_ci(diff)[2]))


def random_effects(summary: list[dict], suffix: str = "_k4") -> dict:
    """DerSimonian–Laird pooling over cells of the mean paired log-ratio peer / Gimbal
    (``suffix`` "_k4": the default configuration, "": frame_every = 1)."""
    out = {}
    key = f"kl_factor_vs_gimbal{suffix}"
    for peer in PEERS:
        means, variances = [], []
        for r in summary:
            lo, hi = r["best"][key][peer][1:]
            mean = np.log(r["best"][key][peer][0])
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
        out[peer] = {
            "factor": float(np.exp(mu)),
            "lo": float(np.exp(mu - 1.96 * se_mu)),
            "hi": float(np.exp(mu + 1.96 * se_mu)),
            "i2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0,
            "k": k,
        }
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
    lines = [
        "| cell | κ | "
        + " | ".join(methods)
        + " | Gimbal better than all (Holm p<0.05) |"
        + (" grid edge |" if label == "best" else ""),
        "|---" * (len(methods) + 3 + (label == "best")) + "|",
    ]
    for r in summary:
        t = r[label]
        vals = " | ".join(f"{t['mean_kl'].get(m, float('nan')):.2f}" for m in methods)
        edge = f" {', '.join(r['edge']) or '—'} |" if label == "best" else ""
        lines.append(
            f"| {fmt_cell(r)} | {r['kappa']:.2f} | {vals} | "
            f"{'yes' if t['gimbal_wins_all'] else 'no'} |{edge}"
        )
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
    global SUFFIX
    if "--c019" in sys.argv:
        CHANGES.append(("c019", "c018"))
        SUFFIX = "_c019"
    report = ["# E2.1–E2.4 frame efficiency (generated by analyze_e21.py)", ""]
    gates = {}
    for suite in ("main", "tie", "drift", "tails"):
        rows = load(suite)
        if not rows:
            continue
        summary = summarize(rows)
        (RESULTS / f"e21_{suite}_summary{SUFFIX}.json").write_text(
            json.dumps(summary, indent=1, default=str)
        )
        for label in ("best", "matched"):
            report += [
                f"## Suite `{suite}` — {label} memory (mean frame KL, lower is better)",
                "",
                markdown(summary, label),
                "",
            ]
        if any(r["method"].endswith("_c018") for r in rows):
            report += [
                f"### Suite `{suite}`: effect of C-019 (momentum beta_1 = 0.95 instead of 0.9)",
                "",
                "Same seeds and streams; geometric mean over seeds of the paired frame-KL "
                "ratio, beta_1 = 0.95 / 0.9, at each one's best memory, percentile-bootstrap "
                "95% interval. The momentum enters the frame statistic only through the "
                "innovation, whose factor is about zero on these zero-mean streams.",
                "",
                *centering_effect(rows, summary, after="", before="_c018", change="C-019"),
                "",
            ]
        if any(r["method"].endswith("_c015") for r in rows):
            report += [
                f"### Suite `{suite}`: effect of C-018 (adaptive amortization; the memory "
                "grid is unchanged, so the new default rate does not enter here)",
                "",
                "Same seeds and streams; geometric mean over seeds of the paired frame-KL "
                "ratio, current default / the same configuration before C-018, at each "
                "one's best memory, percentile-bootstrap 95% interval.",
                "",
                *centering_effect(rows, summary, after="", before="_c015", change="C-018"),
                "",
            ]
        if any(r["method"].endswith("_c013") for r in rows):
            report += [
                f"### Suite `{suite}`: effect of C-015 (sign-equivariant spectral-norm estimate)",
                "",
                "Same seeds and streams; geometric mean over seeds of the paired frame-KL "
                "ratio, the configuration after C-015 / before C-015, at each "
                "one's best memory, percentile-bootstrap 95% interval.",
                "",
                *centering_effect(rows, summary, after="_c015", before="_c013", change="C-015"),
                "",
            ]
        if any(r["method"].endswith("_c012") for r in rows):
            report += [
                f"### Suite `{suite}`: effect of C-013 (frame statistics on the "
                "empirical-Bayes innovation)",
                "",
                "Geometric mean over seeds of the paired frame-KL ratio at each "
                "configuration's best memory, percentile-bootstrap 95% interval; < 1 means "
                "the default with C-013 is better. These streams have zero-mean gradients, "
                "so the change should cost little here.",
                "",
                *centering_effect(rows, summary),
                "",
            ]
        gates[f"{suite}_k4_all_wins_best"] = all(
            r["best"].get("gimbal_k4_wins_all", False) for r in summary
        )
        # Holm-adjusted over all five peers, so conservative for this subset
        beats = {
            fmt_cell(r): [q for q in PRACTICAL if r["best"]["holm_p_k4"].get(q, 1.0) >= 0.05]
            for r in summary
        }
        gates[f"{suite}_k4_beats_practical_all"] = all(not v for v in beats.values())
        lost = {
            fmt_cell(r): [q for q in PEERS if r["best"]["holm_p_k4"].get(q, 1.0) >= 0.05]
            for r in summary
        }
        report += [
            f"Default (frame_every = 4) below SOAP, SOAP real-time and KL-SOAP (Holm "
            f"p < 0.05) in every cell of suite `{suite}`: "
            f"**{gates[f'{suite}_k4_beats_practical_all']}**. Cells where some peer or "
            "exact-eigenvector control is not significantly worse: "
            + ("; ".join(f"{c} ({', '.join(v)})" for c, v in lost.items() if v) or "none")
            + ".",
            "",
        ]
        if suite == "main":
            sep = [r for r in summary if r["gamma"] == 0.0]
            nonsep = [r for r in summary if r["gamma"] > 0.0]
            # the default configuration since C-012 is frame_every = 4 ("gimbal_k4")
            gates["G2.1_k4_nonseparable_all_wins_best"] = all(
                r["best"]["gimbal_k4_wins_all"] for r in nonsep
            )
            gates["G2.1_k4_nonseparable_all_wins_matched"] = all(
                r["matched"].get("gimbal_k4_wins_all", False) for r in nonsep
            )
            worst_k4 = max(
                r["best"]["mean_kl"]["gimbal_k4"] / min(r["best"]["mean_kl"][p] for p in PEERS)
                for r in sep
            )
            upper_k4 = max(separable_upper_ratio(rows, r, "gimbal_k4") for r in sep)
            gates["G2.1_k4_separable_worst_ratio_vs_best_peer"] = worst_k4
            gates["G2.1_k4_separable_worst_upper_ratio"] = upper_k4
            gates["G2.1_k4_pass"] = bool(
                gates["G2.1_k4_nonseparable_all_wins_best"]
                and worst_k4 <= 1.25
                and upper_k4 <= 1.25
            )
            gates["G2.1_nonseparable_all_wins_best"] = all(
                r["best"]["gimbal_wins_all"] for r in nonsep
            )
            gates["G2.1_nonseparable_all_wins_matched"] = all(
                r["matched"]["gimbal_wins_all"] for r in nonsep
            )
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
            report += [
                "## Random-effects pooling across main-suite cells",
                "",
                "Paired log-ratio log(KL_peer / KL_Gimbal) per seed at best memories; "
                "DerSimonian–Laird pooling of the per-cell means (cells as studies).",
                "",
            ]
            for suffix, title in (("_k4", "default (frame_every = 4)"), ("", "frame_every = 1")):
                pooled = random_effects(summary, suffix)
                report += [
                    f"Gimbal {title}:",
                    "",
                    "| peer | pooled factor | 95% CI | I² | cells |",
                    "|---|---|---|---|---|",
                ]
                for peer, v in pooled.items():
                    report.append(
                        f"| {peer} | {v['factor']:.2f} | [{v['lo']:.2f}, "
                        f"{v['hi']:.2f}] | {v['i2']:.2f} | {v['k']} |"
                    )
                report.append("")
                gates[f"random_effects_factor_vs_gimbal{suffix}"] = {
                    k: v["factor"] for k, v in pooled.items()
                }
        if suite == "tie":
            gates["G2.2_tie_all_wins"] = all(r["best"]["gimbal_wins_all"] for r in summary)
            report += [
                "### Tie suite: mean frame KL by memory (longer memory to the right for "
                "EMA methods, to the left for Gimbal)",
                "",
                memory_table(summary, ["gimbal", "soap", "pooled_eigh", "klsoap"]),
                "",
            ]
        if suite == "drift":
            gates["G2.3_drift_all_wins"] = all(r["best"]["gimbal_wins_all"] for r in summary)
        if suite == "tails":
            gates["G2.4h_tails_all_wins"] = all(r["best"]["gimbal_wins_all"] for r in summary)
    report += ["## Gate items", "", "```", json.dumps(gates, indent=1), "```"]
    (RESULTS / f"e21_report{SUFFIX}.md").write_text("\n".join(report))
    (RESULTS / f"e21_gate{SUFFIX}.json").write_text(json.dumps(gates, indent=1))
    print("\n".join(report[-6:]))


if __name__ == "__main__":
    main()
