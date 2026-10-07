"""Generate the paper's numbers: ``paper/generated/macros.tex`` and table files.

Every number in the paper comes from a result file through this script (Phase 09 rule): Phase 02
Monte Carlo (`experiments/phase2/results`), the small language model (`experiments/phase3/results`),
TPU performance (`benchmarks/tpu/results`), the reference evaluations (`runs/reference`,
`runs/smoke`), tuning (`runs/tuning`) and the confirmatory runs (`analysis/results/phase08.json`).
Missing results produce ``\\pending`` so that the paper builds at every stage.

Usage: python paper/make_generated.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = ROOT / "paper" / "generated"
P2 = ROOT / "experiments" / "phase2" / "results"
P3 = ROOT / "experiments" / "phase3" / "results"
macros: dict[str, str] = {}


def load(path: pathlib.Path):
    if not path.exists():
        return None
    if path.suffix == ".gz":
        return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()
                if x.strip()]
    return json.loads(path.read_text())


def mac(name: str, value, fmt: str = "{:.4f}") -> None:
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        macros[name] = r"\pending"
    elif isinstance(value, str):
        macros[name] = value
    else:
        macros[name] = fmt.format(value)


def phase02() -> None:
    e21 = load(P2 / "e21_gate.json") or {}
    re_k4 = e21.get("random_effects_factor_vs_gimbal_k4", {})
    mac("FactorSoap", re_k4.get("soap"), "{:.1f}")
    mac("FactorKlsoap", re_k4.get("klsoap"), "{:.1f}")
    mac("FactorSoapRt", re_k4.get("soap_rt"), "{:.1f}")
    mac("SepWorstRatio", e21.get("G2.1_k4_separable_worst_ratio_vs_best_peer"), "{:.3f}")
    mac("SepWorstUpper", e21.get("G2.1_k4_separable_worst_upper_ratio"), "{:.3f}")
    e25 = load(P2 / "e25_gate_full_c018.json")
    rows = []
    if e25:
        for key, c in e25["configs"].items():
            g = re.match(r"gamma=([\d.]+),noise=([\d.]+)", key)
            vs = c["vs_peer"]
            rows.append(f"{g.group(1)} & {g.group(2)} & {c['mean']:.4f} & "
                        f"{vs['adamw']['geo_ratio']:.3f} & {vs['soap']['geo_ratio']:.3f} & "
                        f"{vs['soap_rt']['geo_ratio']:.3f} & {vs['klsoap']['geo_ratio']:.3f} & "
                        f"{vs['muon']['geo_ratio']:.3f} & {'yes' if c['pass'] else 'no'} \\\\")
        mac("GtwoFourPass", "passes" if e25.get("G2.4") else "fails")
    (GEN / "e25_table.tex").write_text(
        "\\begin{tabular}{ccccccccc}\n\\toprule\n$\\gamma$ & $\\sigma$ & Gimbal loss & "
        "/AdamW & /SOAP & /SOAP-rt & /KL-SOAP & /Muon & lowest \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    e211 = (P2 / "e211_report.md").read_text() if (P2 / "e211_report.md").exists() else ""
    starts = glob_min = 0
    for line in e211.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 8 and cells[2].isdigit():
            starts += sum(int(c) for c in cells[2:7])
            glob_min += int(cells[2])
    mac("LandscapeStarts", f"{starts:,}" if starts else None)
    mac("LandscapeGlobal", f"{glob_min:,}" if starts else None)


def phase03() -> None:
    lm = P3 / "lm"

    def final(name: str):
        p = lm / f"{name}.summary.json"
        return json.loads(p.read_text())["final_val_loss"] if p.exists() else None

    # E3.2: selected learning rate per optimizer (Stage A, seed 0), seeds 0-2.
    gate = load(P3 / "e27_gate.json") or {}
    report = (P3 / "e27_report.md").read_text() if (P3 / "e27_report.md").exists() else ""
    rows = []
    for line in report.splitlines():
        m = re.match(r"\| (\w+) \| ([\d.e-]+) \| 3 \| ([\d.]+) \| ([\d., ]+) \|", line)
        if m:
            rows.append(f"{m.group(1).replace('_', '-')} & {float(m.group(2)):g} & "
                        f"{m.group(3)} & {m.group(4)} \\\\")
    (GEN / "e32_table.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\noptimizer & lr & mean & seeds 0, 1, 2 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    g35 = load(P3 / "g35_c018.json")
    if g35:
        for opt, name in (("gimbal", "Gimbal"), ("soap", "Soap"), ("adamw", "Adamw")):
            mac(f"GthreeFive{name}", float(np.mean(g35["final_val_loss"][opt])))
        d = g35["gimbal_minus_peer"]["soap"]
        mac("GthreeFiveDiffSoap", float(np.mean(d)), "{:+.4f}")
        mac("GthreeFiveDiffSoapMin", float(max(d)), "{:+.4f}")
        mac("GthreeFiveP", g35["holm_p_one_sided"]["soap"], "{:.3f}")
    kill = {}
    for m_ in ("gimbal", "soap", "soap_rt", "klsoap", "adamw"):
        lr = 0.001 if m_ == "adamw" else 0.004
        vals = [final(f"{m_}_lr{lr:g}_s{s}_c018") for s in (20, 21, 22)]
        kill[m_] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    mac("KillKlsoap", kill["klsoap"])
    mac("KillSoapRt", kill["soap_rt"])
    e26 = load(P3 / "e26_real_gradients.json")
    if e26:
        kap = np.array([r["kappa_pooled_frame"] for r in e26])
        gain = np.array([r["gain_gimbal_vs_soap_nats"] for r in e26])
        gkl = np.array([r["gain_gimbal_vs_kl_nats"] for r in e26])
        mac("KappaMedian", float(np.median(kap)), "{:.3f}")
        mac("KappaMax", float(kap.max()), "{:.3f}")
        mac("EThreeOneMatrices", str(len(e26)))
        mac("EThreeOneWinsSoap", str(int((gain > 0).sum())))
        mac("EThreeOneGainSoap", float(np.median(gain)), "{:,.0f}")
        mac("EThreeOneWinsKl", str(int((gkl > 0).sum())))
        mac("EThreeOneGainKl", float(np.median(gkl)), "{:,.0f}")
    del gate
    e25j = load(P3 / "e25_jax_gate.json")
    if e25j:
        worst = max(abs(c["mean_diff"]) / c["seed_sd"] for c in e25j["checks"].values()
                    if c["seed_sd"] > 0)
        mac("JaxEtwoFiveWorst", worst, "{:.3f}")


def phase04_05() -> None:
    st = load(ROOT / "benchmarks" / "tpu" / "results" / "step_times.json") or {}

    def ms(k):
        return 1e3 * st[k]["steady_state_s"] if k in st else None

    mac("StepAdamw", ms("adamw"), "{:.1f}")
    mac("StepSoap", ms("soap"), "{:.1f}")
    mac("StepGimbal", ms("gimbal_adapt_a05"), "{:.1f}")
    mac("StepGimbalKtwo", ms("gimbal_k2_a05"), "{:.1f}")
    mac("StepGimbalKone", ms("gimbal_k1_a05"), "{:.1f}")
    if "soap" in st and "gimbal_adapt_a05" in st:
        mac("StepGimbalOverSoap", 100 * (st["gimbal_adapt_a05"]["steady_state_s"]
                                         / st["soap"]["steady_state_s"] - 1), "{:.1f}")
        mac("StepKoneOverSoap", 100 * (st["gimbal_k1_a05"]["steady_state_s"]
                                       / st["soap"]["steady_state_s"] - 1), "{:.1f}")
        mac("StepKtwoOverSoap", 100 * (st["gimbal_k2_a05"]["steady_state_s"]
                                       / st["soap"]["steady_state_s"] - 1), "{:.1f}")
    att = load(ROOT / "benchmarks" / "tpu" / "results" / "attention.json") or {}
    mac("AttnXla", att.get("xla_q0_k0_b1"), "{:.3f}")
    mac("AttnFlashDefault", att.get("flash_q0_k0_b1"), "{:.3f}")
    mac("AttnFlashTuned", att.get("flash_q512_k512_b1"), "{:.3f}")
    ref = load(ROOT / "runs" / "reference" / "gpt2_on_val.json")
    if ref:
        mac("GptTwoOurVal", ref["val_loss"], "{:.3f}")
        mac("GptTwoPublished", ref["published_reference"]["value"], "{:.4f}")
    a1k = load(ROOT / "runs" / "smoke" / "adamw_1000" / "final.json")
    mac("AdamwOneK", a1k.get("val_loss") if a1k else None, "{:.2f}")


def tuning() -> None:
    sel = load(ROOT / "runs" / "tuning" / "selection.json") or {}
    knob = {"adamw": "b2", "soap": "shampoo_beta", "gimbal": "rot_rate"}
    for opt, name in (("adamw", "Adamw"), ("soap", "Soap"), ("gimbal", "Gimbal")):
        s = sel.get(opt)
        mac(f"Lr{name}", s["lr"] if s else None, "{:.3g}")
        mac(f"Knob{name}", s[knob[opt]] if s else None, "{:g}")


SEED_WORD = {2: "Two", 3: "Three"}
NAMES = {"adamw": "Adamw", "soap": "Soap", "gimbal": "Gimbal"}


def _mean(vals):
    return float(np.mean(vals)) if vals and all(v is not None for v in vals) else None


def main_runs(results: pathlib.Path, main_dir: pathlib.Path) -> None:
    """Confirmatory runs: macros and tables from ``phase08.json``, ``compute.json`` and the
    one-time test evaluations (``runs/main/<optimizer>/seed<k>/test.json``)."""
    res = load(results / "phase08.json") or {
        "runs": {}, "paired": {}, "decision": {}, "pooled_within_run_se": None}
    comp = load(results / "compute.json") or {}
    seeds = (2, 3)
    rows = []
    for opt, name in NAMES.items():
        runs = [res["runs"].get(f"{opt}/seed{s}", {}) for s in seeds]
        tests = [load(main_dir / opt / f"seed{s}" / "test.json") for s in seeds]
        vals = [r.get("val_loss") for r in runs]
        mac(f"Val{name}", _mean(vals))
        mac(f"Ppl{name}", math.exp(_mean(vals)) if _mean(vals) else None, "{:.2f}")
        for s, v in zip(seeds, vals, strict=True):
            mac(f"Val{name}Seed{SEED_WORD[s]}", v)
        tl = [t["test_loss"] if t else None for t in tests]
        mac(f"Test{name}", _mean(tl))
        mac(f"TestPpl{name}", math.exp(_mean(tl)) if _mean(tl) else None, "{:.2f}")
        steps = [r.get("steady_state_step_s") for r in runs]
        mac(f"MainStep{name}", 1e3 * _mean(steps) if _mean(steps) else None, "{:.1f}")
        mac(f"Spikes{name}", sum(r.get("spikes", 0) or 0 for r in runs) if all(runs) else None,
            "{:d}")
        mac(f"ChipHours{name}", comp.get("total_chip_hours", {}).get(opt), "{:.0f}")
        mac(f"TuneChipHours{name}", comp.get("stages", {}).get("tuning", {}).get(opt, {})
            .get("chip_hours"), "{:.0f}")
        mac(f"MainChipHours{name}", comp.get("stages", {}).get("main", {}).get(opt, {})
            .get("chip_hours"), "{:.0f}")
        for s, r, t in zip(seeds, runs, tests, strict=True):
            if not r:
                continue
            vl = "diverged" if r.get("diverged") else f"{r['val_loss']:.4f}"
            se = f"{r['bootstrap_se']:.4f}" if r.get("bootstrap_se") else "--"
            ppl = f"{r['val_ppl']:.2f}" if r.get("val_ppl") else "--"
            tst = f"{t['test_loss']:.4f}" if t else "--"
            st = f"{1e3 * r['steady_state_step_s']:.1f}" if r.get("steady_state_step_s") else "--"
            gn = f"{r['max_grad_norm_after_500']:.2f}" if r.get("max_grad_norm_after_500") \
                else "--"
            rows.append(f"{opt} & {s} & {vl} & {se} & {ppl} & {tst} & {st} & "
                        f"{r.get('spikes', '--')} & {gn} \\\\")
    (GEN / "main_table.tex").write_text(
        "\\begin{tabular}{lcccccccc}\n\\toprule\noptimizer & seed & val.\\ loss & SE & "
        "val.\\ ppl & test loss & step (ms) & spikes & max $\\|g\\|$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    mac("PooledSe", res.get("pooled_within_run_se"), "{:.4f}")
    mac("TwoPooledSe", 2 * res["pooled_within_run_se"] if res.get("pooled_within_run_se")
        else None, "{:.4f}")
    mac("PooledSeedSd", res.get("pooled_seed_sd"), "{:.4f}")
    prow = []
    for c, name in (("adamw", "Adamw"), ("soap", "Soap")):
        p = res["paired"].get(c)
        per = p["per_seed"] if p else {}
        mac(f"Diff{name}", p["mean_diff"] if p else None, "{:+.4f}")
        for s in seeds:
            e = per.get(str(s)) or per.get(s)
            mac(f"Diff{name}Seed{SEED_WORD[s]}", e["gimbal_minus"] if e else None, "{:+.4f}")
        pse = [e.get("paired_bootstrap_se") for e in per.values()]
        mac(f"PairedSe{name}", _mean(pse) if pse else None, "{:.4f}")
        mac(f"BeatsLoss{name}", ("yes" if p["beats_on_loss"] else "no") if p else None)
        mac(f"BeatsWall{name}", ("yes" if p["beats_on_wallclock"] else "no") if p else None)
        mult = [e["token_multiplier"] for e in per.values()]
        mac(f"Mult{name}", _mean(mult) if mult else None, "{:.3f}")
        tok = [e["gimbal_tokens_to_target"] for e in per.values()]
        mac(f"TokensTo{name}", _mean(tok) / 1e9 if mult and _mean(tok) else None, "{:.2f}")
        wall = [e["gimbal_wallclock_to_target_s"] for e in per.values()]
        cw = [e["competitor_wallclock_s"] for e in per.values()]
        mac(f"WallTo{name}", _mean(wall) / 60 if wall and _mean(wall) else None, "{:.1f}")
        mac(f"CompWall{name}", _mean(cw) / 60 if cw and _mean(cw) else None, "{:.1f}")
        for s in seeds:
            e = per.get(str(s)) or per.get(s)
            if not e:
                continue
            tk = f"{e['gimbal_tokens_to_target'] / 1e9:.2f}" if e["gimbal_tokens_to_target"] \
                else "not reached"
            mu = f"{e['token_multiplier']:.3f}" if e["token_multiplier"] else "--"
            wl = f"{e['gimbal_wallclock_to_target_s'] / 60:.1f}" \
                if e["gimbal_wallclock_to_target_s"] else "--"
            ps = f"{e['paired_bootstrap_se']:.4f}" if e.get("paired_bootstrap_se") else "--"
            prow.append(f"{c} & {s} & {e['gimbal_minus']:+.4f} & {ps} & {tk} & {mu} & {wl} & "
                        f"{e['competitor_wallclock_s'] / 60:.1f} \\\\")
    (GEN / "paired_table.tex").write_text(
        "\\begin{tabular}{lccccccc}\n\\toprule\n$c$ & seed & Gimbal $-$ $c$ & paired SE & "
        "tokens to $c$'s loss (B) & multiplier & Gimbal min & $c$ min \\\\\n\\midrule\n"
        + "\n".join(prow) + "\n\\bottomrule\n\\end{tabular}\n")
    dec = res.get("decision", {})
    mac("Headline", ("allowed" if dec.get("headline_claim_allowed") else "not allowed")
        if dec else None)
    mac("NoSlowerThanSoap", ("yes" if dec.get("no_slower_than_soap") else "no") if dec else None)
    gs = [res["runs"].get(f"gimbal/seed{s}", {}).get("steady_state_step_s") for s in seeds]
    ss = [res["runs"].get(f"soap/seed{s}", {}).get("steady_state_step_s") for s in seeds]
    mac("MainStepGimbalOverSoap", 100 * (_mean(gs) / _mean(ss) - 1)
        if _mean(gs) and _mean(ss) else None, "{:.1f}")
    # Mechanism diagnostics (D-005 item 8), medians over Gimbal's two runs.
    for b, bn in (("mlp", "Mlp"), ("attn", "Attn")):
        for key, mn, fmt in (("kappa_noise_corrected_median", "KappaMain", "{:.4f}"),
                             ("kappa_raw_median", "KappaRawMain", "{:.4f}"),
                             ("shrink_c_median", "ShrinkMain", "{:.2f}")):
            vals = [res["runs"].get(f"gimbal/seed{s}", {}).get("mechanism", {})
                    .get(f"{b}/{key}/final") for s in seeds]
            mac(f"{mn}{bn}", _mean(vals), fmt)
    for opt, name in (("soap", "Soap"), ("gimbal", "Gimbal")):
        own = [v for s in seeds for k, v in res["runs"].get(f"{opt}/seed{s}", {})
               .get("mechanism", {}).items() if k.endswith("/own_minus_pooled")]
        mac(f"ProbeOwnMinusPooled{name}", float(np.median(own)) if own else None, "{:,.0f}")
        mac(f"ProbeOwnBetter{name}", f"{sum(v < 0 for v in own)}/{len(own)}" if own else None)
    kp = [v for s in seeds for o in ("soap", "gimbal") for k, v in res["runs"]
          .get(f"{o}/seed{s}", {}).get("mechanism", {}).items()
          if k.endswith("/kappa_pooled_frame")]
    mac("ProbeKappa", float(np.median(kp)) if kp else None, "{:.4f}")


def tex(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")
            .replace("%", "\\%").replace("#", "\\#"))


def lean_table() -> None:
    """Theorem map of formal/README.md (statement, Lean names, file)."""
    rows = []
    for line in (ROOT / "formal" / "README.md").read_text().splitlines():
        m = re.match(r"\| (.+?) \| (.+?) \| `(.+?)` \| (.+?) \|$", line)
        if m and m.group(1) != "Theory item":
            item, names, file, _ = m.groups()
            names = ", ".join("\\texttt{" + tex(n.strip(" `")) + "}" for n in names.split(","))
            rows.append(f"{tex(item)} & {names} & \\texttt{{{tex(file)}}} \\\\")
    (GEN / "lean_table.tex").write_text(
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.6cm}"
        ">{\\raggedright\\arraybackslash}p{9.0cm}>{\\raggedright\\arraybackslash}p{3.2cm}}"
        "\n\\toprule\nstatement & Lean names & file "
        "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    mac("LeanCount", str(sum(1 for line in (ROOT / "formal" / "audit" / "axioms_output.txt")
                             .read_text().splitlines() if "depends on axioms" in line)))


def tuning_table() -> None:
    """Every Phase 06 trial (from runs/tuning/*/final.json)."""
    rows = []
    pat = re.compile(r"^(adamw|soap|gimbal)_lr([0-9.e-]+)(?:_(b2|shampoo_beta|rot_rate)"
                     r"([0-9.]+))?_s(\d+)$")
    runs = ROOT / "runs" / "tuning"
    if runs.exists():
        for d in sorted(runs.iterdir()):
            m = pat.match(d.name)
            if not m or not (d / "final.json").exists():
                continue
            fin = json.loads((d / "final.json").read_text())
            opt, lr, key, val, seed = m.groups()
            loss = "diverged" if fin.get("diverged") else f"{fin.get('val_loss', float('nan')):.4f}"
            knob = f"{tex(key)}={val}" if key else "default"
            rows.append(f"{opt} & {float(lr):.4g} & {knob} & {seed} & {loss} \\\\")
    (GEN / "tuning_table.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\noptimizer & peak lr & secondary & seed & "
        "final validation loss \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    global GEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "analysis" / "results"))
    parser.add_argument("--main-runs", default=str(ROOT / "runs" / "main"))
    parser.add_argument("--out", default=str(GEN), help="output directory (dry runs)")
    args = parser.parse_args()
    GEN = pathlib.Path(args.out)
    GEN.mkdir(parents=True, exist_ok=True)
    lean_table()
    tuning_table()
    phase02()
    phase03()
    phase04_05()
    tuning()
    main_runs(pathlib.Path(args.results), pathlib.Path(args.main_runs))
    lines = ["% generated by paper/make_generated.py; do not edit",
             "\\providecommand{\\pending}{\\textbf{[pending]}}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros")


if __name__ == "__main__":
    main()
