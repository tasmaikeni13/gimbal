"""Phase 08: every number and figure of the confirmatory comparison, from the raw run logs.

One command (G8.1): ``python analysis/phase08.py``. Reads ``runs/main/<optimizer>/seed<k>/``
(``config.json``, ``log.jsonl``, ``eval.jsonl``, ``diag.jsonl``, ``final.json``,
``val_seq_losses.npy``) and writes ``analysis/results/phase08.json``, ``analysis/results/
phase08_tables.md`` and figures. Definitions are those fixed before the runs (D-005); the
decision rule is applied exactly as written in ``phases/08_analysis_and_decision.md``. The test
split is not read here (``scripts/tpu/eval_test.py`` runs once, after the decision).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "results"
OPTS = ("adamw", "soap", "gimbal")
COMPETITORS = ("adamw", "soap")
SEEDS = (2, 3)
BLOCK = 512  # evaluation batch (sequences): bootstrap unit
BOOT = 10_000
SEQ_LEN = 1024


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] \
        if path.exists() else []


def load(root: pathlib.Path) -> dict:
    runs = {}
    for opt in OPTS:
        for seed in SEEDS:
            d = root / opt / f"seed{seed}"
            if not (d / "final.json").exists():
                continue
            runs[(opt, seed)] = {
                "config": json.loads((d / "config.json").read_text()),
                "final": json.loads((d / "final.json").read_text()),
                "log": read_jsonl(d / "log.jsonl"),
                "eval": [r for r in read_jsonl(d / "eval.jsonl") if "val_loss_subset" in r],
                "diag": read_jsonl(d / "diag.jsonl"),
                "seq": np.load(d / "val_seq_losses.npy").astype(np.float64)
                if (d / "val_seq_losses.npy").exists() else None,
                "probe": json.loads((d / "frame_probe.json").read_text())
                if (d / "frame_probe.json").exists() else None,
            }
    return runs


def block_means(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Loss sums and sequence counts per block of BLOCK consecutive sequences."""
    starts = np.arange(0, len(seq), BLOCK)
    return np.add.reduceat(seq, starts), np.diff(np.append(starts, len(seq)))


def bootstrap_se(seq: np.ndarray, rng: np.random.Generator) -> float:
    sums, counts = block_means(seq)
    idx = rng.integers(0, len(sums), size=(BOOT, len(sums)))
    est = sums[idx].sum(1) / (counts[idx].sum(1) * SEQ_LEN)
    return float(est.std(ddof=1))


def paired_bootstrap_se(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    sa, counts = block_means(a)
    sb, _ = block_means(b)
    idx = rng.integers(0, len(sa), size=(BOOT, len(sa)))
    est = (sa[idx].sum(1) - sb[idx].sum(1)) / (counts[idx].sum(1) * SEQ_LEN)
    return float(est.std(ddof=1))


def steady_state(log: list[dict], lo: int = 200, block: int = 20) -> float:
    t = {r["step"]: r["step_time"] for r in log}
    hi = max(t) + 1
    means = [np.mean([t[s] for s in range(b, b + block)]) for b in range(lo, hi - block + 1, block)
             if all(s in t for s in range(b, b + block))]
    return float(np.median(means))


def crossing(evals: list[dict], target: float) -> float | None:
    """First step at which the evaluation curve reaches ``target`` (linear interpolation)."""
    pts = sorted((r["step"], r["val_loss_subset"]) for r in evals)
    prev = None
    for step, loss in pts:
        if loss <= target:
            if prev is None:
                return float(step)
            s0, l0 = prev
            return float(s0 + (l0 - target) / (l0 - loss) * (step - s0))
        prev = (step, loss)
    return None


def spikes(log: list[dict], start: int = 500, window: int = 100, jump: float = 0.5) -> int:
    loss = np.array([r["loss"] for r in log])
    count = 0
    for i in range(max(start, window), len(loss)):
        if loss[i] > np.median(loss[i - window:i]) + jump:
            count += 1
    return count


DIAG_KEYS = ("kappa_noise_corrected_median", "kappa_raw_median", "shrink_c_median",
             "kappa_D_mean")


def mechanism(r: dict) -> dict:
    """D-005 item 8: κ of Gimbal's flow variances and the shrinkage factor (median over the
    matrices of a bucket) at the last diagnostic step and their median over training after step
    1,000; the largest orthogonality defect of the frames; the held-out frame quality of the final
    frames (``frame_probe.json``) as differences of L, i.e. of the frame KL (negative = better)."""
    out: dict = {}
    for b in ("attn", "mlp"):
        for k in DIAG_KEYS:
            pts = [(x["step"], x[f"{b}/{k}"]) for x in r["diag"] if f"{b}/{k}" in x]
            if pts:
                out[f"{b}/{k}/final"] = pts[-1][1]
                out[f"{b}/{k}/median_after_1000"] = float(np.median(
                    [v for s, v in pts if s >= 1000] or [pts[-1][1]]))
        for side in ("QL", "QR"):
            vals = [x[f"{b}/orth_defect_{side}"] for x in r["diag"]
                    if f"{b}/orth_defect_{side}" in x]
            if vals:
                out[f"{b}/orth_defect_{side}/max"] = max(vals)
    if r["probe"]:
        for label, m in r["probe"]["matrices"].items():
            out[f"probe/{label}/own_minus_pooled"] = m["L_own_frame"] - m["L_pooled_fit_frame"]
            out[f"probe/{label}/own_minus_identity"] = m["L_own_frame"] - m["L_identity"]
            out[f"probe/{label}/kappa_pooled_frame"] = m["kappa_pooled_frame"]
    return out


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "main"))
    parser.add_argument("--out", default=str(OUT), help="output directory (dry runs)")
    args = parser.parse_args()
    OUT = pathlib.Path(args.out)
    runs = load(pathlib.Path(args.root))
    rng = np.random.default_rng(20261006)
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"runs": {}, "paired": {}, "decision": {}}
    for (opt, seed), r in sorted(runs.items()):
        fin, log = r["final"], r["log"]
        row = {"diverged": bool(fin.get("diverged")), "val_loss": fin.get("val_loss"),
               "val_ppl": math.exp(fin["val_loss"]) if fin.get("val_loss") else None,
               "train_seconds": fin.get("train_seconds"),
               "steady_state_step_s": steady_state(log) if log else None,
               "spikes": spikes(log) if log else None,
               "max_grad_norm_after_500": max((x["grad_norm"] for x in log if x["step"] >= 500),
                                              default=None),
               "final_eval_subset": r["eval"][-1]["val_loss_subset"] if r["eval"] else None,
               "eval_seconds": sum(x.get("eval_seconds", 0) for x in r["eval"]),
               "diag_seconds": sum(x.get("diag_seconds", 0) for x in r["diag"]),
               "checkpoint_seconds": sum(x.get("checkpoint_seconds", 0) for x in r["diag"]),
               "tokens": fin.get("steps_done", 0) * r["config"]["tokens_per_step"]
               if fin.get("steps_done") else None}
        if r["seq"] is not None:
            row["bootstrap_se"] = bootstrap_se(r["seq"], rng)
        row["mechanism"] = mechanism(r)
        res["runs"][f"{opt}/seed{seed}"] = row
    ses = [v["bootstrap_se"] for v in res["runs"].values() if "bootstrap_se" in v]
    pooled_se = float(np.sqrt(np.mean(np.square(ses)))) if ses else None
    res["pooled_within_run_se"] = pooled_se
    seed_sd = {}
    for opt in OPTS:
        vals = [res["runs"].get(f"{opt}/seed{s}", {}).get("val_loss") for s in SEEDS]
        if all(v is not None for v in vals):
            seed_sd[opt] = abs(vals[0] - vals[1]) / math.sqrt(2)
    res["seed_sd"] = seed_sd
    res["pooled_seed_sd"] = float(np.sqrt(np.mean(np.square(list(seed_sd.values()))))) \
        if seed_sd else None
    # Paired comparisons of Gimbal with each competitor (unit: run; pairing: seed).
    for c in COMPETITORS:
        per_seed = {}
        for s in SEEDS:
            g, o = runs.get(("gimbal", s)), runs.get((c, s))
            if not g or not o or g["final"].get("diverged") or o["final"].get("diverged"):
                continue
            d = g["final"]["val_loss"] - o["final"]["val_loss"]
            entry = {"gimbal_minus": d}
            if g["seq"] is not None and o["seq"] is not None:
                entry["paired_bootstrap_se"] = paired_bootstrap_se(g["seq"], o["seq"], rng)
            target = o["eval"][-1]["val_loss_subset"]
            step = crossing(g["eval"], target)
            tps = g["config"]["tokens_per_step"]
            entry.update({
                "target_subset_loss": target,
                "gimbal_steps_to_target": step,
                "gimbal_tokens_to_target": step * tps if step is not None else None,
                "token_multiplier": (o["final"]["steps_done"] * tps / (step * tps))
                if step else None,
                "gimbal_wallclock_to_target_s": step * res["runs"][f"gimbal/seed{s}"]
                ["steady_state_step_s"] if step else None,
                "competitor_wallclock_s": o["final"]["steps_done"]
                * res["runs"][f"{c}/seed{s}"]["steady_state_step_s"],
            })
            per_seed[s] = entry
        if len(per_seed) == len(SEEDS):
            diffs = [per_seed[s]["gimbal_minus"] for s in SEEDS]
            mean = float(np.mean(diffs))
            loss_win = bool(all(d < 0 for d in diffs) and pooled_se is not None
                            and -mean > 2 * pooled_se)
            g_step = np.mean([res["runs"][f"gimbal/seed{s}"]["steady_state_step_s"]
                              for s in SEEDS])
            c_step = np.mean([res["runs"][f"{c}/seed{s}"]["steady_state_step_s"] for s in SEEDS])
            wall = [per_seed[s]["gimbal_wallclock_to_target_s"] for s in SEEDS]
            comp = [per_seed[s]["competitor_wallclock_s"] for s in SEEDS]
            wall_win = bool(g_step < c_step or (all(w is not None for w in wall)
                                                and np.mean(wall) < np.mean(comp)))
            res["paired"][c] = {"per_seed": per_seed, "mean_diff": mean,
                                "two_pooled_se": 2 * pooled_se if pooled_se else None,
                                "beats_on_loss": loss_win,
                                "gimbal_step_s": float(g_step), "competitor_step_s": float(c_step),
                                "beats_on_wallclock": wall_win}
    if all(c in res["paired"] for c in COMPETITORS):
        headline = bool(all(res["paired"][c]["beats_on_loss"] for c in COMPETITORS)
                        and res["paired"]["soap"]["gimbal_step_s"]
                        <= res["paired"]["soap"]["competitor_step_s"])
        res["decision"] = {
            "beats_on_loss": {c: res["paired"][c]["beats_on_loss"] for c in COMPETITORS},
            "beats_on_wallclock": {c: res["paired"][c]["beats_on_wallclock"]
                                   for c in COMPETITORS},
            "no_slower_than_soap": res["paired"]["soap"]["gimbal_step_s"]
            <= res["paired"]["soap"]["competitor_step_s"],
            "headline_claim_allowed": headline}
    (OUT / "phase08.json").write_text(json.dumps(res, indent=1, default=float))
    tables(res)
    figures(runs)
    print(json.dumps(res.get("decision", {}), indent=1))


def tables(res: dict) -> None:
    lines = ["# Phase 08 tables (generated by `analysis/phase08.py`)", "",
             "| run | final val loss | perplexity | bootstrap SE | steady-state step (ms) | "
             "spikes | max grad norm | train s |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res["runs"].items():
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] else ("diverged" if r["diverged"] else "—")
        ppl = f"{r['val_ppl']:.2f}" if r["val_ppl"] else "—"
        se = f"{r['bootstrap_se']:.4f}" if r.get("bootstrap_se") else "—"
        st = f"{1e3 * r['steady_state_step_s']:.1f}" if r["steady_state_step_s"] else "—"
        gn = f"{r['max_grad_norm_after_500']:.2f}" if r["max_grad_norm_after_500"] else "—"
        lines.append(f"| {name} | {vl} | {ppl} | {se} | {st} | {r['spikes']} | {gn} | "
                     f"{r['train_seconds']:.0f} |")
    lines += ["", f"Pooled within-run bootstrap SE: {res['pooled_within_run_se']}; pooled seed "
              f"SD: {res['pooled_seed_sd']}.", ""]
    for c, p in res["paired"].items():
        lines += [f"## Gimbal vs {c}", "", "| seed | Gimbal − c | paired bootstrap SE | "
                  "tokens to c's final loss | token multiplier | Gimbal wall-clock to target (s) "
                  "| c wall-clock (s) |", "|---|---|---|---|---|---|---|"]
        for s, e in p["per_seed"].items():
            tok = f"{e['gimbal_tokens_to_target'] / 1e9:.3f}B" \
                if e["gimbal_tokens_to_target"] else "not reached"
            mult = f"{e['token_multiplier']:.3f}" if e["token_multiplier"] else "—"
            wall = f"{e['gimbal_wallclock_to_target_s']:.0f}" \
                if e["gimbal_wallclock_to_target_s"] else "—"
            pse = f"{e['paired_bootstrap_se']:.4f}" if e.get("paired_bootstrap_se") else "—"
            lines.append(f"| {s} | {e['gimbal_minus']:+.4f} | {pse} | {tok} | {mult} | {wall} | "
                         f"{e['competitor_wallclock_s']:.0f} |")
        lines += ["", f"Mean paired difference {p['mean_diff']:+.4f} (twice the pooled SE: "
                  f"{p['two_pooled_se']}); beats on loss: **{p['beats_on_loss']}**; step time "
                  f"Gimbal {1e3 * p['gimbal_step_s']:.1f} ms vs {1e3 * p['competitor_step_s']:.1f}"
                  f" ms; beats on wall-clock: **{p['beats_on_wallclock']}**.", ""]
    mech = {n: r["mechanism"] for n, r in res["runs"].items() if r.get("mechanism")}
    keys = sorted({k for m in mech.values() for k in m})
    if keys:
        lines += ["## Mechanism diagnostics (D-005 item 8)", "",
                  "| quantity | " + " | ".join(mech) + " |", "|---|" + "---|" * len(mech)]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(
                f"{m[k]:.4g}" if k in m else "—" for m in mech.values()) + " |")
        lines.append("")
    if res["decision"]:
        lines += ["## Decision rule", "", "```", json.dumps(res["decision"], indent=1), "```", ""]
    (OUT / "phase08_tables.md").write_text("\n".join(lines))


def figures(runs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"adamw": "#1f77b4", "soap": "#ff7f0e", "gimbal": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (opt, seed), r in sorted(runs.items()):
        tps = r["config"]["tokens_per_step"]
        ev = sorted((x["step"], x["val_loss_subset"]) for x in r["eval"])
        if not ev:
            continue
        steps, loss = zip(*ev, strict=True)
        style = "-" if seed == SEEDS[0] else "--"
        axes[0].plot(np.array(steps) * tps / 1e9, loss, style, color=colors[opt],
                     label=f"{opt} seed {seed}")
        step_s = steady_state(r["log"]) if r["log"] else 0
        axes[1].plot(np.array(steps) * step_s / 60, loss, style, color=colors[opt],
                     label=f"{opt} seed {seed}")
    for ax, xl in zip(axes, ("tokens (billions)", "wall-clock (minutes, steady-state step time "
                                                    "× steps)"), strict=True):
        ax.set_xlabel(xl)
        ax.set_ylabel("validation loss (5.24M-token subset)")
        ax.set_ylim(top=4.0)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=150)
    # Gimbal's mechanism diagnostics over training: the noise-corrected non-separability index
    # of its flow variances (the E3.1 premise statistic, G3.4 threshold 0.05) and the shrinkage
    # factor c of Proposition 5.5 (median over the matrices of each bucket).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for (opt, seed), r in sorted(runs.items()):
        for b, lab, st in (("mlp", "MLP", "-"), ("attn", "attention", ":")):
            for ax, key in zip(axes, ("kappa_noise_corrected_median", "shrink_c_median"),
                               strict=True):
                pts = [(x["step"], x[f"{b}/{key}"]) for x in r["diag"] if f"{b}/{key}" in x]
                if pts:
                    s, k = zip(*pts, strict=True)
                    ax.plot(s, k, st, color=colors[opt],
                            alpha=1.0 if seed == SEEDS[0] else 0.5,
                            label=f"{opt} seed {seed} {lab}")
    axes[0].axhline(0.05, color="grey", lw=0.8, ls="--", label="G3.4 threshold 0.05")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("noise-corrected κ of D (median)")
    axes[1].set_ylabel("shrinkage factor c (median)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "kappa.png", dpi=150)


if __name__ == "__main__":
    main()
