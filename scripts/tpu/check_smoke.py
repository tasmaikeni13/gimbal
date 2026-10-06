"""Evaluate the Phase 05 smoke tests from their logs; writes runs/smoke/smoke.json."""

from __future__ import annotations

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2] / "runs" / "smoke"


def rows(name: str, file: str = "log.jsonl") -> list[dict]:
    path = ROOT / name / file
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main() -> None:
    out = {}
    ov = rows("overfit")
    if ov:
        tail = [r["loss"] for r in ov[-10:]]
        out["1_overfit_1M_tokens"] = {"first_loss": ov[0]["loss"], "last10_mean": sum(tail) / 10,
                                      "pass": sum(tail) / 10 < 0.1}
    zl = rows("zero_lr", "eval.jsonl")
    if len(zl) >= 2:
        out["2_zero_lr"] = {"val_start": zl[0]["val_loss_subset"],
                            "val_end": zl[-1]["val_loss_subset"],
                            "pass": zl[0]["val_loss_subset"] == zl[-1]["val_loss_subset"]}
    a, b = rows("adamw_200a"), rows("adamw_200b")
    if a:
        first = sum(r["loss"] for r in a[:10]) / 10
        last = sum(r["loss"] for r in a[-10:]) / 10
        out["3_adamw_200_steps"] = {"first10": first, "last10": last,
                                    "finite": all(math.isfinite(r["loss"]) for r in a),
                                    "pass": last < first - 2.0
                                    and all(math.isfinite(r["loss"]) for r in a)}
    if a and b:
        diffs = [abs(x["loss"] - y["loss"]) for x, y in zip(a[:50], b[:50], strict=True)]
        out["4_determinism_50_steps"] = {"max_abs_diff": max(diffs), "pass": max(diffs) <= 1e-6}
    for name, start in (("resume_adamw", 20), ("resume_gimbal", 60)):
        r = rows(name)
        first = {x["step"]: x["loss"] for x in r[:len(r) - 20]}
        second = {x["step"]: x["loss"] for x in r[len(r) - 20:]}
        if second:
            d = [abs(first[s] - second[s]) for s in second if s in first]
            out[f"5_{name}_next_20_steps"] = {"resumed_from": start, "compared": len(d),
                                              "max_abs_diff": max(d) if d else None,
                                              "pass": len(d) == 20 and max(d) <= 1e-6}
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    (ROOT / "smoke.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
