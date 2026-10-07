"""One-time evaluation of the confirmatory runs on the held-out test split (Phase 08).

The test split may be read only after the decision rule has been applied to the validation
results (`phases/README.md` §2): the script refuses to run unless ``analysis/decision.md`` exists
and records the decision. It loads each run's final checkpoint, evaluates the 20M-token test split
and writes ``runs/main/<optimizer>/seed<k>/test.json`` and ``test_seq_losses.npy``.

Run on all hosts: scripts/tpu/launch.sh <logdir> python scripts/tpu/eval_test.py
"""

from __future__ import annotations

import json
import os
import pathlib

import jax
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    decision = ROOT / "analysis" / "decision.md"
    if not decision.exists() or "Decision rule outcome" not in decision.read_text():
        raise SystemExit("analysis/decision.md does not record the decision: the test split "
                         "stays closed")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from gimbal.train import checkpoint
    from gimbal.train.data import TokenFile
    from gimbal.train.train import Trainer

    writer = os.environ.get("GIMBAL_WORKER", "0") == "0"
    test = TokenFile(ROOT / "data" / "tokens" / "test.bin")
    for opt in ("adamw", "soap", "gimbal"):
        for seed in (2, 3):
            run = ROOT / "runs" / "main" / opt / f"seed{seed}"
            meta = json.loads((run / "config.json").read_text())
            cfg = meta["config"]
            frozen = yaml.safe_load((ROOT / "configs" / "frozen" / f"{opt}.yaml").read_text())
            cfg["optimizers"][opt].update(frozen["optimizer"])
            trainer = Trainer(cfg, opt)
            step = checkpoint.latest(run / "ckpt")
            params, _ = checkpoint.restore(run / "ckpt", step, trainer, seed)
            losses = trainer.evaluate(params, test, test.n_seq)
            if writer:
                np.save(run / "test_seq_losses.npy", losses.astype(np.float32))
                out = {"checkpoint_step": step, "test_loss": float(losses.sum() / (
                    len(losses) * test.seq_len)), "test_tokens": int(len(losses) * test.seq_len)}
                out["test_ppl"] = float(np.exp(out["test_loss"]))
                (run / "test.json").write_text(json.dumps(out, indent=1))
                print(opt, seed, out, flush=True)


if __name__ == "__main__":
    main()
