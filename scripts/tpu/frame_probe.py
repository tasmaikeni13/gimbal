"""Mechanism diagnostics at 125M (Phase 08, D-005 item 8): quality of the final frames.

For the final checkpoints of the SOAP and Gimbal confirmatory runs, compute 32 independent
minibatch gradients (240 validation sequences each, at the final weights) of four matrices (the
query projection and the MLP up projection of the first and last layer), split them into a fit
half and a held-out half as in E3.1, and score frames on the held-out half by
``L(U) = 1/2 Σ_ij log mean_k (Q_Lᵀ G_k Q_R)²_ij`` (differences of L are differences of the frame
KL of Proposition 1): the run's own frame, the pooled frame of the fit half (SOAP's estimator
with exact eigenvectors), and the identity (AdamW). Also the non-separability index κ of the
variances in the pooled frame. Writes ``runs/main/<optimizer>/seed<k>/frame_probe.json``.

Run on all hosts after the main runs: scripts/tpu/launch.sh <logdir> python
scripts/tpu/frame_probe.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import jax
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
# (bucket, index in the bucket, label): bucket layouts of gimbal.train.model
PROBES = (("attn", 0, "layer0_q"), ("attn", 44, "layer11_q"),
          ("mlp", 1, "layer0_up"), ("mlp", 34, "layer11_up"))
SAMPLES = 32


def heldout(ql: np.ndarray, qr: np.ndarray, g: np.ndarray) -> float:
    z = np.einsum("ia,kij,jb->kab", ql, g, qr, optimize=True)
    return 0.5 * float(np.sum(np.log((z * z).mean(axis=0) + 1e-300)))


def eigh_desc(s: np.ndarray) -> np.ndarray:
    return np.linalg.eigh(0.5 * (s + s.T))[1][:, ::-1]


def main() -> None:
    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    from jax.experimental import multihost_utils

    from gimbal.train import checkpoint
    from gimbal.train.data import TokenFile, global_batch
    from gimbal.train.train import Trainer

    sys.path.insert(0, str(ROOT / "experiments" / "phase2"))
    from common import nonseparability_index

    writer = os.environ.get("GIMBAL_WORKER", "0") == "0"
    val = TokenFile(ROOT / "data" / "tokens" / "val.bin")
    for opt in ("soap", "gimbal"):
        for seed in (2, 3):
            run = ROOT / "runs" / "main" / opt / f"seed{seed}"
            cfg = json.loads((run / "config.json").read_text())["config"]
            frozen = yaml.safe_load((ROOT / "configs" / "frozen" / f"{opt}.yaml").read_text())
            cfg["optimizers"][opt].update(frozen["optimizer"])
            trainer = Trainer(cfg, opt)
            step = checkpoint.latest(run / "ckpt")
            params, state = checkpoint.restore(run / "ckpt", step, trainer, seed)
            grads = {label: [] for _, _, label in PROBES}
            batch = cfg["train"]["batch"]
            for k in range(SAMPLES):
                idx = np.arange(k * batch, (k + 1) * batch)
                _, g, _ = trainer._grad_fn(params, global_batch(val, idx, trainer.batch_sharding))
                for bucket, i, label in PROBES:
                    full = multihost_utils.process_allgather(g[bucket], tiled=True)
                    grads[label].append(np.asarray(full[i], np.float64))
            out = {"checkpoint_step": step, "samples": SAMPLES, "matrices": {}}
            for bucket, i, label in PROBES:
                st = state["matrix"][bucket]
                ql = np.asarray(multihost_utils.process_allgather(st["QL"], tiled=True)[i],
                                np.float64)
                qr = np.asarray(multihost_utils.process_allgather(st["QR"], tiled=True)[i],
                                np.float64)
                g = np.stack(grads[label])
                fit, ev = g[: SAMPLES // 2], g[SAMPLES // 2:]
                pl = eigh_desc(np.einsum("kij,klj->il", fit, fit, optimize=True))
                pr = eigh_desc(np.einsum("kji,kjl->il", fit, fit, optimize=True))
                z = np.einsum("ia,kij,jb->kab", pl, fit, pr, optimize=True)
                out["matrices"][label] = {
                    "shape": list(g.shape[1:]),
                    "L_own_frame": heldout(ql, qr, ev),
                    "L_pooled_fit_frame": heldout(pl, pr, ev),
                    "L_identity": heldout(np.eye(g.shape[1]), np.eye(g.shape[2]), ev),
                    "kappa_pooled_frame": nonseparability_index((z * z).mean(0) + 1e-300,
                                                                SAMPLES // 2),
                }
            if writer:
                (run / "frame_probe.json").write_text(json.dumps(out, indent=1))
                print(opt, seed, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
