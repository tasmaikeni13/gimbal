"""Training entry point for the TPU runs (Phases 04–07).

Runs the same program on every host of the slice (``scripts/tpu/launch.sh``). One mesh axis
``"data"`` spans all chips: the batch is sharded along it, parameters are replicated, and the
optimizer state is sharded along the bucket axis (:mod:`gimbal.jax.distributed`). A step is two
compiled programs: the gradient (model forward/backward, float32 reduce-scatter, gradient norm),
compiled once per run, and the optimizer update, compiled once per step kind (first step, warm
start, frame move, QR refresh), so the steady state contains no eigendecomposition, QR or branch.

Worker 0 writes ``<run_dir>/{config.json, log.jsonl, eval.jsonl, diag.jsonl, final.json,
val_seq_losses.npy}``. Example::

    scripts/tpu/launch.sh runs/logs/x python -m gimbal.train.train --optimizer gimbal \\
        --lr 3e-3 --seed 0 --steps 2500 --run-dir runs/tuning/gimbal_lr3e-3_s0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import time
from dataclasses import asdict, fields
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from gimbal.jax import distributed as dist
from gimbal.jax.adamw import AdamWConfig
from gimbal.jax.gimbal import GimbalConfig
from gimbal.jax.soap import SOAPConfig

from . import checkpoint
from .data import TokenFile, eval_batches, train_batches, train_order
from .model import ModelConfig, count_params, init_params, loss_fn, token_losses

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = {"adamw": AdamWConfig, "soap": SOAPConfig, "gimbal": GimbalConfig}


def code_version() -> dict:
    """Commit of the code that runs: a snapshot records it (scripts/tpu/snapshot.sh)."""
    snap = ROOT / "SNAPSHOT_COMMIT"
    if snap.exists():
        diff = (ROOT / "SNAPSHOT_DIFF").read_text()
        return {"git": snap.read_text().strip(), "git_dirty": bool(diff.strip()),
                "snapshot": str(ROOT)}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "configs"],
                           capture_output=True, text=True, cwd=ROOT)
    return {"git": git.stdout.strip(), "git_dirty": bool(dirty.stdout.strip())}


def is_writer() -> bool:
    """Worker 0 of the slice writes the run files (JAX numbers processes by chip coordinates)."""
    return os.environ.get("GIMBAL_WORKER", "0") == "0"


def lr_at(step: int, peak: float, total: int, warmup: int, final_frac: float) -> float:
    """Linear warm-up for ``warmup`` steps, then cosine decay to ``final_frac`` of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def build_spec(cfg: dict, name: str) -> tuple[dist.OptimizerSpec, float]:
    hp = dict(cfg["optimizers"][name])
    lr = float(hp.pop("lr"))
    hp["weight_decay"] = cfg["train"]["weight_decay"]
    allowed = {f.name for f in fields(CONFIGS[name])}
    unknown = set(hp) - allowed
    if unknown:
        raise ValueError(f"unknown {name} hyper-parameters {sorted(unknown)}")
    matrix = CONFIGS[name](**hp)
    other = AdamWConfig(weight_decay=0.0, **cfg["other"])
    if name == "adamw":
        # The AdamW baseline is one optimizer: its secondary knob (b2) applies to every parameter.
        other = AdamWConfig(b1=matrix.b1, b2=matrix.b2, eps=matrix.eps, weight_decay=0.0)
    return dist.OptimizerSpec(name, matrix, other), lr


def set_path(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = yaml.safe_load(value)


class Trainer:
    def __init__(self, cfg: dict, optimizer: str) -> None:
        self.cfg = cfg
        self.model_cfg = ModelConfig(**cfg["model"])
        self.spec, self.peak_lr = build_spec(cfg, optimizer)
        devices = np.array(jax.devices())
        self.n_dev = len(devices)
        self.mesh = Mesh(devices, (dist.AXIS,))
        self.batch_sharding = NamedSharding(self.mesh, P(dist.AXIS, None))
        self.replicated = NamedSharding(self.mesh, P())
        shapes = jax.eval_shape(partial(init_params, self.model_cfg), jax.random.PRNGKey(0))
        self.param_shapes = {k: v.shape for k, v in shapes.items()}
        self.buckets = dist.make_buckets(self.param_shapes, self.n_dev)
        self._update_fns: dict = {}
        self._grad_fn = self._make_grad_fn()
        self._eval_fn = self._make_eval_fn()

    # -- placement ------------------------------------------------------------------------
    def init(self, seed: int) -> tuple[dict, dict]:
        """Parameters (replicated) and optimizer state, created directly in their layouts."""
        params = jax.jit(partial(init_params, self.model_cfg),
                         out_shardings=self.replicated)(jax.random.PRNGKey(seed))
        make = partial(dist.init_state, self.spec, self.buckets)
        shard = self.state_shardings(jax.eval_shape(make, params))
        return params, jax.jit(make, out_shardings=shard)(params)

    def init_params_abstract(self) -> dict:
        return jax.eval_shape(partial(init_params, self.model_cfg), jax.random.PRNGKey(0))

    def abstract_state(self, step: int) -> dict:
        """Shapes of the optimizer state after ``step`` steps (Gimbal frees its warm-start
        buffers at the warm-start step)."""
        params = jax.eval_shape(partial(init_params, self.model_cfg), jax.random.PRNGKey(0))
        state = jax.eval_shape(partial(dist.init_state, self.spec, self.buckets), params)
        warm = getattr(self.spec.matrix, "warm_start_steps", 0)
        if self.spec.name == "gimbal" and warm > 1 and step >= warm:
            state = dist.drop_gimbal_warm_buffers(state)
        return state

    def state_shardings(self, state: dict) -> dict:
        return jax.tree.map(lambda s: NamedSharding(self.mesh, s), dist.state_specs(state),
                            is_leaf=lambda x: isinstance(x, P))

    # -- compiled steps -------------------------------------------------------------------
    def _grad_specs(self) -> dict:
        out = {b.key: P(dist.AXIS) for b in self.buckets}
        out["embed"] = P(dist.AXIS)
        for k in dist.OTHER_KEYS[1:]:
            out[k] = P()
        return out

    def _make_grad_fn(self):
        mcfg, buckets, n_dev = self.model_cfg, self.buckets, self.n_dev

        def local(params, batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch, mcfg)
            grads = dist.reduce_gradients(grads, buckets, n_dev)
            return jax.lax.pmean(loss, dist.AXIS), grads, dist.global_norm(grads)

        param_specs = {k: P() for k in self.param_shapes}
        fn = shard_map(local, mesh=self.mesh, in_specs=(param_specs, P(dist.AXIS, None)),
                       out_specs=(P(), self._grad_specs(), P()), check_rep=False)
        return jax.jit(fn)

    def _make_update_fn(self, kind, state: dict):
        spec, buckets, n_dev = self.spec, self.buckets, self.n_dev
        clip = float(self.cfg["train"]["clip"])

        def local(state, grads, params, gnorm, lr, t):
            scale = jnp.minimum(1.0, clip / (gnorm + 1e-6))
            grads = jax.tree.map(lambda g: g * scale, grads)
            return dist.optimizer_update(spec, kind, state, grads, params, buckets, n_dev, lr, t)

        param_specs = {k: P() for k in self.param_shapes}
        in_state = dist.state_specs(state)
        out_state = in_state
        if spec.name == "gimbal" and kind.restart:
            out_state = dist.state_specs(dist.drop_gimbal_warm_buffers(state))
        fn = shard_map(local, mesh=self.mesh,
                       in_specs=(in_state, self._grad_specs(), param_specs, P(), P(), P()),
                       out_specs=(out_state, param_specs), check_rep=False)
        return jax.jit(fn, donate_argnums=(0, 1, 2))

    def update(self, state, grads, params, gnorm, lr: float, t: int):
        kind = self.spec.kind(t)
        key = (kind, tuple(sorted(state["matrix"][self.buckets[0].key])))
        if key not in self._update_fns:
            self._update_fns[key] = self._make_update_fn(kind, jax.eval_shape(lambda: state))
        return self._update_fns[key](state, grads, params, gnorm, jnp.float32(lr), jnp.int32(t))

    def _make_eval_fn(self):
        mcfg = self.model_cfg
        param_specs = {k: P() for k in self.param_shapes}
        fn = shard_map(lambda params, batch: token_losses(params, batch, mcfg), mesh=self.mesh,
                       in_specs=(param_specs, P(dist.AXIS, None)), out_specs=P(dist.AXIS),
                       check_rep=False)
        return jax.jit(fn)

    def evaluate(self, params, data: TokenFile, n_seq: int) -> np.ndarray:
        """Per-sequence summed losses of the first ``n_seq`` sequences (exact, no padding)."""
        from jax.experimental import multihost_utils

        out = []
        for batch, real in eval_batches(data, n_seq, self.cfg["eval"]["batch"],
                                        self.batch_sharding):
            losses = multihost_utils.process_allgather(self._eval_fn(params, batch), tiled=True)
            out.append(np.asarray(losses)[:real])
        return np.concatenate(out)


def diagnostics(spec: dist.OptimizerSpec, buckets: tuple, state: dict, t: int) -> dict:
    """Optimizer diagnostics (Phase 05 logging) over the real (unpadded) matrices of each bucket:
    the frames' orthogonality defect (max) and, for Gimbal, the non-separability index κ of its
    flow variances D (share of the variance of log D not explained by the additive fit; mean)."""
    from jax.experimental import multihost_utils

    from gimbal.jax.gimbal import _shrunk_variances, separability_stats

    out = {}
    for b in buckets:
        st = state["matrix"][b.key]
        if "QL" not in st:
            continue
        stats = {}
        for side in ("QL", "QR"):
            q = st[side][:b.count]
            gram = jnp.einsum("kij,kil->kjl", q, q, precision=jax.lax.Precision.HIGHEST)
            stats[f"orth_defect_{side}"] = jnp.max(jnp.abs(gram - jnp.eye(q.shape[-1])))
        if spec.name == "gimbal":
            w_full = jnp.float32(1.0 - (1.0 - spec.matrix.rot_rate) ** t)

            def kappa(vf, vf_odd, w_odd, w_full=w_full):
                d = _shrunk_variances(vf, vf_odd, w_full, w_odd, spec.matrix.floor)
                ld = jnp.log(d)
                resid = ld - ld.mean(1, keepdims=True) - ld.mean(0, keepdims=True) + ld.mean()
                return jnp.var(resid) / jnp.maximum(jnp.var(ld), 1e-12)

            stats["kappa_D_mean"] = jnp.mean(jax.vmap(kappa)(
                st["VF"][:b.count], st["VF_odd"][:b.count], st["w_odd"][:b.count]))
            stats_fn = partial(separability_stats, w_full=w_full, floor=spec.matrix.floor)
            raw = jax.vmap(lambda a, b_, c, f=stats_fn: f(a, b_, w_odd=c))(
                st["VF"][:b.count], st["VF_odd"][:b.count], st["w_odd"][:b.count])
            for name, v in raw.items():
                stats[f"{name}_mean"] = jnp.mean(v)
                stats[f"{name}_median"] = jnp.median(v)
        for k, v in stats.items():
            out[f"{b.key}/{k}"] = float(multihost_utils.process_allgather(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base_125m.yaml"))
    parser.add_argument("--optimizer", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--set", action="append", default=[], help="dotted.key=yaml_value")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-before", type=int, default=None,
                        help="resume from the newest checkpoint at or before this step "
                             "(restart after a divergence)")
    parser.add_argument("--full-eval", action="store_true", default=True)
    parser.add_argument("--no-full-eval", dest="full_eval", action="store_false")
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="stop after this many steps (smoke tests); the schedule still uses "
                             "--steps")
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.distributed.initialize()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    for item in args.set:
        k, v = item.split("=", 1)
        set_path(cfg, k, v)
    if args.lr is not None:
        cfg["optimizers"][args.optimizer]["lr"] = args.lr
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    steps = int(cfg["train"]["steps"])
    run_dir = (ROOT / args.run_dir) if not os.path.isabs(args.run_dir) else pathlib.Path(
        args.run_dir)
    trainer = Trainer(cfg, args.optimizer)
    warmup = max(1, round(cfg["train"]["warmup_frac"] * steps))
    batch = int(cfg["train"]["batch"])
    tokens_per_step = batch * trainer.model_cfg.seq_len

    train_data = TokenFile(ROOT / cfg["data"]["train"], trainer.model_cfg.seq_len)
    val_data = TokenFile(ROOT / cfg["data"]["val"], trainer.model_cfg.seq_len)
    subset = int(cfg["data"].get("subset_seqs", 0))
    if subset:
        # Smoke test 1: repeat a small fixed subset (epochs of seeded permutations).
        rng = np.random.default_rng(args.seed)
        need = steps * int(cfg["train"]["batch"])
        order = np.concatenate([rng.permutation(subset) for _ in range(-(-need // subset))])
    else:
        order = train_order(train_data.n_seq, args.seed)

    params, state = trainer.init(args.seed)
    start = 0
    ckpt_dir = run_dir / "ckpt"
    if (args.resume or args.resume_before is not None) and checkpoint.latest(
            ckpt_dir, args.resume_before) is not None:
        start = checkpoint.latest(ckpt_dir, args.resume_before)
        params, state = checkpoint.restore(ckpt_dir, start, trainer, args.seed)

    writer = is_writer()
    if writer:
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "optimizer": args.optimizer, "peak_lr": trainer.peak_lr, "seed": args.seed,
            "steps": steps, "warmup": warmup, "tokens_per_step": tokens_per_step,
            "matrix_config": asdict(trainer.spec.matrix), "other_config": asdict(
                trainer.spec.other), "config": cfg, "n_params": count_params(params),
            "n_devices": trainer.n_dev, "jax": jax.__version__,
            **code_version(),
            "argv": vars(args),
        }
        (run_dir / "config.json").write_text(json.dumps(meta, indent=1, default=str))
    mode = "a" if start > 0 else "w"  # a fresh start (or retry) replaces earlier partial logs
    log = (run_dir / "log.jsonl").open(mode) if writer else None
    evals = (run_dir / "eval.jsonl").open(mode) if writer else None
    diag = (run_dir / "diag.jsonl").open(mode) if writer else None

    def emit(f, rec):
        if f is not None:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    stop = steps if args.stop_after is None else min(steps, args.stop_after)
    if args.eval_at_start:
        v = trainer.evaluate(params, val_data, int(cfg["eval"]["subset_seqs"]))
        emit(evals, {"step": start, "val_loss_subset": float(v.sum() / (
            len(v) * trainer.model_cfg.seq_len)), "tokens": start * tokens_per_step})
    pending = None  # (step, loss, gnorm, lr): read one step late so the device never idles
    # Divergence (Phase 07): loss above twice its minimum over the previous 100 steps for 50
    # consecutive steps, or a non-finite loss. The rule was written for language-model losses
    # (~3 nats, where doubling means several nats); an absolute margin of 1 nat keeps it from
    # firing on the relative fluctuations of a nearly zero loss (F-029, overfitting smoke test).
    recent: list[float] = []
    above = 0
    t_last = time.perf_counter()
    t_start = t_last
    diverged = False
    eval_every = int(cfg["eval"]["every"])
    for s, batch_arr in enumerate(train_batches(train_data, order, batch, start, stop,
                                                trainer.batch_sharding), start=start):
        lr = lr_at(s, trainer.peak_lr, steps, warmup, cfg["train"]["final_lr_frac"])
        loss, grads, gnorm = trainer._grad_fn(params, batch_arr)
        state, params = trainer.update(state, grads, params, gnorm, lr, s + 1)
        if pending is not None:
            ps, ploss, pgnorm, plr = pending
            ploss, pgnorm = float(ploss), float(pgnorm)
            now = time.perf_counter()
            emit(log, {"step": ps, "loss": ploss, "lr": plr, "grad_norm": pgnorm,
                       "step_time": now - t_last, "tokens_per_s": tokens_per_step / (now - t_last),
                       "wall": now - t_start})
            t_last = now
            if not math.isfinite(ploss):
                diverged = True
                break
            low = min(recent) if len(recent) == 100 else math.inf
            above = above + 1 if ploss > 2 * low and ploss > low + 1.0 else 0
            recent = (recent + [ploss])[-100:]
            if above >= 50:
                diverged = True
                break
        pending = (s, loss, gnorm, lr)
        done = s + 1
        if done % eval_every == 0 or done == stop:
            jax.block_until_ready(params)
            t0 = time.perf_counter()
            v = trainer.evaluate(params, val_data, int(cfg["eval"]["subset_seqs"]))
            emit(evals, {"step": done, "val_loss_subset": float(v.sum() / (
                len(v) * trainer.model_cfg.seq_len)), "eval_seconds": time.perf_counter() - t0,
                "tokens": done * tokens_per_step})
            t_last += time.perf_counter() - t0  # evaluation time is not step time
        # Diagnostics and checkpoints are overheads: like evaluation, they are timed separately
        # and kept out of the step times (Phase 08 reports them apart from the step time).
        if done % int(cfg["log"]["diag_every"]) == 0:
            jax.block_until_ready(state)
            t0 = time.perf_counter()
            rec = diagnostics(trainer.spec, trainer.buckets, state, done)
            emit(diag, {"step": done, "diag_seconds": time.perf_counter() - t0, **rec})
            t_last += time.perf_counter() - t0
        ck = int(cfg["log"].get("ckpt_every", 0))
        if ck and done % ck == 0 and done < stop:
            jax.block_until_ready(params)
            t0 = time.perf_counter()
            checkpoint.save(ckpt_dir, done, params, state)
            emit(diag, {"step": done, "checkpoint_seconds": time.perf_counter() - t0})
            t_last += time.perf_counter() - t0
    if pending is not None and not diverged:
        ps, ploss, pgnorm, plr = pending
        now = time.perf_counter()
        emit(log, {"step": ps, "loss": float(ploss), "lr": plr, "grad_norm": float(pgnorm),
                   "step_time": now - t_last, "tokens_per_s": tokens_per_step / (now - t_last),
                   "wall": now - t_start})
        diverged = not math.isfinite(float(ploss))
    final = {"steps_done": stop if not diverged else None, "diverged": diverged,
             "train_seconds": time.perf_counter() - t_start}
    if args.full_eval and not diverged and stop == steps:
        v = trainer.evaluate(params, val_data, val_data.n_seq)
        final.update({"val_loss": float(v.sum() / (len(v) * trainer.model_cfg.seq_len)),
                      "val_tokens": int(len(v) * trainer.model_cfg.seq_len)})
        if writer:
            np.save(run_dir / "val_seq_losses.npy", v.astype(np.float32))
        if int(cfg["log"].get("ckpt_every", 0)):
            # Full state: the final frames feed the Phase 08 frame diagnostics, the parameters
            # the one-time test evaluation. The periodic checkpoints are only for restarts.
            checkpoint.save(ckpt_dir, stop, params, state, keep=1)
    if writer:
        (run_dir / "final.json").write_text(json.dumps(final, indent=1))
        print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
