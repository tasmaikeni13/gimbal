# TPU scripts

The v4-32 slice has four hosts; multi-host JAX needs the same command on every host.

| Script | Purpose |
|---|---|
| `launch.sh <logdir> <command>` | run a command on all hosts (worker 0 locally, 1–3 over ssh), one log per host; copies new compilation-cache entries to the workers first |
| `snapshot.sh <name>` | freeze `src/`, `configs/`, `scripts/` and `benchmarks/` to `~/gimbal_snapshots/<name>` on every host, with the commit and diff (runs execute frozen code, F-028) |
| `sync.sh`, `sync_cache.sh` | copy the code, or new compilation-cache entries (F-031), from worker 0 to workers 1–3 |
| `single_host_env.sh` | environment for using one host's four chips alone (tests, small checks) |
| `env_check.py` | versions, devices and memory on every host (`runs/env.json`) |
| `smoke_tests.sh`, `check_smoke.py` | Phase 05 smoke tests and their verdicts (`runs/smoke/`) |
| `time_variants.sh` | 400-step runs that measure steady-state step times (Phase 04) |
| `tune.py`, `freeze_configs.py` | Phase 06 sweep and the frozen configurations |
| `main_runs.py`, `keep_checkpoint.sh` | Phase 07 confirmatory runs; keeping the step-1000 checkpoints for the frame probe (D-009) |
| `frame_probe.py`, `eval_test.py` | Phase 08 mechanism probe and the one-time test evaluation |
