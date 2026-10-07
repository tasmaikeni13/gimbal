# TPU benchmarks (Phase 04)

Every number in [`docs/performance.md`](../../docs/performance.md) is read from `results/` by
`make_performance_doc.py`. Each script runs on all four hosts through `scripts/tpu/launch.sh`.

| Script | Measures | Output |
|---|---|---|
| `bench_attention.py` | gradient-step time for XLA attention and flash-attention tile sizes | `results/attention.json` |
| `bench_variants.py` | where the gradient step spends its time (layer stacking, attention, loss, reduction) | `results/variants*.json` |
| `bench_collectives.py` | all-reduce, reduce-scatter and all-gather of a gradient bucket in float32 and bfloat16 | `results/collectives.json` |
| `bench_components.py` | component timings: attention kernels, gradient step, optimizer updates | `results/components.json` |
| `step_times.py` | steady-state step time (C-017) from the logs of `scripts/tpu/time_variants.sh` | `results/step_times.json` |
| `profile_step.py` | device time of one training step by XLA operation (trace) | `results/profile_<optimizer>.json` |
| `profile_update.py` | device time of one optimizer update kind, traced alone | `results/profile_update_<optimizer>.json` |
| `xplane.py` | minimal reader of the profiler's `*.xplane.pb` traces, used by the two profilers | — |
| `dump_hlo.py` | optimized HLO of the gradient and update programs and their collectives (checks for unintended all-gathers of optimizer state) | `runs/hlo/` (not tracked) |
| `make_performance_doc.py` | writes `docs/performance.md` | — |

`results/tests_tpu.txt` is the output of the JAX test suite on one host's chips
(`scripts/tpu/single_host_env.sh`).
