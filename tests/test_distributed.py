"""The training loop's distributed optimizer path agrees with the per-matrix steps (float64).

Runs ``tests/check_distributed.py`` in a subprocess (it needs 4 simulated CPU devices, which must be
configured before JAX starts) and checks AdamW, SOAP and Gimbal over 120 steps, past Gimbal's warm
start and into its adaptive schedule.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def test_distributed_step_matches_per_matrix_steps():
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    out = subprocess.run(
        [sys.executable, str(HERE / "check_distributed.py"), "--steps", "120"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    res = json.loads(out.stdout.strip().splitlines()[-1])
    # AdamW and SOAP: the same arithmetic in another order is bit-identical here. Gimbal's first
    # steps amplify float64 rounding to ~1e-8 (F-025); an orchestration error would be >= 1e-3.
    tol = {"adamw": 1e-12, "soap": 1e-12, "gimbal": 1e-6}
    for opt, buckets in res.items():
        for key, r in buckets.items():
            assert r["max_param_change"] > 1e-2, (opt, key, r)  # the parameters really moved
            assert r["rel"] < tol[opt], (opt, key, r)
