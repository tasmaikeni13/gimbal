"""Test configuration: JAX tests run on the CPU unless ``GIMBAL_TEST_PLATFORM`` says otherwise.

On a TPU host the default would claim the accelerator (and, on a multi-host slice, wait for the
other hosts); ``GIMBAL_TEST_PLATFORM=tpu pytest tests/test_jax_optimizers.py`` runs them on TPU
(Phase 04, G4.1).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", os.environ.get("GIMBAL_TEST_PLATFORM", "cpu"))
