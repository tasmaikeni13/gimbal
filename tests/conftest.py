"""Test configuration: JAX tests run on the CPU unless ``GIMBAL_TEST_PLATFORM`` says otherwise.

On a TPU host the default would claim the accelerator (and, on a multi-host slice, wait for the
other hosts); ``GIMBAL_TEST_PLATFORM=tpu pytest tests/test_jax_optimizers.py`` runs them on TPU
(Phase 04, G4.1).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", os.environ.get("GIMBAL_TEST_PLATFORM", "cpu"))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _free_compiled_programs():
    """Drop JAX's compiled executables after each test. Without this, a run of the whole suite
    accumulates thousands of XLA CPU executables in one process and crashed (segmentation fault)
    late in the run; every test passes on its own."""
    yield
    import sys

    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
