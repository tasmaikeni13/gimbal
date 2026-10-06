"""JAX implementations of the optimizers of the TPU study (Phase 03): AdamW, SOAP and Gimbal.

Each optimizer is a pure per-matrix step (``adamw.step``, ``soap.step``, ``gimbal.step``) that the
training loop batches over same-shaped layers and shards across devices
(:mod:`gimbal.jax.distributed`); :mod:`gimbal.jax.optax_api` wraps the same steps as Optax
transformations. The other reference optimizers (KL-SOAP, Muon, NorMuon, SPlus, ARO) exist in
PyTorch only (``gimbal.torch``); the TPU study compares AdamW, SOAP and Gimbal (decision D-003).
"""

from . import adamw, gimbal, soap
from .adamw import AdamWConfig
from .gimbal import GimbalConfig
from .soap import SOAPConfig

__all__ = ["AdamWConfig", "GimbalConfig", "SOAPConfig", "adamw", "gimbal", "soap"]
