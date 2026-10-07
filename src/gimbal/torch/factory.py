"""Build an optimizer pair with the routing used in every comparison.

Hidden 2-D matrices go to the matrix optimizer under test; everything else (embeddings, output
head, norms, biases) goes to one AdamW configured identically for every method, so that only the
matrix rule differs between runs (phases/03_reference_implementations.md).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from .aro import AROSinkhorn
from .gimbal import Gimbal
from .klsoap import KLSOAP
from .muon import Muon, NorMuon
from .soap import SOAP
from .splus import SPlus

MATRIX_OPTIMIZERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adamw": lambda params, **kw: torch.optim.AdamW(params, **kw),
    "soap": lambda params, **kw: SOAP(params, **kw),
    "soap_rt": lambda params, **kw: SOAP(params, realtime=True, **kw),
    "klsoap": lambda params, **kw: KLSOAP(params, **kw),
    "muon": lambda params, **kw: Muon(params, **kw),
    "normuon": lambda params, **kw: NorMuon(params, **kw),
    "splus": lambda params, **kw: SPlus(params, **kw),
    "aro": lambda params, **kw: AROSinkhorn(params, **kw),
    "gimbal": lambda params, **kw: Gimbal(params, **kw),
}


@dataclass
class OptimizerPair:
    """A matrix optimizer for hidden matrices and an AdamW for the remaining parameters."""

    matrix: torch.optim.Optimizer | None
    other: torch.optim.Optimizer | None

    def zero_grad(self) -> None:
        """Clear the gradients of both optimizers (set to ``None``)."""
        for opt in (self.matrix, self.other):
            if opt is not None:
                opt.zero_grad(set_to_none=True)

    def step(self) -> None:
        """Step both optimizers."""
        for opt in (self.matrix, self.other):
            if opt is not None:
                opt.step()

    def set_lr_scale(self, scale: float) -> None:
        """Multiply every group's base learning rate by ``scale`` (for schedules)."""
        for opt in (self.matrix, self.other):
            if opt is None:
                continue
            for group in opt.param_groups:
                group.setdefault("base_lr", group["lr"])
                group["lr"] = group["base_lr"] * scale


def build(
    name: str,
    hidden_matrices: Iterable[torch.Tensor],
    other_params: Iterable[torch.Tensor],
    matrix_kwargs: dict | None = None,
    other_kwargs: dict | None = None,
) -> OptimizerPair:
    """Pair the matrix optimizer ``name`` (hidden matrices) with AdamW (everything else).

    Parameters
    ----------
    name : str
        Key of ``MATRIX_OPTIMIZERS``.
    hidden_matrices, other_params : iterable of torch.Tensor
        The two parameter groups; either may be empty.
    matrix_kwargs, other_kwargs : dict, optional
        Keyword arguments of the two optimizers.

    Returns
    -------
    OptimizerPair
    """
    if name not in MATRIX_OPTIMIZERS:
        raise KeyError(f"unknown optimizer {name!r}; choose from {sorted(MATRIX_OPTIMIZERS)}")
    hidden = list(hidden_matrices)
    other = list(other_params)
    matrix = MATRIX_OPTIMIZERS[name](hidden, **(matrix_kwargs or {})) if hidden else None
    other_opt = torch.optim.AdamW(other, **(other_kwargs or {})) if other else None
    return OptimizerPair(matrix, other_opt)
