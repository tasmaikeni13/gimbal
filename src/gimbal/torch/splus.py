"""SPlus (Frans, Levine & Abbeel, arXiv:2506.07254), following the published JAX snippet.

``U = Q_L sign(Q_L^T M Q_R) Q_R^T * 2/(m+n)``, with Kronecker statistics (EMA ``b2``) whose
eigenvectors are recomputed every ``inverse_every`` steps, and an exponential moving average of
the parameters (``ema_rate``) intended for evaluation (:meth:`eval_params`).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from ._linalg import compute_dtype, rotate, unrotate


class SPlus(Optimizer):
    """SPlus: sign of the momentum in a Shampoo frame, with parameter averaging (see module)."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.1,
        b1: float = 0.9,
        b2: float = 0.999,
        ema_rate: float = 0.999,
        inverse_every: int = 100,
        weight_decay: float = 1e-2,
        max_dim: int = 10000,
        eps: float = 1e-30,
    ) -> None:
        defaults = dict(
            lr=lr,
            b1=b1,
            b2=b2,
            ema_rate=ema_rate,
            inverse_every=inverse_every,
            weight_decay=weight_decay,
            max_dim=max_dim,
            eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _eigvecs(s: torch.Tensor, eps: float) -> torch.Tensor:
        _, q = torch.linalg.eigh(s + eps * torch.eye(s.shape[0], dtype=s.dtype))
        return q

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimization step.

        Parameters
        ----------
        closure : callable, optional
            Re-evaluates the model and returns the loss (PyTorch convention).

        Returns
        -------
        torch.Tensor or None
            The closure's loss, if a closure was given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = compute_dtype(p.grad)
                m, n = g.shape
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["mom"] = torch.zeros_like(g)
                    state["ema"] = torch.zeros_like(g)
                    max_dim = group["max_dim"]
                    state["L"] = torch.zeros(m, m, dtype=g.dtype) if m < max_dim else None
                    state["R"] = torch.zeros(n, n, dtype=g.dtype) if n < max_dim else None
                    state["QL"] = torch.eye(m, dtype=g.dtype) if m < max_dim else None
                    state["QR"] = torch.eye(n, dtype=g.dtype) if n < max_dim else None
                state["step"] += 1
                t = state["step"]
                mom = state["mom"].mul_(group["b1"]).add_(g, alpha=1 - group["b1"])
                upd = unrotate(
                    torch.sign(rotate(mom, state["QL"], state["QR"])), state["QL"], state["QR"]
                )
                b2 = group["b2"]
                if state["L"] is not None:
                    state["L"].mul_(b2).add_(g @ g.T, alpha=1 - b2)
                if state["R"] is not None:
                    state["R"].mul_(b2).add_(g.T @ g, alpha=1 - b2)
                if t % group["inverse_every"] == 0 or t == 1:
                    if state["L"] is not None:
                        state["QL"] = self._eigvecs(state["L"], group["eps"])
                    if state["R"] is not None:
                        state["QR"] = self._eigvecs(state["R"], group["eps"])
                upd = upd * (2.0 / (m + n))
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(upd.to(p.dtype), alpha=-group["lr"])
                rate = group["ema_rate"]
                state["ema"].mul_(rate).add_(p.to(g.dtype), alpha=1 - rate)
        return loss

    @torch.no_grad()
    def eval_params(self) -> dict[torch.Tensor, torch.Tensor]:
        """Bias-corrected parameter EMA for evaluation (SPlus's iterate averaging)."""
        out = {}
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state:
                    out[p] = state["ema"] / (1 - group["ema_rate"] ** state["step"])
        return out
