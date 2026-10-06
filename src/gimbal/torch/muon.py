"""Muon (Jordan et al., 2024) and NorMuon (Li et al., arXiv:2510.05491)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from ._linalg import compute_dtype, zeropower_ns5


class Muon(Optimizer):
    """Momentum orthogonalized by five quintic Newton–Schulz steps (reference recipe).

    Update: ``buf <- lerp(buf, g, 1 - momentum)``; Nesterov ``u = lerp(g, buf, momentum)``;
    ``u <- NS5(u) * max(1, m/n)^0.5``; decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = compute_dtype(p.grad)
                state = self.state[p]
                if not state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.lerp_(g, 1 - beta)
                u = g.lerp(buf, beta) if group["nesterov"] else buf
                u = zeropower_ns5(u, steps=group["ns_steps"])
                u = u * max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-group["lr"])
        return loss


class NorMuon(Optimizer):
    """Muon followed by neuron-wise (row-wise) second-moment normalization (Alg. 1 of
    arXiv:2510.05491) and the RMS-matching step size ``0.2 * lr * sqrt(mn) / ||O_hat||_F``."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = compute_dtype(p.grad)
                state = self.state[p]
                if not state:
                    state["M"] = torch.zeros_like(g)
                    state["v"] = torch.zeros(g.shape[0], 1, dtype=g.dtype)
                m_buf, v_buf = state["M"], state["v"]
                m_buf.mul_(b1).add_(g, alpha=1 - b1)
                o = zeropower_ns5(m_buf, steps=group["ns_steps"])
                v_buf.mul_(b2).add_((o * o).mean(dim=1, keepdim=True), alpha=1 - b2)
                o_hat = o / (v_buf.sqrt() + group["eps"])
                scale = 0.2 * (o.numel() ** 0.5) / (o_hat.norm() + 1e-30)
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(o_hat.to(p.dtype), alpha=-group["lr"] * scale)
        return loss
