"""ARO-Sinkhorn (Gong et al., arXiv:2602.09006, Sections 3–4).

``R_t = QR(M_t f(R_{t-1}^T M_t)^T)`` (one-sided, left), ``ΔW = R_t f(R_t^T M_t)``, with
``f`` = five rounds of simultaneous row/column normalization (SinkGD), and the RMS
re-normalization ``ΔW <- 0.2 sqrt(mn) ΔW / ||ΔW||_F`` used in the paper for all non-Adam methods.
The paper's shifted Cholesky-QR is a speed optimization; plain QR is used here.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from ._linalg import compute_dtype, qr_orth, sinkhorn_normalize


class AROSinkhorn(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 3e-3,
        momentum: float = 0.95,
        sinkhorn_iters: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, sinkhorn_iters=sinkhorn_iters, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta, iters = group["momentum"], group["sinkhorn_iters"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = compute_dtype(p.grad)
                state = self.state[p]
                if not state:
                    state["M"] = torch.zeros_like(g)
                    state["R"] = torch.eye(g.shape[0], dtype=g.dtype)
                m_buf = state["M"].mul_(beta).add_(g, alpha=1 - beta)
                rot = state["R"]
                rot = qr_orth(m_buf @ sinkhorn_normalize(rot.T @ m_buf, iters).T)
                state["R"] = rot
                delta = rot @ sinkhorn_normalize(rot.T @ m_buf, iters)
                delta = delta * (0.2 * delta.numel() ** 0.5 / (delta.norm() + 1e-30))
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(delta.to(p.dtype), alpha=-group["lr"])
        return loss
