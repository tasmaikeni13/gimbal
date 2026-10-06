"""KL-SOAP (Lin et al., arXiv:2509.03378; large-scale recipe of arXiv:2607.20548, Alg. 2).

Kronecker factors are accumulated with the KL (maximum-likelihood, Kronecker *product*) rule

    L <- (1 - b_k) L + (b_k / n) G R^{-1} G^T,   R <- (1 - b_k) R + (b_k / m) G^T L^{-1} G,

with the inverses taken in the current frame using approximate eigenvalues
``diag(Q^T L Q)``. The frame is refreshed by one power-iteration step + QR every ``frequency``
steps (1 in the large-scale recipe) and Adam runs in the frame. The momentum is stored in rotated
coordinates and re-expressed when the frame changes, as in the reference algorithm; the second
moment is re-ordered with the frame (sorting by estimated eigenvalues).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from ._linalg import compute_dtype, eigh_desc, qr_orth, rotate, unrotate


class KLSOAP(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        beta_kron: float = 0.95,
        eps: float = 1e-8,
        kl_damping: float = 1e-6,
        weight_decay: float = 0.0,
        frequency: int = 1,
        max_precond_dim: int = 10000,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            beta_kron=beta_kron,
            eps=eps,
            kl_damping=kl_damping,
            weight_decay=weight_decay,
            frequency=frequency,
            max_precond_dim=max_precond_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if p.ndim != 2:
                        raise ValueError("KL-SOAP here handles 2-D parameters")
                    self._update(p, group)
        return loss

    @staticmethod
    def _inv_in_frame(q: torch.Tensor, lam: torch.Tensor, damping: float) -> torch.Tensor:
        lam = lam.clamp_min(0.0)
        lam = lam + damping * lam.mean() + 1e-30
        return (q / lam) @ q.T

    def _init(self, g: torch.Tensor, state: dict, group: dict) -> None:
        m, n = g.shape
        max_dim = group["max_precond_dim"]
        state["step"] = 0
        state["m"] = torch.zeros_like(g)
        state["v"] = torch.zeros_like(g)
        state["L"] = g @ g.T / n if m <= max_dim else None
        state["R"] = g.T @ g / m if n <= max_dim else None
        state["QL"] = eigh_desc(state["L"])[0] if state["L"] is not None else None
        state["QR"] = eigh_desc(state["R"])[0] if state["R"] is not None else None

    def _update(self, p: torch.Tensor, group: dict) -> None:
        g = compute_dtype(p.grad)
        state = self.state[p]
        first = not state
        if first:
            self._init(g, state, group)
        state["step"] += 1
        t = state["step"]
        m, n = g.shape
        bk, damp = group["beta_kron"], group["kl_damping"]
        ql, qr = state["QL"], state["QR"]

        if not first:
            # KL factor accumulation with inverses from the current frame (lines 4-7 of Alg. 2).
            lam_l = torch.diag(ql.T @ state["L"] @ ql) if ql is not None else None
            lam_r = torch.diag(qr.T @ state["R"] @ qr) if qr is not None else None
            r_inv = self._inv_in_frame(qr, lam_r, damp) if qr is not None else None
            l_inv = self._inv_in_frame(ql, lam_l, damp) if ql is not None else None
            if state["L"] is not None:
                gr = g @ r_inv if r_inv is not None else g
                state["L"].mul_(1 - bk).add_(gr @ g.T, alpha=bk / n)
            if state["R"] is not None:
                lg = l_inv @ g if l_inv is not None else g
                state["R"].mul_(1 - bk).add_(g.T @ lg, alpha=bk / m)
            if t % group["frequency"] == 0:
                m_orig = unrotate(state["m"], ql, qr)
                v_buf = state["v"]
                if ql is not None:
                    order = torch.argsort(torch.diag(ql.T @ state["L"] @ ql), descending=True)
                    v_buf = v_buf.index_select(0, order)
                    ql = qr_orth(state["L"] @ ql[:, order])
                if qr is not None:
                    order = torch.argsort(torch.diag(qr.T @ state["R"] @ qr), descending=True)
                    v_buf = v_buf.index_select(1, order)
                    qr = qr_orth(state["R"] @ qr[:, order])
                state["v"] = v_buf
                state["QL"], state["QR"] = ql, qr
                state["m"] = rotate(m_orig, ql, qr)

        b1, b2 = group["betas"]
        g_rot = rotate(g, ql, qr)
        state["m"].mul_(b1).add_(g_rot, alpha=1 - b1)
        state["v"].mul_(b2).addcmul_(g_rot, g_rot, value=1 - b2)
        m_hat = state["m"] / (1 - b1**t)
        v_hat = state["v"] / (1 - b2**t)
        update = unrotate(m_hat / (v_hat.sqrt() + group["eps"]), ql, qr)
        if group["weight_decay"] != 0:
            p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-group["lr"])
