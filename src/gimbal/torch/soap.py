"""SOAP (Vyas et al., arXiv:2409.11321), following the official implementation.

Two modes:

* ``realtime=False`` (default): the reference algorithm. The first call only initializes the
  preconditioner; Kronecker factors are updated after the parameter step; the frame is refreshed
  every ``precondition_frequency`` steps by one power-iteration step and a QR decomposition,
  after sorting by estimated eigenvalues and re-ordering the second moment accordingly.
* ``realtime=True``: the "real-time" variant of arXiv:2607.20548 (Section 5.4.2): the factors
  include the current gradient and the frame is refreshed every step *before* the gradient is
  projected.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from ._linalg import compute_dtype, eigh_desc, qr_orth, rotate, unrotate


class SOAP(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        correct_bias: bool = True,
        realtime: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=1 if realtime else precondition_frequency,
            max_precond_dim=max_precond_dim,
            correct_bias=correct_bias,
            realtime=realtime,
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
                        raise ValueError("SOAP here handles 2-D parameters; route others to AdamW")
                    self._update(p, group)
        return loss

    # -- preconditioner bookkeeping (mirrors the official code) ------------------------------
    @staticmethod
    def _shampoo_beta(group: dict) -> float:
        sb = group["shampoo_beta"]
        return sb if sb >= 0 else group["betas"][1]

    def _accumulate(self, g: torch.Tensor, state: dict, group: dict) -> None:
        sb = self._shampoo_beta(group)
        if state["GG"][0] is not None:
            state["GG"][0].lerp_(g @ g.T, 1 - sb)
        if state["GG"][1] is not None:
            state["GG"][1].lerp_(g.T @ g, 1 - sb)

    def _refresh_qr(self, state: dict) -> None:
        """One power-iteration step + QR per side, after sorting by estimated eigenvalues."""
        v_buf = state["exp_avg_sq"]
        new_q = []
        for dim, (gg, q) in enumerate(zip(state["GG"], state["Q"], strict=True)):
            if gg is None:
                new_q.append(None)
                continue
            est_eig = torch.diag(q.T @ gg @ q)
            order = torch.argsort(est_eig, descending=True)
            v_buf = v_buf.index_select(dim, order)
            new_q.append(qr_orth(gg @ q[:, order]))
        state["exp_avg_sq"] = v_buf
        state["Q"] = new_q

    def _init(self, g: torch.Tensor, state: dict, group: dict) -> None:
        m, n = g.shape
        max_dim = group["max_precond_dim"]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(g)
        state["exp_avg_sq"] = torch.zeros_like(g)
        state["GG"] = [
            torch.zeros(m, m, dtype=g.dtype) if m <= max_dim else None,
            torch.zeros(n, n, dtype=g.dtype) if n <= max_dim else None,
        ]
        self._accumulate(g, state, group)
        state["Q"] = [eigh_desc(gg)[0] if gg is not None else None for gg in state["GG"]]

    def _update(self, p: torch.Tensor, group: dict) -> None:
        g = compute_dtype(p.grad)
        state = self.state[p]
        realtime = group["realtime"]
        if "Q" not in state:
            self._init(g, state, group)
            if not realtime:
                return  # official SOAP: the first step only initializes the preconditioner
        elif realtime:
            # Refresh the frame with factors that already contain the current gradient.
            m_orig = unrotate(state["exp_avg"], *state["Q"])
            self._accumulate(g, state, group)
            self._refresh_qr(state)
            state["exp_avg"] = rotate(m_orig, *state["Q"])

        b1, b2 = group["betas"]
        g_rot = rotate(g, *state["Q"])
        state["step"] += 1
        t = state["step"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(b1).add_(g_rot, alpha=1 - b1)
        exp_avg_sq.mul_(b2).add_(g_rot.square(), alpha=1 - b2)
        denom = exp_avg_sq.sqrt().add_(group["eps"])
        step_size = group["lr"]
        if group["correct_bias"]:
            step_size = step_size * (1 - b2**t) ** 0.5 / (1 - b1**t)
        update = unrotate(exp_avg / denom, *state["Q"])
        p.add_(update.to(p.dtype), alpha=-step_size)
        if group["weight_decay"] > 0:
            p.add_(p, alpha=-group["lr"] * group["weight_decay"])

        if not realtime:
            # Official order: factors are updated after the step; the frame every f steps.
            m_orig = unrotate(state["exp_avg"], *state["Q"])
            self._accumulate(g, state, group)
            if t % group["precondition_frequency"] == 0:
                self._refresh_qr(state)
            state["exp_avg"] = rotate(m_orig, *state["Q"])
