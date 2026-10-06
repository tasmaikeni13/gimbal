"""Gimbal (PyTorch reference implementation).

Gimbal runs Adam in a rotated frame ``(Q_L, Q_R)``, like SOAP, but estimates the frame under
the same model that Adam's diagonal assumes: the gradient's second moment is diagonal, with free
entries ``D``, in a Kronecker frame. The maximum-likelihood frame of that model is a joint
diagonalization problem. Gimbal follows it with one natural-gradient step on ``O(m) x O(n)`` per
iteration, using only matrix products (theory/gimbal_theory.md, Section 5):

    Z   = Q_L^T G Q_R                      rotated gradient
    A   = 1 / D,  D = V_hat + floor        current variance estimates
    S_L = Z (Z * A)^T,   E_L = S_L - S_L^T     score of the left rotation   (Lemma A)
    F_L = D A^T + A D^T - 2n               Fisher information per pair     (Theorem 2)
    Ω_L = -rot_rate * E_L / (F_L + damping * n)
    Q_L <- Q_L (I + Ω_L + Ω_L^2 / 2)       retraction, defect Ω^4/4         (Theorem 5)

and symmetrically on the right. There are no Kronecker factor buffers and, after the optional
eigendecomposition used to initialize the frame, no QR or eigendecomposition.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from ._linalg import (
    compute_dtype,
    eigh_desc,
    expm2,
    eye_like,
    polish_until_orthogonal,
    rotate,
    skew_spectral_norm,
    unrotate,
)


class Gimbal(Optimizer):
    """Adam in a Kronecker frame estimated online by maximum likelihood.

    Parameters
    ----------
    params:
        2-D parameters only. Route embeddings, norms and biases to AdamW.
    lr, betas, eps, weight_decay:
        As in AdamW (decoupled weight decay).
    rot_rate:
        Fraction of the per-pair natural-gradient step taken per iteration (``alpha``). The frame
        estimate behaves like an average over about ``(2 - alpha) / alpha`` gradients (Theorem 4).
    damping:
        Levenberg–Marquardt damping ``delta`` of the Fisher normalizer, in units of the mean
        per-group Fisher information. Bounds the rotation of nearly degenerate pairs.
    floor:
        Relative floor added to the variance estimates, as a fraction of their mean.
    max_angle:
        Clip on each generator entry (radians); a trust region for early, noisy steps.
    max_rotation:
        Cap on the spectral norm of each step's generator (radians).
    polish_every:
        Period of the Newton–Schulz re-orthonormalization of the frames. One polish per step
        keeps the frames orthogonal to working precision (Theorem 5.3).
    transport:
        If True, transport the second moment to the new frame with the doubly-stochastic map
        ``V <- (P_L o P_L)^T V (P_R o P_R)`` (Theorem 7), with columns renormalized to sum to one
        so that the retraction's small orthogonality defect cannot change the total.
    init:
        ``"eigh"`` initializes the frame from the eigenvectors of the first gradient's
        ``G G^T`` and ``G^T G``; ``"identity"`` starts from the identity frame.
    max_precond_dim:
        Sides larger than this keep an identity frame (one-sided mode).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rot_rate: float = 0.1,
        damping: float = 0.1,
        floor: float = 1e-8,
        max_angle: float = 0.25,
        max_rotation: float = 1.0,
        polish_every: int = 1,
        transport: bool = False,
        init: str = "eigh",
        max_precond_dim: int = 8192,
    ) -> None:
        if init not in ("eigh", "identity"):
            raise ValueError(f"unknown init {init!r}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rot_rate=rot_rate,
            damping=damping,
            floor=floor,
            max_angle=max_angle,
            max_rotation=max_rotation,
            polish_every=polish_every,
            transport=transport,
            init=init,
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
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Gimbal handles 2-D parameters; route the rest to AdamW")
                self._update(p, group)
        return loss

    def _init_state(self, state: dict, g: torch.Tensor, group: dict) -> None:
        m, n = g.shape
        state["step"] = 0
        state["M"] = torch.zeros_like(g)
        state["V"] = torch.zeros_like(g)
        max_dim = group["max_precond_dim"]
        if group["init"] == "eigh":
            state["QL"] = eigh_desc(g @ g.T)[0] if m <= max_dim else None
            state["QR"] = eigh_desc(g.T @ g)[0] if n <= max_dim else None
        else:
            state["QL"] = eye_like(m, g) if m <= max_dim else None
            state["QR"] = eye_like(n, g) if n <= max_dim else None

    def _update(self, p: torch.Tensor, group: dict) -> None:
        g = compute_dtype(p.grad)
        state = self.state[p]
        if not state:
            self._init_state(state, g, group)
        state["step"] += 1
        t = state["step"]
        b1, b2 = group["betas"]
        ql, qr = state["QL"], state["QR"]

        # Adam in the current frame. M lives in parameter coordinates, so it never needs
        # transport; V lives in rotated coordinates.
        z = rotate(g, ql, qr)
        m_buf, v_buf = state["M"], state["V"]
        m_buf.mul_(b1).add_(g, alpha=1 - b1)
        v_buf.mul_(b2).addcmul_(z, z, value=1 - b2)
        v_hat = v_buf / (1 - b2**t)
        n_rot = rotate(m_buf, ql, qr).div_(1 - b1**t).div_(v_hat.sqrt().add_(group["eps"]))
        update = unrotate(n_rot, ql, qr)
        if group["weight_decay"] != 0:
            p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-group["lr"])

        # Frame flow: one natural-gradient step of the joint-diagonalization likelihood.
        self._flow(state, z, v_hat, group)

    @staticmethod
    def _generator(
        z: torch.Tensor, d: torch.Tensor, a: torch.Tensor, group: dict, side: str
    ) -> torch.Tensor:
        """Damped natural-gradient generator for one side (skew-symmetric)."""
        if side == "left":
            groups = z.shape[1]
            s = z @ (z * a).T
            fisher = d @ a.T
        else:
            groups = z.shape[0]
            s = z.T @ (z * a)
            fisher = d.T @ a
        score = s - s.T
        fisher = (fisher + fisher.T - 2.0 * groups).clamp_min(0.0)
        omega = score.mul_(-group["rot_rate"]).div_(fisher + group["damping"] * groups)
        omega.fill_diagonal_(0.0)
        omega.clamp_(-group["max_angle"], group["max_angle"])
        # Spectral trust region: with ||Ω||_2 <= 1 the retraction's singular values stay in
        # [1, 1.12], inside the convergence region of the Newton–Schulz polish.
        norm = skew_spectral_norm(omega)
        if norm > group["max_rotation"]:
            omega.mul_(group["max_rotation"] / norm)
        return omega

    def _flow(self, state: dict, z: torch.Tensor, v_hat: torch.Tensor, group: dict) -> None:
        d = v_hat + (group["floor"] * v_hat.mean() + 1e-30)
        a = d.reciprocal()
        p_left = p_right = None
        if state["QL"] is not None:
            p_left = expm2(self._generator(z, d, a, group, "left"))
            state["QL"] = state["QL"] @ p_left
        if state["QR"] is not None:
            p_right = expm2(self._generator(z, d, a, group, "right"))
            state["QR"] = state["QR"] @ p_right
        if group["transport"]:
            v_buf = state["V"]
            # Rows of P o P sum to one for an orthogonal P; renormalizing them makes the transport
            # preserve the total second moment exactly despite the retraction's tiny defect.
            if p_left is not None:
                t_left = p_left * p_left
                v_buf.copy_((t_left / t_left.sum(dim=1, keepdim=True)).T @ v_buf)
            if p_right is not None:
                t_right = p_right * p_right
                v_buf.copy_(v_buf @ (t_right / t_right.sum(dim=1, keepdim=True)))
        if state["step"] % group["polish_every"] == 0:
            for key in ("QL", "QR"):
                if state[key] is not None:
                    state[key] = polish_until_orthogonal(state[key])
