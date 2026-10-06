"""Gimbal (PyTorch reference implementation).

Gimbal runs Adam in a rotated frame ``(Q_L, Q_R)``, like SOAP, but estimates the frame under
the same model that Adam's diagonal assumes: the gradient's second moment is diagonal, with free
entries ``D``, in a Kronecker frame. The maximum-likelihood frame of that model is a joint
diagonalization problem. Gimbal follows it with one natural-gradient step on ``O(m) x O(n)`` per
iteration, using only matrix products (theory/gimbal_theory.md, Section 5):

    Z   = Q_L^T (G - c M) Q_R              rotated innovation (M: momentum, Proposition 5.7)
    A   = 1 / D                            D: variance estimates (Lemma 5.4, Proposition 5.5)
    S_L = Z (Z * A)^T,   E_L = S_L - S_L^T     score of the left rotation   (Lemma A)
    F_L = D A^T + A D^T - 2n               Fisher information per pair     (Theorem 2)
    Ω_L = -rot_rate * E_L / (F_L + damping * n)
    Q_L <- Q_L (I + Ω_L + Ω_L^2 / 2)       retraction, defect Ω^4/4         (Theorem 5)

and symmetrically on the right. ``D`` is an exponential average of ``Z^2`` with the frame's own
memory, shrunk toward its separable (Kronecker) fit by an empirical-Bayes factor estimated from
a split-sample noise estimate. The mean subtracted from the gradient is the bias-corrected momentum
of the previous step times its own empirical-Bayes factor ``c``, so zero-mean gradients are used as
they are and a gradient with a real mean (a deterministic descent signal) does not tilt the frame.
After a short warm start (two eigendecompositions in total) there are no Kronecker factor buffers
and no QR or eigendecomposition.
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


def _rotated_copy(x: torch.Tensor, ql: torch.Tensor | None,
                  qr: torch.Tensor | None) -> torch.Tensor:
    """``rotate`` that never aliases its input (with no frame on either side it returns ``x``)."""
    y = rotate(x, ql, qr)
    return y.clone() if y is x else y


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
        The default 0.05 = 1 - beta_2 gives the frame, its flow variances and Adam's second moment
        one window (effective sample size 39 for beta_2 = 0.95), the convention of SOAP's default
        ``shampoo_beta = beta_2``; 0.02 (C-002, chosen on stationary synthetic streams) lagged on
        language-model gradients (F-027, change C-018).
    rot_schedule:
        ``"constant"`` uses ``rot_rate`` at every step. ``"bias_corrected"`` uses
        ``alpha_t = alpha / (1 - (1 - alpha)^t)`` (capped at ``rot_rate_max``): the frame estimate
        then behaves like a bias-corrected exponential average of per-step estimates, i.e. like a
        batch estimate early in training and like a tracker with memory ``~2/alpha`` later.
    rot_rate_max:
        Cap on the scheduled rotation rate.
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
        Period, in frame moves, of the Newton–Schulz re-orthonormalization. One polish per move
        keeps the frames orthogonal to working precision (Theorem 5.3).
    frame_every:
        Move the frame once every ``frame_every`` steps using the mean score of those steps
        (amortizes the retraction, polish and Fisher costs; see ``_flow``). The default 4 keeps
        the optimizer's cost below SOAP's in the cost model (Proposition 9, E2.8) at a frame
        quality within a few percent of moving every step (E2.1, E2.9); change C-012.
    frame_schedule:
        ``"fixed"`` moves the frame every ``frame_every`` steps. ``"adaptive"`` moves it every
        ``k_t = clamp(round(frame_every · rot_rate / α_t), 1, frame_every)`` steps, where ``α_t`` is
        the scheduled rotation rate: the amortized step matches the per-step flow to first order
        in ``k·α`` (Section 5), so the product is held at its steady-state value and the early,
        fast-moving phase is not amortized. Default ``"adaptive"`` since C-018 (F-027): the cost
        after the first ~50 steps is that of ``frame_every``.
    flow_beta:
        Variance estimate used by the frame flow. ``None`` reuses Adam's second moment (memory set
        by ``betas[1]``). A float keeps a separate EMA with that coefficient; ``"tied"`` uses
        ``1 - rot_rate``, so the frame and the variances it is fitted to share one estimation window
        (joint maximum likelihood with a common forgetting factor, Lemma 5.4).
    flow_shrink:
        Shrink the log of the flow's variance estimate toward its additive (separable) fit with
        the positive-part James–Stein factor ``c = (1 - noise / residual)_+`` (Proposition 5.5).
        The noise level is measured by splitting the average into interleaved odd and even steps.
        Needs ``flow_beta``; together they cost two m x n buffers.
    flow_center:
        Statistic the frame is fitted to. ``False`` uses the gradient, i.e. the zero-mean model of
        the uncentered second moment. ``True`` uses the innovation ``G_t - M_{t-1}`` (``M`` the
        bias-corrected momentum), the likelihood of a gradient with a non-zero mean. ``"adaptive"``
        uses ``G_t - c M_{t-1}`` with the positive-part James–Stein factor
        ``c = (1 - noise / ||M_{t-1}||^2)_+`` of the momentum as an estimate of the mean: about 0
        for zero-mean gradients, where subtracting the momentum would only add noise, and about 1
        when a real mean dominates (Proposition 5.7; change C-013). Centering reuses the
        momentum's rotation, so it adds no matrix product.
    transport:
        If True, transport the second moment to the new frame with the doubly-stochastic map
        ``V <- (P_L o P_L)^T V (P_R o P_R)`` (Theorem 7), with columns renormalized to sum to one
        so that the retraction's small orthogonality defect cannot change the total.
    init:
        ``"eigh"`` initializes the frame from the eigenvectors of the first gradient's
        ``G G^T`` and ``G^T G``; ``"identity"`` starts from the identity frame. ``"pooled"`` starts
        like ``"eigh"`` and, after ``warm_start_steps`` steps, restarts the flow from the
        eigenvectors of the pooled factors of those steps (a consistent estimator; the flow then
        acts as Fisher scoring from a consistent start, Le Cam's one-step construction). The
        second moments are transported to the new frame (Theorem 7) and the factor buffers are
        freed.
    warm_start_steps:
        Length of the pooled warm start (only for ``init="pooled"``).
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
        rot_rate: float = 0.05,
        rot_schedule: str = "bias_corrected",
        rot_rate_max: float = 0.5,
        damping: float = 0.003,
        floor: float = 1e-8,
        max_angle: float = 0.25,
        max_rotation: float = 1.0,
        polish_every: int = 1,
        frame_every: int = 4,
        frame_schedule: str = "adaptive",
        flow_beta: float | str | None = "tied",
        flow_shrink: bool = True,
        flow_center: bool | str = "adaptive",
        transport: bool = False,
        init: str = "pooled",
        warm_start_steps: int = 50,
        max_precond_dim: int = 8192,
    ) -> None:
        if init not in ("pooled", "eigh", "identity"):
            raise ValueError(f"unknown init {init!r}")
        if flow_center not in (False, True, "adaptive"):
            raise ValueError(f"unknown flow_center {flow_center!r}")
        if frame_schedule not in ("fixed", "adaptive"):
            raise ValueError(f"unknown frame_schedule {frame_schedule!r}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rot_rate=rot_rate,
            rot_schedule=rot_schedule,
            rot_rate_max=rot_rate_max,
            damping=damping,
            floor=floor,
            max_angle=max_angle,
            max_rotation=max_rotation,
            polish_every=polish_every,
            frame_every=frame_every,
            frame_schedule=frame_schedule,
            flow_beta=flow_beta,
            flow_shrink=flow_shrink,
            flow_center=flow_center,
            transport=transport,
            init=init,
            warm_start_steps=warm_start_steps,
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
        if group["init"] == "pooled" and group["warm_start_steps"] > 1:
            state["L_acc"] = g @ g.T if m <= max_dim else None
            state["R_acc"] = g.T @ g if n <= max_dim else None
        if group["init"] in ("eigh", "pooled"):
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
        g_flow, z_flow = g, z
        if group["flow_center"] and t > 1:
            # Frame statistics on the innovation G_t - c M_{t-1} (Proposition 5.7). M_{t-1} is
            # predictable, so the plug-in mean adds no cross term to the expected score. Rotating
            # M_{t-1} instead of M_t gives both from one product, as rot(M_t) is linear in it.
            c_mu = 1.0
            if group["flow_center"] == "adaptive":
                c_mu = self._mean_shrinkage(m_buf, v_buf, b1, b2, t)
            scale = c_mu / (1 - b1 ** (t - 1))
            m_rot = _rotated_copy(m_buf, ql, qr)
            z_flow = z - scale * m_rot
            if "L_acc" in state:
                g_flow = g - scale * m_buf
            m_rot.mul_(b1).add_(z, alpha=1 - b1)
            m_buf.mul_(b1).add_(g, alpha=1 - b1)
        else:
            m_buf.mul_(b1).add_(g, alpha=1 - b1)
            m_rot = _rotated_copy(m_buf, ql, qr)
        v_buf.mul_(b2).addcmul_(z, z, value=1 - b2)
        v_hat = v_buf / (1 - b2**t)
        n_rot = m_rot.div_(1 - b1**t).div_(v_hat.sqrt().add_(group["eps"]))
        update = unrotate(n_rot, ql, qr)
        if group["weight_decay"] != 0:
            p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-group["lr"])

        # Pooled warm start: accumulate factors, then restart the frame from their eigenvectors.
        restarted = False
        if "L_acc" in state:
            if t > 1:
                if state["L_acc"] is not None:
                    state["L_acc"].add_(g_flow @ g_flow.T)
                if state["R_acc"] is not None:
                    state["R_acc"].add_(g_flow.T @ g_flow)
            if t >= group["warm_start_steps"]:
                self._warm_start(state)
                # This gradient is already part of the pooled estimate: express it in the new
                # frame for the variance estimate and skip this step's frame move (Remark 5.6).
                z_flow = rotate(g_flow, state["QL"], state["QR"])
                restarted = True

        # Variances seen by the flow, then one natural-gradient step of the likelihood.
        flow_beta = group["flow_beta"]
        if flow_beta is None:
            d = v_hat + (group["floor"] * v_hat.mean() + torch.finfo(v_hat.dtype).tiny)
        else:
            beta_d = 1.0 - group["rot_rate"] if flow_beta == "tied" else float(flow_beta)
            vf = state.setdefault("VF", torch.zeros_like(g))
            vf.mul_(beta_d).addcmul_(z_flow, z_flow, value=1 - beta_d)
            w_full = 1 - beta_d**t
            if group["flow_shrink"]:
                # Interleaved split: VF_odd averages the odd steps, VF - VF_odd the even ones.
                vf_odd = state.setdefault("VF_odd", torch.zeros_like(g))
                vf_odd.mul_(beta_d)
                w_odd = state.get("w_odd", 0.0) * beta_d
                if t % 2 == 1:
                    vf_odd.addcmul_(z_flow, z_flow, value=1 - beta_d)
                    w_odd += 1 - beta_d
                state["w_odd"] = w_odd
                d = self._shrunk_variances(vf, vf_odd, w_full, w_odd, group["floor"])
            else:
                v_flow = vf / w_full
                d = v_flow + (group["floor"] * v_flow.mean() + torch.finfo(v_flow.dtype).tiny)
        if not restarted:
            self._flow(state, z_flow, d, group)

    @staticmethod
    def _mean_shrinkage(m_buf: torch.Tensor, v_buf: torch.Tensor, b1: float, b2: float,
                        t: int) -> torch.Tensor:
        """Empirical-Bayes factor of the previous momentum as an estimate of the mean (Prop. 5.7).

        The bias-corrected momentum over ``t - 1`` gradients carries sampling noise ``eta`` times
        the total gradient variance, ``eta`` being the sum of its squared normalized weights
        (Lean: ``ema_weight_sq_sum``). With ``S_M = ||M||^2`` and ``S_V = sum V`` (Adam's
        uncentered second moment, whose sum does not depend on the frame), the total variance is
        about ``(S_V - S_M) / (1 - eta)``, and ``c = (1 - noise / S_M)_+`` is the plug-in of the
        risk-optimal factor ``S / (S + noise)`` (Lean: ``shrinkage_risk_eq_iff``).
        """
        eta = (1 - b1) * (1 + b1 ** (t - 1)) / ((1 + b1) * (1 - b1 ** (t - 1)))
        if 1 - eta < 1e-6:
            # One gradient (t = 2) or no momentum (b1 = 0): mean and noise are not separable.
            return torch.zeros((), dtype=m_buf.dtype, device=m_buf.device)
        s_m = m_buf.square().sum() / (1 - b1 ** (t - 1)) ** 2
        s_v = v_buf.sum() / (1 - b2 ** (t - 1))
        noise = eta * (s_v - s_m).clamp_min(0.0) / (1 - eta)
        return (1 - noise / s_m.clamp_min(torch.finfo(s_m.dtype).tiny)).clamp(0.0, 1.0)

    @staticmethod
    def _shrunk_variances(vf: torch.Tensor, vf_odd: torch.Tensor, w_full: float, w_odd: float,
                          floor: float) -> torch.Tensor:
        """Empirical-Bayes variance estimate for the flow (Proposition 5.5).

        ``log D`` is split into its additive fit ``r_i + c_j`` (the separable, Kronecker-product
        model) and a residual. The residual is multiplied by ``(1 - noise / mean(residual^2))_+``,
        where ``noise`` is the variance of ``log V`` that sampling alone produces after the
        additive fit; it is measured from the log-ratio of the odd-step and even-step averages,
        whose variance is about four times that of the full average. Separable arrays are thus
        estimated from row and column means, non-separable ones keep their interaction.
        """
        m, n = vf.shape
        v_hat = vf / w_full
        # The smallest normal number guards against log(0) without breaking scale invariance
        # (an absolute constant such as 1e-30 would dominate gradients below ~1e-15; F-014).
        fl = floor * v_hat.mean() + torch.finfo(v_hat.dtype).tiny
        log_v = (v_hat + fl).log()
        row = log_v.mean(dim=1, keepdim=True)
        col = log_v.mean(dim=0, keepdim=True)
        additive = row + col - log_v.mean()
        resid = log_v - additive
        w_even = w_full - w_odd
        if w_odd > 0.0 and w_even > 1e-12 * w_full:
            ratio = (vf_odd / w_odd + fl).log() - ((vf - vf_odd).clamp_min(0.0) / w_even + fl).log()
            noise = ratio.var(unbiased=False) * (0.25 * (1 - 1 / m) * (1 - 1 / n))
            shrink = (1 - noise / resid.square().mean().clamp_min(1e-30)).clamp(0.0, 1.0)
        else:
            shrink = torch.zeros((), dtype=vf.dtype, device=vf.device)
        d = (additive + shrink * resid).exp()
        return d.mul_((v_hat + fl).mean() / d.mean())

    @staticmethod
    def _warm_start(state: dict) -> None:
        """Replace the frame by the pooled-factor eigenvectors and transport the moments."""
        moves = {}
        for key, acc_key in (("QL", "L_acc"), ("QR", "R_acc")):
            if state[acc_key] is not None and state[key] is not None:
                new = eigh_desc(state[acc_key])[0]
                moves[key] = state[key].T @ new  # old -> new change of frame
                state[key] = new
        for buf_key in ("V", "VF", "VF_odd"):
            buf = state.get(buf_key)
            if buf is None:
                continue
            if "QL" in moves:
                buf.copy_((moves["QL"] * moves["QL"]).T @ buf)
            if "QR" in moves:
                buf.copy_(buf @ (moves["QR"] * moves["QR"]))
        state.pop("L_acc")
        state.pop("R_acc")
        state.pop("flow_acc", None)

    @staticmethod
    def _score_and_fisher(
        z: torch.Tensor, d: torch.Tensor, a: torch.Tensor, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Skew score E (Lemma A) and Fisher information F (Theorem 2) for one side."""
        if side == "left":
            groups = z.shape[1]
            s = z @ (z * a).T
            fisher = d @ a.T
        else:
            groups = z.shape[0]
            s = z.T @ (z * a)
            fisher = d.T @ a
        fisher = (fisher + fisher.T - 2.0 * groups).clamp_min(0.0)
        return s - s.T, fisher, groups

    @staticmethod
    def _generator(score: torch.Tensor, fisher: torch.Tensor, groups: int, rate: float,
                   group: dict) -> torch.Tensor:
        """Damped natural-gradient generator (skew-symmetric) with the trust-region caps."""
        omega = score.mul(-rate).div_(fisher + group["damping"] * groups)
        omega.fill_diagonal_(0.0)
        omega.clamp_(-group["max_angle"], group["max_angle"])
        # Spectral trust region: with ||Ω||_2 <= 1 the retraction's singular values stay in
        # [1, 1.12], inside the convergence region of the Newton–Schulz polish.
        norm = skew_spectral_norm(omega)
        if norm > group["max_rotation"]:
            omega.mul_(group["max_rotation"] / norm)
        return omega

    def _flow(self, state: dict, z: torch.Tensor, d: torch.Tensor, group: dict) -> None:
        """Accumulate the likelihood score; every ``frame_every`` steps move the frame.

        With ``frame_every = k`` the frame takes one natural-gradient step per k gradients using
        the mean score of all k (no sample is wasted) and the effective rate
        ``1 - prod_s (1 - alpha_s)``, which equals the per-step rate for k = 1. Retraction,
        polish and the Fisher matrix are then paid once per k steps.
        """
        alpha = group["rot_rate"]
        if group["rot_schedule"] == "bias_corrected" and alpha > 0:
            alpha = min(group["rot_rate_max"], alpha / (1 - (1 - alpha) ** state["step"]))
        a = d.reciprocal()
        sides = [(k, s) for k, s in (("QL", "left"), ("QR", "right")) if state[k] is not None]
        acc = state.setdefault("flow_acc", {"keep": 1.0, "count": 0})
        acc["keep"] *= 1.0 - alpha
        acc["count"] += 1
        fishers = {}
        for key, side in sides:
            score, fisher, groups = self._score_and_fisher(z, d, a, side)
            acc[key] = score if acc["count"] == 1 else acc[key] + score
            fishers[key] = (fisher, groups)
        k = group["frame_every"]
        if group["frame_schedule"] == "adaptive" and k > 1 and alpha > 0:
            k = int(min(k, max(1, round(k * group["rot_rate"] / alpha))))
        if acc["count"] < k:
            return
        rate = 1.0 - acc["keep"]
        moves = {}
        for key, _ in sides:
            fisher, groups = fishers[key]
            omega = self._generator(acc[key] / acc["count"], fisher, groups, rate, group)
            moves[key] = expm2(omega)
            state[key] = state[key] @ moves[key]
        state["flow_acc"] = {"keep": 1.0, "count": 0}
        if group["transport"]:
            # Rows of P o P sum to one for an orthogonal P; renormalizing them makes the transport
            # preserve the total second moment exactly despite the retraction's tiny defect.
            maps = {}
            for key in moves:
                sq = moves[key] * moves[key]
                maps[key] = sq / sq.sum(dim=1, keepdim=True)
            for buf_key in ("V", "VF", "VF_odd"):
                buf = state.get(buf_key)
                if buf is None:
                    continue
                if "QL" in maps:
                    buf.copy_(maps["QL"].T @ buf)
                if "QR" in maps:
                    buf.copy_(buf @ maps["QR"])
        state["frame_moves"] = state.get("frame_moves", 0) + 1
        if state["frame_moves"] % group["polish_every"] == 0:
            for key, _ in sides:
                state[key] = polish_until_orthogonal(state[key])
