"""Correctness tests for the reference optimizers (phases/03_reference_implementations.md).

The invariants come from theory/gimbal_theory.md: orthogonality (Theorem 5), equivariance
(Theorem 6), transport (Theorem 7), descent and scale invariance (Theorem 8).
"""

from __future__ import annotations

import pytest
import torch

from gimbal.torch import KLSOAP, SOAP, AROSinkhorn, Gimbal, Muon, NorMuon, SPlus
from gimbal.torch._linalg import expm2, ns_polish, zeropower_ns5

from .third_party.soap_official import SOAP as OfficialSOAP

torch.set_default_dtype(torch.float64)


def random_orthogonal(n: int, gen: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q * torch.sign(torch.diag(r))


def run_steps(opt_cls, grads, shape=(6, 4), w0=None, **kw):
    w = torch.nn.Parameter(torch.zeros(shape) if w0 is None else w0.clone())
    opt = opt_cls([w], **kw)
    for g in grads:
        w.grad = g.clone()
        opt.step()
    return w.detach(), opt


@pytest.fixture
def grads():
    gen = torch.Generator().manual_seed(0)
    scales = torch.logspace(0, -2, 6)[:, None] * torch.logspace(0, -1, 4)[None, :]
    return [scales * torch.randn(6, 4, generator=gen) for _ in range(40)]


def test_soap_matches_official(grads):
    # The official code runs eigh/QR in float32 and allocates with the default dtype.
    torch.set_default_dtype(torch.float32)
    try:
        grads32 = [g.float() for g in grads]
        kw = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.01, precondition_frequency=3)
        ours, _ = run_steps(SOAP, grads32, **kw)
        ref, _ = run_steps(OfficialSOAP, grads32, **kw)
    finally:
        torch.set_default_dtype(torch.float64)
    assert torch.allclose(ours, ref, rtol=1e-4, atol=1e-6)


def test_gimbal_frames_stay_orthogonal():
    gen = torch.Generator().manual_seed(1)
    w = torch.nn.Parameter(torch.zeros(12, 9))
    opt = Gimbal([w], lr=1e-3, rot_rate=0.3)
    for _ in range(2000):
        w.grad = torch.randn(12, 9, generator=gen) * torch.linspace(0.1, 3, 9)
        opt.step()
    st = opt.state[w]
    for q in (st["QL"], st["QR"]):
        assert torch.linalg.norm(q.T @ q - torch.eye(q.shape[0])) < 1e-8


def test_gimbal_equivariance():
    # Square, full-rank gradients so that the eigh initialization is unique up to signs; the
    # update is invariant to the sign (gauge) of each frame vector.
    gen = torch.Generator().manual_seed(2)
    grads = [torch.randn(5, 5, generator=gen) * torch.linspace(0.2, 2, 5) for _ in range(40)]
    p_left, p_right = random_orthogonal(5, gen), random_orthogonal(5, gen)
    w0 = torch.randn(5, 5, generator=gen)
    kw = dict(lr=1e-2, weight_decay=0.1, rot_rate=0.2, transport=True)
    w_a, _ = run_steps(Gimbal, grads, shape=(5, 5), w0=w0, **kw)
    w_b, _ = run_steps(Gimbal, [p_left @ g @ p_right.T for g in grads], shape=(5, 5),
                       w0=p_left @ w0 @ p_right.T, **kw)
    assert torch.allclose(w_b, p_left @ w_a @ p_right.T, atol=1e-8)


def test_adamw_is_not_equivariant(grads):
    gen = torch.Generator().manual_seed(2)
    p_left, p_right = random_orthogonal(6, gen), random_orthogonal(4, gen)
    w0 = torch.randn(6, 4, generator=gen)
    w_a, _ = run_steps(torch.optim.AdamW, grads, w0=w0, lr=1e-2)
    w_b, _ = run_steps(torch.optim.AdamW, [p_left @ g @ p_right.T for g in grads],
                       w0=p_left @ w0 @ p_right.T, lr=1e-2)
    assert not torch.allclose(w_b, p_left @ w_a @ p_right.T, atol=1e-4)


def test_gimbal_frame_scale_invariance(grads):
    _, opt_a = run_steps(Gimbal, grads, rot_rate=0.2, init="identity")
    _, opt_b = run_steps(Gimbal, [1e3 * g for g in grads], rot_rate=0.2, init="identity")
    sa = next(iter(opt_a.state.values()))
    sb = next(iter(opt_b.state.values()))
    assert torch.allclose(sa["QL"], sb["QL"], atol=1e-9)
    assert torch.allclose(sa["QR"], sb["QR"], atol=1e-9)


def test_gimbal_descent_identity(grads):
    w = torch.nn.Parameter(torch.zeros(6, 4))
    opt = Gimbal([w], lr=1.0, betas=(0.0, 0.95))
    for g in grads[:5]:
        w.grad = g.clone()
        before = w.detach().clone()
        opt.step()
        update = before - w.detach()  # lr = 1, no weight decay
        assert torch.sum(g * update) > 0


def test_transport_preserves_total_second_moment():
    gen = torch.Generator().manual_seed(3)
    w = torch.nn.Parameter(torch.zeros(5, 7))
    opt = Gimbal([w], rot_rate=0.3, transport=True)
    for _ in range(50):
        w.grad = torch.randn(5, 7, generator=gen)
        opt.step()
    st = opt.state[w]
    b2 = 0.95
    # Without transport the total would follow the EMA of squared rotated gradients; with an
    # orthogonal frame change the doubly-stochastic transport keeps the total unchanged, so the
    # total equals the EMA of squared Frobenius norms (rotations preserve them).
    gen = torch.Generator().manual_seed(3)
    total = 0.0
    for _ in range(50):
        g = torch.randn(5, 7, generator=gen)
        total = b2 * total + (1 - b2) * float((g * g).sum())
    assert abs(float(st["V"].sum()) - total) / total < 1e-6


def test_retraction_defect_is_fourth_order():
    gen = torch.Generator().manual_seed(4)
    a = torch.randn(8, 8, generator=gen) * 0.1
    omega = a - a.T
    x = expm2(omega)
    defect = x.T @ x - torch.eye(8)
    assert torch.allclose(defect, 0.25 * torch.linalg.matrix_power(omega, 4), atol=1e-12)
    y = ns_polish(x)
    e = defect
    expected = -0.75 * e @ e + 0.25 * e @ e @ e
    assert torch.allclose(y.T @ y - torch.eye(8), expected, atol=1e-12)


def test_newton_schulz_is_nearly_orthogonal():
    gen = torch.Generator().manual_seed(5)
    o = zeropower_ns5(torch.randn(16, 8, generator=gen))
    s = torch.linalg.svdvals(o)
    assert s.min() > 0.5 and s.max() < 1.3


@pytest.mark.parametrize(
    "cls, kw",
    [
        (Gimbal, dict(lr=3e-2)),
        (SOAP, dict(lr=3e-2)),
        (SOAP, dict(lr=3e-2, realtime=True)),
        (KLSOAP, dict(lr=3e-2)),
        (Muon, dict(lr=3e-2)),
        (NorMuon, dict(lr=3e-2)),
        (SPlus, dict(lr=3e-1)),
        (AROSinkhorn, dict(lr=3e-2)),
    ],
)
def test_reduces_quadratic(cls, kw):
    gen = torch.Generator().manual_seed(6)
    target = torch.randn(10, 6, generator=gen)
    w = torch.nn.Parameter(torch.zeros(10, 6))
    opt = cls([w], **kw)
    start = float(((w.detach() - target) ** 2).sum())
    for _ in range(200):
        w.grad = 2 * (w.detach() - target)
        opt.step()
    assert float(((w.detach() - target) ** 2).sum()) < 0.2 * start
