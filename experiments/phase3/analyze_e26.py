"""E3.1 (formerly E2.6): premise check on real language-model gradients (Phase 03, gate G3.4).

Input: gradient snapshots written by ``e27_small_lm.py --snapshots ...`` (K independent minibatch
gradients per matrix at fixed weights). For every matrix the samples are split in half:

* fit half: estimate frames — SOAP's pooled frame (exact eigenvectors of sum GG^T and G^T G, the
  best case for SOAP), the KL (flip-flop) frame, the Gimbal frame (batch maximum likelihood of the
  KRD model by Fisher scoring, started from the pooled frame), and the identity (AdamW);
* held-out half: score each frame by L(U) = 1/2 sum_ij log mean_k (Q_L^T G_k Q_R)_ij^2.
  Differences of L between frames are differences of the frame KL J (Proposition 1) up to
  estimation noise, in nats.

Also reports the non-separability index kappa of D in the pooled frame (noise-corrected).

Usage: python experiments/phase3/analyze_e26.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase2"))
from common import nonseparability_index  # noqa: E402  (shared helpers in experiments/phase2)

RESULTS = pathlib.Path(__file__).parent / "results"


def einsum(spec: str, *ops: np.ndarray) -> np.ndarray:
    """np.einsum with an optimized contraction order: the three-operand products here would
    otherwise run as one naive loop over every index (hours instead of seconds)."""
    return np.einsum(spec, *ops, optimize=True)


def eigh_desc(s):
    w, v = np.linalg.eigh(0.5 * (s + s.T))
    return v[:, ::-1]


def heldout_score(ql, qr, g_eval):
    z = einsum("ia,kij,jb->kab", ql, g_eval, qr)
    d = (z * z).mean(axis=0)
    return 0.5 * float(np.sum(np.log(d + 1e-300)))


def pooled_frame(g):
    return eigh_desc(einsum("kij,klj->il", g, g)), eigh_desc(einsum("kji,kjl->il", g, g))


def kl_frame(g, iters=15, damp=1e-4):
    k, m, n = g.shape
    lf = einsum("kij,klj->il", g, g) / (n * k)
    rf = einsum("kji,kjl->il", g, g) / (m * k)
    for _ in range(iters):
        wr, vr = np.linalg.eigh(rf)
        wr = wr.clip(min=0) + damp * wr.clip(min=0).mean() + 1e-300
        r_inv = (vr / wr) @ vr.T
        lf = einsum("kij,jl,kml->im", g, r_inv, g) / (n * k)
        wl, vl = np.linalg.eigh(lf)
        wl = wl.clip(min=0) + damp * wl.clip(min=0).mean() + 1e-300
        l_inv = (vl / wl) @ vl.T
        rf = einsum("kji,jl,klm->im", g, l_inv, g) / (m * k)
    return eigh_desc(lf), eigh_desc(rf)


def polish(q):
    for _ in range(4):
        gram = q.T @ q
        if np.abs(gram - np.eye(len(gram))).max() < 1e-12:
            break
        q = q @ (1.5 * np.eye(len(gram)) - 0.5 * gram)
    return q


def eb_variances(z, floor=1e-8):
    """Batch analogue of Gimbal's empirical-Bayes variances (Proposition 5.5): log of the sample
    variance shrunk toward its additive fit, noise level from an odd/even split of the samples."""
    k, m, n = z.shape
    v = (z * z).mean(axis=0)
    fl = floor * v.mean() + 1e-300
    log_v = np.log(v + fl)
    additive = log_v.mean(1, keepdims=True) + log_v.mean(0, keepdims=True) - log_v.mean()
    resid = log_v - additive
    ratio = np.log((z[0::2] ** 2).mean(0) + fl) - np.log((z[1::2] ** 2).mean(0) + fl)
    noise = ratio.var() * 0.25 * (1 - 1 / m) * (1 - 1 / n)
    c = float(np.clip(1 - noise / max((resid**2).mean(), 1e-300), 0.0, 1.0))
    d = np.exp(additive + c * resid)
    return d * (v + fl).mean() / d.mean(), c


def gimbal_mle_frame(g, ql, qr, iters=60, step=0.5, damping=0.003, floor=1e-8, shrink=False):
    """Batch maximum likelihood of the KRD frame by damped Fisher scoring (the batch analogue
    of Gimbal's online flow: same score, same Fisher normalizer). With ``shrink`` the variances
    are the empirical-Bayes estimate of Proposition 5.5, as in the optimizer."""
    k, m, n = g.shape
    for _ in range(iters):
        z = einsum("ia,kij,jb->kab", ql, g, qr)
        if shrink:
            d = eb_variances(z, floor)[0]
        else:
            d = (z * z).mean(axis=0)
            d = d + floor * d.mean() + 1e-300
        a = 1.0 / d
        s_l = einsum("kij,klj->il", z, z * a) / k
        e_l = s_l - s_l.T
        f_l = d @ a.T
        f_l = (f_l + f_l.T - 2 * n).clip(min=0)
        om_l = -step * e_l / (f_l + damping * n)
        np.fill_diagonal(om_l, 0)
        s_r = einsum("kji,kjl->il", z, z * a) / k
        e_r = s_r - s_r.T
        f_r = d.T @ a
        f_r = (f_r + f_r.T - 2 * m).clip(min=0)
        om_r = -step * e_r / (f_r + damping * m)
        np.fill_diagonal(om_r, 0)
        for om in (om_l, om_r):
            nrm = np.linalg.norm(om, 2)
            if nrm > 0.5:
                om *= 0.5 / nrm
        ql = polish(ql @ (np.eye(m) + om_l + 0.5 * om_l @ om_l))
        qr = polish(qr @ (np.eye(n) + om_r + 0.5 * om_r @ om_r))
    return ql, qr


def analyze_file(path: pathlib.Path) -> list[dict]:
    data = np.load(path)
    m = re.search(r"snap_(.+)_step(\d+)\.npz", path.name)
    run_id, step = m.group(1), int(m.group(2))
    rows = []
    for name in data.files:
        g = data[name]
        k = g.shape[0] // 2
        g_fit, g_eval = g[:k], g[k:]
        ql_p, qr_p = pooled_frame(g_fit)
        ql_k, qr_k = kl_frame(g_fit)
        ql_g, qr_g = gimbal_mle_frame(g_fit, ql_p.copy(), qr_p.copy())
        ql_e, qr_e = gimbal_mle_frame(g_fit, ql_p.copy(), qr_p.copy(), shrink=True)
        eye_l, eye_r = np.eye(g.shape[1]), np.eye(g.shape[2])
        scores = {
            "identity": heldout_score(eye_l, eye_r, g_eval),
            "soap_pooled": heldout_score(ql_p, qr_p, g_eval),
            "kl": heldout_score(ql_k, qr_k, g_eval),
            "gimbal_mle": heldout_score(ql_g, qr_g, g_eval),
            "gimbal_eb": heldout_score(ql_e, qr_e, g_eval),
        }
        z = einsum("ia,kij,jb->kab", ql_p, g_fit, qr_p)
        d_pooled = (z * z).mean(axis=0) + 1e-300
        rows.append(
            {
                "run": run_id,
                "step": step,
                "matrix": name,
                "shape": list(g.shape[1:]),
                "samples_fit": k,
                "kappa_pooled_frame": nonseparability_index(d_pooled, k),
                **{f"L_{key}": v for key, v in scores.items()},
                "shrink_factor_pooled_frame": eb_variances(z)[1],
                "gain_gimbal_vs_soap_nats": scores["soap_pooled"] - scores["gimbal_eb"],
                "gain_gimbal_vs_kl_nats": scores["kl"] - scores["gimbal_eb"],
                "gain_free_mle_vs_soap_nats": scores["soap_pooled"] - scores["gimbal_mle"],
                "gain_soap_vs_identity_nats": scores["identity"] - scores["soap_pooled"],
            }
        )
    return rows


def main() -> None:
    files = sorted((RESULTS / "lm").glob("snap_*.npz"))
    rows = [r for f in files for r in analyze_file(f)]
    (RESULTS / "e26_real_gradients.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "# E3.1 real-gradient premise check (generated by analyze_e26.py)",
        "",
        "Gimbal = batch analogue with empirical-Bayes variances (as in the optimizer); "
        "free MLE = without shrinkage. Gains are held-out log-likelihood differences in nats "
        "(positive = first frame better).",
        "",
        "| run | step | matrix | κ (pooled frame) | shrink c | SOAP gain over identity | "
        "Gimbal gain over SOAP | Gimbal gain over KL | free MLE gain over SOAP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['step']} | {r['matrix']} | {r['kappa_pooled_frame']:.3f} |"
            f" {r['shrink_factor_pooled_frame']:.2f} |"
            f" {r['gain_soap_vs_identity_nats']:.1f} | {r['gain_gimbal_vs_soap_nats']:.1f} |"
            f" {r['gain_gimbal_vs_kl_nats']:.1f} | {r['gain_free_mle_vs_soap_nats']:.1f} |"
        )
    if rows:
        g = np.array([r["gain_gimbal_vs_soap_nats"] for r in rows])
        kap = np.array([r["kappa_pooled_frame"] for r in rows])
        lines += [
            "",
            f"Matrices: {len(rows)}; Gimbal frame better than SOAP's on held-out "
            f"gradients in {int((g > 0).sum())}/{len(rows)}; median gain {np.median(g):.1f} "
            f"nats; median κ {np.median(kap):.3f}.",
        ]
    (RESULTS / "e26_report.md").write_text("\n".join(lines))
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
