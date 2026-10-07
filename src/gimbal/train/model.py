"""Llama-style decoder for the TPU study (Phase 05 specification), in plain JAX.

12 layers, d = 768, 12 heads, SwiGLU hidden 2048, RoPE (θ = 10⁴), pre-norm RMSNorm, no biases,
tied input/output embeddings, vocabulary padded to 50304, context 1024. float32 master weights,
bfloat16 compute.

Parameters are stored in the layout the optimizers use (Phase 04 shape bucketing): the four
attention matrices of every layer form one ``[L, 4, d, d]`` stack (q, k, v, o) and the three MLP
matrices one ``[L, 3, d, f]`` stack (gate, up, and the down projection stored transposed). All
hidden matrices therefore fall into two buckets, ``48 × (768 × 768)`` and ``36 × (768 × 2048)``.
Storing the down projection transposed does not change any optimizer: AdamW is elementwise and
SOAP and Gimbal are transpose-equivariant (their left and right rules are mirror images).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

COMPUTE = jnp.bfloat16


@dataclass(frozen=True)
class ModelConfig:
    """Llama-style decoder: RMSNorm pre-norm, rotary embeddings, SwiGLU MLP, tied embeddings."""

    vocab: int = 50304
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 2048
    seq_len: int = 1024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    attention: str = "flash"  # "flash" (Pallas TPU kernel), "splash" or "xla"
    flash_block_q: int = 0  # Pallas flash-attention tiles (0: the kernel's defaults)
    flash_block_k: int = 0
    flash_block_b: int = 1
    scan_layers: bool = False  # unrolled layers: backward 14% faster than lax.scan (Phase 04)

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.d_model // self.n_heads


def init_params(cfg: ModelConfig, key: jax.Array) -> dict:
    """N(0, 0.02) initialization; output projections (o, down) scaled by 1/sqrt(2L)."""
    d, f, L = cfg.d_model, cfg.d_ff, cfg.n_layers
    k_emb, k_attn, k_mlp = jax.random.split(key, 3)
    std, out_scale = cfg.init_std, 1.0 / math.sqrt(2 * L)
    attn = std * jax.random.normal(k_attn, (L, 4, d, d), jnp.float32)
    attn = attn.at[:, 3].multiply(out_scale)
    mlp = std * jax.random.normal(k_mlp, (L, 3, d, f), jnp.float32)
    mlp = mlp.at[:, 2].multiply(out_scale)
    return {
        "embed": std * jax.random.normal(k_emb, (cfg.vocab, d), jnp.float32),
        "attn": attn,
        "mlp": mlp,
        "norm_attn": jnp.ones((L, d), jnp.float32),
        "norm_mlp": jnp.ones((L, d), jnp.float32),
        "norm_f": jnp.ones((d,), jnp.float32),
    }


def count_params(params: dict) -> int:
    """Number of scalar parameters."""
    return sum(int(x.size) for x in jax.tree.leaves(params))


def _rms_norm(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
    xf = x.astype(jnp.float32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * w).astype(COMPUTE)


def _rope_tables(cfg: ModelConfig) -> tuple[jax.Array, jax.Array]:
    half = cfg.head_dim // 2
    inv = 1.0 / (cfg.rope_theta ** (jnp.arange(half, dtype=jnp.float32) / half))
    ang = jnp.arange(cfg.seq_len, dtype=jnp.float32)[:, None] * inv[None, :]
    return jnp.cos(ang), jnp.sin(ang)  # [T, half]


def _rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotary embedding on ``[B, H, T, Dh]`` (rotate-half convention)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].astype(jnp.float32), x[..., half:].astype(jnp.float32)
    out = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return out.astype(COMPUTE)


def _xla_attention(q: jax.Array, k: jax.Array, v: jax.Array, scale: float) -> jax.Array:
    t = q.shape[2]
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * scale
    mask = jnp.tril(jnp.ones((t, t), bool))
    s = jnp.where(mask, s, -1e30)
    p = jax.nn.softmax(s, axis=-1).astype(COMPUTE)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v)


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kind: str,
    scale: float,
    blocks: tuple[int, int, int] = (0, 0, 1),
) -> jax.Array:
    """Causal attention on ``[B, H, T, Dh]`` with the chosen implementation."""
    if kind == "xla":
        return _xla_attention(q, k, v, scale)
    if kind == "flash":
        from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, flash_attention

        bq, bk, bb = blocks
        sizes = None
        if bq:
            sizes = BlockSizes(
                block_q=bq,
                block_k_major=bk,
                block_k=bk,
                block_b=bb,
                block_q_major_dkv=bq,
                block_k_major_dkv=bk,
                block_k_dkv=bk,
                block_q_dkv=bq,
                block_k_major_dq=bk,
                block_k_dq=bk,
                block_q_dq=bq,
            )
        return flash_attention(q, k, v, causal=True, sm_scale=scale, block_sizes=sizes)
    if kind == "splash":
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel as sk,
        )
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_mask as sm,
        )

        h, t = q.shape[1], q.shape[2]
        mask = sm.MultiHeadMask([sm.CausalMask((t, t)) for _ in range(h)])
        kernel = sk.make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1)
        return jax.vmap(kernel)((q * scale).astype(q.dtype), k, v)
    raise ValueError(kind)


def forward(params: dict, tokens: jax.Array, cfg: ModelConfig) -> jax.Array:
    """Logits (float32) for ``tokens`` of shape ``[B, T]``."""
    x = forward_hidden(params, tokens, cfg)
    return jnp.einsum(
        "btd,vd->btv", x, params["embed"].astype(COMPUTE), preferred_element_type=jnp.float32
    )


def forward_hidden(params: dict, tokens: jax.Array, cfg: ModelConfig) -> jax.Array:
    """Final normalized hidden state ``[B, T, d]`` (bfloat16)."""
    b, t = tokens.shape
    h, dh = cfg.n_heads, cfg.head_dim
    cos, sin = _rope_tables(cfg)
    cos, sin = cos[:t], sin[:t]
    x = params["embed"].astype(COMPUTE)[tokens]
    scale = 1.0 / math.sqrt(dh)

    def layer(x, w):
        attn_w, mlp_w, n_attn, n_mlp = w
        a = attn_w.astype(COMPUTE)
        y = _rms_norm(x, n_attn, cfg.norm_eps)
        q, k, v = (jnp.einsum("btd,de->bte", y, a[i]) for i in range(3))
        q, k, v = (z.reshape(b, t, h, dh).transpose(0, 2, 1, 3) for z in (q, k, v))
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        o = attention(
            q, k, v, cfg.attention, scale, (cfg.flash_block_q, cfg.flash_block_k, cfg.flash_block_b)
        )
        o = o.transpose(0, 2, 1, 3).reshape(b, t, h * dh)
        x = x + jnp.einsum("btd,de->bte", o, a[3])
        m = mlp_w.astype(COMPUTE)
        y = _rms_norm(x, n_mlp, cfg.norm_eps)
        gate = jnp.einsum("btd,df->btf", y, m[0])
        up = jnp.einsum("btd,df->btf", y, m[1])
        x = x + jnp.einsum("btf,df->btd", jax.nn.silu(gate) * up, m[2])
        return x, None

    stacks = (params["attn"], params["mlp"], params["norm_attn"], params["norm_mlp"])
    if cfg.scan_layers:
        x, _ = jax.lax.scan(layer, x, stacks)
    else:
        for i in range(cfg.n_layers):
            x, _ = layer(x, tuple(w[i] for w in stacks))
    return _rms_norm(x, params["norm_f"], cfg.norm_eps)


def loss_fn(params: dict, batch: jax.Array, cfg: ModelConfig) -> jax.Array:
    """Mean next-token cross-entropy; ``batch`` is ``[B, T + 1]`` token ids."""
    logits = forward(params, batch[:, :-1], cfg)
    targets = batch[:, 1:]
    logz = jax.nn.logsumexp(logits, axis=-1)
    gold = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return jnp.mean(logz - gold)


def token_losses(params: dict, batch: jax.Array, cfg: ModelConfig) -> jax.Array:
    """Per-sequence summed losses (for exact evaluation means and bootstrap over batches)."""
    logits = forward(params, batch[:, :-1], cfg)
    targets = batch[:, 1:]
    logz = jax.nn.logsumexp(logits, axis=-1)
    gold = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return jnp.sum(logz - gold, axis=-1)
