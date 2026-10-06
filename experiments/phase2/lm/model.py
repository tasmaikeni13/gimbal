"""A small Llama-style decoder for CPU-scale optimizer comparisons (Phase 02, E2.6–E2.7).

Byte-level vocabulary (256), pre-norm RMSNorm, rotary position embeddings, SwiGLU MLP, no
biases. ``hidden_matrices`` returns the 2-D weights that matrix optimizers handle; everything
else (embeddings, output head, norms) goes to AdamW, identically for every method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_hidden: int = 352
    seq_len: int = 128


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads = h
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.w_gate = nn.Linear(d, cfg.mlp_hidden, bias=False)
        self.w_up = nn.Linear(d, cfg.mlp_hidden, bias=False)
        self.w_down = nn.Linear(cfg.mlp_hidden, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        hd = d // self.n_heads
        y = self.norm1(x)
        q = self.wq(y).view(b, t, self.n_heads, hd).transpose(1, 2)
        k = self.wk(y).view(b, t, self.n_heads, hd).transpose(1, 2)
        v = self.wv(y).view(b, t, self.n_heads, hd).transpose(1, 2)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(att.transpose(1, 2).reshape(b, t, d))
        y = self.norm2(x)
        return x + self.w_down(F.silu(self.w_gate(y)) * self.w_up(y))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        hd = cfg.d_model // cfg.n_heads
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(cfg.seq_len).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02
                if name.endswith(("wo.weight", "w_down.weight")):
                    std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
                nn.init.normal_(p, std=std)

    def hidden_matrices(self) -> list[torch.Tensor]:
        return [p for n, p in self.named_parameters() if p.ndim == 2 and n.startswith("blocks.")]

    def other_parameters(self) -> list[torch.Tensor]:
        hidden = {id(p) for p in self.hidden_matrices()}
        return [p for p in self.parameters() if id(p) not in hidden]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        t = idx.shape[1]
        x = self.embed(idx)
        cos, sin = self.cos[:t], self.sin[:t]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
