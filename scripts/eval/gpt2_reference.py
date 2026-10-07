"""Reference point for gate G5.2: the OpenAI GPT-2 (124M) checkpoint on our validation split.

Karpathy's build-nanogpt (commit 6104ab1, ``play.ipynb``) reports a validation loss of 3.2924 for
this checkpoint on its FineWeb-Edu (sample-10BT) validation shard. Scoring the same public weights
on our validation split checks tokenization, data and loss computation end to end against a
published number, and gives the scale against which our AdamW runs are judged.

The weights are read from the Hugging Face ``gpt2`` repository at a pinned revision; the
safetensors file is parsed directly (8-byte header length, JSON header, raw little-endian arrays),
so no extra package is needed. GPT-2: learned positions, LayerNorm, GELU (tanh) MLP, biases, tied
output embedding. PyTorch on the CPU, float32.

Usage: python scripts/eval/gpt2_reference.py --seqs 5120 --threads 64
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[2]
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    raw = pathlib.Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + n])
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32", info["dtype"]
        lo, hi = info["data_offsets"]
        arr = np.frombuffer(raw[base + lo : base + hi], dtype="<f4").reshape(info["shape"])
        out[name] = torch.from_numpy(arr.copy())
    return out


def gpt2_loss(w: dict, tokens: torch.Tensor, n_layer: int = 12, n_head: int = 12) -> torch.Tensor:
    """Per-sequence summed next-token loss of ``tokens`` ``[B, T + 1]``."""
    x_in, y = tokens[:, :-1], tokens[:, 1:]
    b, t = x_in.shape
    d = w["wte.weight"].shape[1]
    x = w["wte.weight"][x_in] + w["wpe.weight"][:t]
    mask = torch.full((t, t), float("-inf")).triu(1)
    for i in range(n_layer):
        p = f"h.{i}."
        h = F.layer_norm(x, (d,), w[p + "ln_1.weight"], w[p + "ln_1.bias"], 1e-5)
        qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]  # Conv1D: [d, 3d]
        q, k, v = qkv.split(d, dim=-1)
        q, k, v = (z.view(b, t, n_head, d // n_head).transpose(1, 2) for z in (q, k, v))
        a = (q @ k.transpose(-1, -2)) / math.sqrt(d // n_head) + mask
        o = (a.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + o @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        h = F.layer_norm(x, (d,), w[p + "ln_2.weight"], w[p + "ln_2.bias"], 1e-5)
        h = F.gelu(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"], approximate="tanh")
        x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
    x = F.layer_norm(x, (d,), w["ln_f.weight"], w["ln_f.bias"], 1e-5)
    logits = x @ w["wte.weight"].T
    return (
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
        .view(b, t)
        .sum(-1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", type=int, default=5120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    w = load_safetensors(hf_hub_download("gpt2", "model.safetensors", revision=REVISION))
    val = np.memmap(ROOT / "data" / "tokens" / "val.bin", dtype=np.uint16, mode="r")
    losses = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, args.seqs, args.batch):
            idx = range(s, min(s + args.batch, args.seqs))
            batch = torch.from_numpy(
                np.stack([val[i * 1024 : i * 1024 + 1025] for i in idx]).astype(np.int64)
            )
            losses.append(gpt2_loss(w, batch))
    per_seq = torch.cat(losses).numpy()
    n_tokens = args.seqs * 1024
    result = {
        "model": "gpt2 (OpenAI, 124M)",
        "revision": REVISION,
        "sequences": args.seqs,
        "tokens": n_tokens,
        "val_loss": float(per_seq.sum() / n_tokens),
        "published_reference": {
            "value": 3.2924,
            "source": "karpathy/build-nanogpt play.ipynb (commit 6104ab1), FineWeb-Edu val shard",
        },
        "seconds": time.time() - t0,
    }
    out = ROOT / "runs" / "reference" / "gpt2_on_val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
