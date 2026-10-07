"""Tokenize FineWeb-Edu (sample-10BT) into the train / validation / test token files of Phase 05.

Specification (``phases/05_data_and_training_pipeline.md``): GPT-2 BPE (``tiktoken`` "gpt2"), the
end-of-text token after every document, ``uint16`` storage, a split by document with a fixed hash,
train = the first 2.5B tokens of the shuffled train documents, validation and test = 20M tokens
each from disjoint documents.

* Corpus: the first ``--files`` parquet shards of ``HuggingFaceFW/fineweb-edu``,
  ``sample/10BT`` at a pinned revision (about 0.75B tokens per shard; four are enough).
* Split: ``sha256(id)`` modulo 1000 — 0–9 validation, 10–19 test, the rest train.
* Shuffle: documents of each split are ordered by ``sha256("order:" + id)``; the token files are
  the concatenation in that order, truncated to the target length.

Outputs ``data/tokens/{train,val,test}.bin`` and ``data/tokens/manifest.json`` (token counts,
document counts, SHA-256 checksums of every input shard and output file).

Usage: python scripts/data/prepare_fineweb_edu.py --files 4 --workers 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "fineweb-edu-10BT" / "sample" / "10BT"
OUT = ROOT / "data" / "tokens"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TARGETS = {"train": 2_500_000_000, "val": 20_000_000, "test": 20_000_000}
EOT = 50256

_ENC = None


def _key(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _encode(batch: tuple[list[str], list[str]]) -> tuple[np.ndarray, ...]:
    """Tokenize one batch of documents; returns split ids, order keys, lengths and tokens."""
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("gpt2")
    texts, ids = batch
    toks = _ENC.encode_ordinary_batch(texts, num_threads=1)
    split = np.array([_key(i) % 1000 for i in ids], dtype=np.int16)
    # 0 = train, 1 = validation, 2 = test
    split = np.where(split < 10, 1, np.where(split < 20, 2, 0)).astype(np.int8)
    order = np.array([_key("order:" + i) for i in ids], dtype=np.uint64)
    lengths = np.array([len(t) + 1 for t in toks], dtype=np.int64)
    flat = np.empty(int(lengths.sum()), dtype=np.uint16)
    pos = 0
    for t in toks:
        flat[pos : pos + len(t)] = t
        flat[pos + len(t)] = EOT
        pos += len(t) + 1
    return split, order, lengths, flat


def _batches(path: pathlib.Path, size: int):
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=size, columns=["text", "id"]):
        yield rb.column("text").to_pylist(), rb.column("id").to_pylist()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=4)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shards = [RAW / f"{i:03d}_00000.parquet" for i in range(args.files)]
    t0 = time.time()
    parts = {"split": [], "order": [], "lengths": [], "tokens": []}
    with mp.Pool(args.workers) as pool:
        for shard in shards:
            for split, order, lengths, flat in pool.imap(
                _encode, _batches(shard, args.batch), chunksize=1
            ):
                parts["split"].append(split)
                parts["order"].append(order)
                parts["lengths"].append(lengths)
                parts["tokens"].append(flat)
            print(
                f"{shard.name}: {sum(len(x) for x in parts['split'])} docs, "
                f"{sum(len(x) for x in parts['tokens']) / 1e9:.3f}B tokens, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    split = np.concatenate(parts["split"])
    order = np.concatenate(parts["order"])
    lengths = np.concatenate(parts["lengths"])
    tokens = np.concatenate(parts["tokens"])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    manifest = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "revision": REVISION,
        "tokenizer": f"tiktoken gpt2 ({tiktoken.__version__}), end-of-text {EOT} after each doc",
        "split_rule": "sha256(id) % 1000: 0-9 val, 10-19 test, else train",
        "order_rule": "sha256('order:' + id), ascending, within each split",
        "inputs": {s.name: sha256_file(s) for s in shards},
        "corpus_docs": int(len(split)),
        "corpus_tokens": int(tokens.size),
        "splits": {},
    }
    for code, name in ((0, "train"), (1, "val"), (2, "test")):
        idx = np.flatnonzero(split == code)
        idx = idx[np.argsort(order[idx], kind="stable")]
        need = TARGETS[name]
        cum = np.cumsum(lengths[idx])
        n_docs = int(np.searchsorted(cum, need) + 1)
        if cum[-1] < need:
            raise SystemExit(f"{name}: only {cum[-1]} tokens available, need {need}; add shards")
        out = np.empty(need, dtype=np.uint16)
        pos = 0
        for d in idx[:n_docs]:
            take = min(int(lengths[d]), need - pos)
            out[pos : pos + take] = tokens[starts[d] : starts[d] + take]
            pos += take
        path = OUT / f"{name}.bin"
        out.tofile(path)
        manifest["splits"][name] = {
            "tokens": int(need),
            "docs_used": n_docs,
            "docs_available": int(idx.size),
            "tokens_available": int(cum[-1]),
            "sha256": sha256_file(path),
        }
        print(name, manifest["splits"][name], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
