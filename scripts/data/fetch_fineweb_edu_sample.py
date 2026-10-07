"""Fetch a small, fixed FineWeb-Edu sample for the CPU-scale experiments of Phase 02.

Rows of ``HuggingFaceFW/fineweb-edu``, config ``sample-10BT``, in the dataset's row order. Two
sources give the same rows: ``--source parquet`` (default) reads the pinned local shards that
``prepare_fineweb_edu.py`` uses (``data/raw/fineweb-edu-10BT``; rows 0–2,915,999), ``--source api``
uses the datasets-server rows API (100 documents per request), which refused deep offsets with an
error page on 2026-10-06. Training and validation documents come from disjoint row ranges.
Output: ``data/raw/fineweb_edu_{train,val}.txt`` (documents separated by a NUL byte) and a JSON
manifest with row ranges and SHA-256 checksums.

The 125M / 2.5B-token pipeline of Phase 05 uses the full shards instead; this sample only feeds
the byte-level CPU language model.

Usage: python scripts/data/fetch_fineweb_edu_sample.py --train-docs 6000 --val-docs 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_rows(offset: int, length: int, retries: int = 5) -> list[str]:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": "train", "offset": offset, "length": length}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as resp:
                payload = json.load(resp)
            return [r["row"]["text"] for r in payload["rows"]]
        except Exception:  # network hiccup: back off and retry
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch rows at offset {offset}")


def fetch_range(start: int, n_docs: int) -> list[str]:
    docs: list[str] = []
    for offset in range(start, start + n_docs, 100):
        docs.extend(fetch_rows(offset, min(100, start + n_docs - offset)))
    return docs


SHARDS = OUT / "fineweb-edu-10BT" / "sample" / "10BT"


def parquet_range(start: int, n_docs: int) -> list[str]:
    """Rows ``[start, start + n_docs)`` of the dataset from the local shards (global row order)."""
    import pyarrow.parquet as pq

    docs: list[str] = []
    offset = 0
    for path in sorted(SHARDS.glob("*.parquet")):
        f = pq.ParquetFile(path)
        rows = f.metadata.num_rows
        lo, hi = max(start, offset), min(start + n_docs, offset + rows)
        if lo < hi:
            table = f.read(columns=["text"]).slice(lo - offset, hi - lo)
            docs.extend(table.column("text").to_pylist())
        offset += rows
    if len(docs) != n_docs:
        raise SystemExit(f"rows {start}..{start + n_docs} are not all in the local shards")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-docs", type=int, default=6000)
    parser.add_argument("--val-docs", type=int, default=600)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument(
        "--val-start", type=int, default=None, help="default 2,000,000 (parquet) or 5,000,000 (api)"
    )
    parser.add_argument("--source", choices=["parquet", "api"], default="parquet")
    args = parser.parse_args()
    if args.val_start is None:
        args.val_start = 2_000_000 if args.source == "parquet" else 5_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "source": args.source,
        "api": API if args.source == "api" else None,
    }
    for split, start, n in (
        ("train", args.train_start, args.train_docs),
        ("val", args.val_start, args.val_docs),
    ):
        docs = (parquet_range if args.source == "parquet" else fetch_range)(start, n)
        blob = "\x00".join(docs).encode("utf-8")
        path = OUT / f"fineweb_edu_{split}.txt"
        path.write_bytes(blob)
        manifest[split] = {
            "row_start": start,
            "n_docs": len(docs),
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        print(split, manifest[split], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
