# Data

Two datasets are used, both from `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

## Token files for the TPU runs (Phases 05–08)

`scripts/data/prepare_fineweb_edu.py` reads the first four parquet shards of the sample (2,916,000 documents, 3,004,523,761 GPT-2 tokens), tokenizes every document with `tiktoken` `gpt2` and appends the end-of-text token (50256), then:

* splits documents by `sha256(id) % 1000: 0-9 val, 10-19 test, else train`;
* orders the documents of each split by `sha256('order:' + id), ascending, within each split`;
* writes the concatenation in that order, truncated to the target length, as `uint16` to `data/tokens/{train,val,test}.bin` (git-ignored).

| split | tokens | documents used | documents available | SHA-256 |
|---|---|---|---|---|
| train | 2,500,000,000 | 2,426,419 | 2,857,405 | `f4a3c2a4603eccb4e034e266592e4337d7b3a6f9983cca24c4cd4271a943a51d` |
| val | 20,000,000 | 19,798 | 29,146 | `841db9a7f1dacb45771ff67a300bf8175a31851d6d17567a0faa6ea368524772` |
| test | 20,000,000 | 19,055 | 29,449 | `05411556994a6a05d7dce6a5d32bec1ea78fdd212c079cc6af342b1821052da2` |

Input shards and their SHA-256 are listed in `data/tokens_manifest.json` (a copy of `data/tokens/manifest.json` written by the script). The test split is read once, in Phase 08, after the decision rule has been applied to the validation results.

A training sequence is 1025 consecutive tokens (input and shifted target) starting at a multiple of 1024; the train split holds 2,441,406 of them. A seed fixes a permutation of the sequences (the same for every optimizer); a run of S steps with 240 sequences per step uses the first 240·S entries of it, so no sequence repeats within a run (`src/gimbal/train/data.py`).

Reproduce:

```bash
python -c "from huggingface_hub import hf_hub_download as d; [d('HuggingFaceFW/fineweb-edu', f'sample/10BT/{i:03d}_00000.parquet', repo_type='dataset', revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9', local_dir='data/raw/fineweb-edu-10BT') for i in range(4)]"
python scripts/data/prepare_fineweb_edu.py --files 4 --workers 64
```

## Byte-level sample for the CPU small language model (Phase 03, E3.1–E3.2)

`scripts/data/fetch_fineweb_edu_sample.py` writes `data/raw/fineweb_edu_{train,val}.txt`: rows 0–5,999 (train) and 2,000,000–2,000,599 (validation) of the same dataset, read from the local shards (`data/raw/manifest.json` records row ranges and checksums).
