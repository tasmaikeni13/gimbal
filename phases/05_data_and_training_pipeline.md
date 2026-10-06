# Phase 05 — Data and training pipeline (125M model, 2.5B FineWeb-Edu tokens)

## Purpose

Build the data, model, training loop, evaluation and logging used by all competitive runs, and show
that an AdamW baseline behaves as published work says it should.

## Depends on

Phase 04 (distributed optimizer step), Phase 03 (optimizers).

## Produces

* `data/README.md` and `scripts/data/` — download, tokenization, sharding, checksums
* `src/gimbal/train/` — model, loop, eval, logging, checkpointing (JAX/Flax)
* `configs/base_125m.yaml` — the single source of truth for the run configuration
* `runs/smoke/` — smoke-test logs

## Specification (change only through protocol §5)

**Data.** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`. Tokenize with the GPT-2 BPE (`tiktoken`
"gpt2"), append the end-of-text token after each document, store `uint16` shards in a GCS bucket in
the same region as the TPU. Split by document with a fixed hash: **train** = first 2.5B tokens of
the shuffled train documents; **validation** = 20M tokens from disjoint documents (used for tuning
and curves); **test** = another disjoint 20M tokens, read only in Phase 08. Record shard checksums.

**Model.** Llama-style decoder, 12 layers, $d=768$, 12 heads, SwiGLU hidden 2048, RoPE
($\theta=10^4$), RMSNorm (pre-norm), no biases, tied input/output embeddings, vocabulary padded to
50304, context 1024. ≈124M parameters (print the exact count). Init: $\mathcal N(0,0.02)$, output
projections scaled by $1/\sqrt{2L}$. bf16 compute, fp32 master weights and optimizer state.

**Training.** (D-003) Global batch 240 × 1024 = 245,760 tokens (the user's sweep of ≈600M tokens
in 2,500 steps fixes it, and the confirmatory runs keep it so that the tuned learning rates
transfer): confirmatory runs 10,172 steps = 2.49987B tokens (every full sequence of the 2.5B-token
train split, each once); tuning runs 2,500 steps = 614M tokens. (Originally: 512 × 1024, 4,768
steps.) Learning-rate
schedule: linear warm-up for 5% of steps, then cosine decay to 10% of peak (one schedule family for
every optimizer; Phase 06 may change the family for all optimizers at once). Global gradient-norm
clipping at 1.0 for all optimizers. Decoupled weight decay on matrices only. Parameter routing as in
Phase 03.

**Evaluation.** Validation loss every 250 steps on a fixed 5M-token subset and at the end on the full
20M validation tokens; perplexity $=e^{\text{loss}}$. Deterministic eval batches.

**Logging (JSONL, one line per step).** loss, lr, grad-norm, step time, tokens/s; every eval;
optimizer diagnostics every 100 steps (Gimbal: $\|\Omega_L\|,\|\Omega_R\|$, orthogonality defect,
non-separability index $\kappa$ of $D$; SOAP-family: frame change per refresh). Upload logs and
checkpoints (every 500 steps, Orbax) to GCS.

**Determinism.** A seed fixes initialization and data order. Use the *same* seed set across
optimizers (paired design).

## Smoke tests (all required)

1. Tiny config overfits 1M tokens to near-zero loss.
2. Zero learning rate leaves the loss constant.
3. 200 AdamW steps at the full config decrease the loss; no NaNs.
4. Two runs with the same seed give identical losses for 50 steps (bitwise or within 1e-6).
5. Resume from a checkpoint reproduces the next 20 steps.
6. Throughput is recorded for every optimizer (feeds Phase 04 G4.2 at the real configuration).

## Exit gate

* G5.1 All smoke tests pass; data checksums recorded.
* G5.2 A short AdamW run (e.g. 1,000 steps) is plausible against published references for ~124M
  models on FineWeb/FineWeb-Edu with the GPT-2 tokenizer (cite the references used; explain any
  difference in tokenizer, data or schedule).

## Failure handling

Loss not decreasing → check data (decode a batch to text), labels shift, masking, LR units, and the
parameter routing. Throughput low → check input pipeline (prefetch, host-to-device), recompilation,
and sharding. Never compare optimizers on a pipeline that has not passed G5.1.
