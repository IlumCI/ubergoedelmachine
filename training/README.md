# Training the Deviant (QDoRA)

The Deviant learns in two ways. In the arena it learns *explicitly* — a landed
attack becomes a known exploit class and a failing test. This directory is its
*implicit* channel: a QDoRA fine-tune on the attacks that landed, so the
technique moves into the weights instead of being rediscovered every bout.

It is a one-shot offline batch job. The harness stays Rust; nothing here runs in
its hot loop.

## Why Python / Unsloth

QDoRA's speed is in Triton/CUDA kernels, and Unsloth's are among the fastest on a
single consumer GPU. Python is only the launcher. There is no faster QDoRA path
in Rust today — `candle` has no mature quantized-training or DoRA support, and
building one would be more code around an identical GPU core, not less time.

## The pipeline

```
bout (train_target, KNOWLEDGE=seed)  ->  ledger.db
ledger.db  ->  export_dataset (POSITIVES_ONLY=1)  ->  deviant-dataset.jsonl
deviant-dataset.jsonl  ->  qdora_deviant.py  ->  QDoRA adapter
adapter  ->  merge + convert  ->  GGUF  ->  serve.ps1
```

1. **Produce landings.** Against intact guards the Warden repels everything (the
   healthy case) and there is nothing to train on. Point the Deviant at the
   weakened foil to generate positive signal:
   ```powershell
   $env:KNOWLEDGE="seed"; $env:LEDGER_DB="bout.db"
   cargo run -p samaritan-adversary --example train_target
   ```
2. **Export the positives.**
   ```powershell
   $env:POSITIVES_ONLY="1"
   cargo run -p samaritan-adversary --example export_dataset -- bout.db deviant-dataset.jsonl
   ```
3. **Gate, then train.** Inspect without a GPU first:
   ```bash
   python training/qdora_deviant.py deviant-dataset.jsonl --dry-run
   ```
   Then, in a CUDA venv (`pip install -r training/requirements.txt`):
   ```bash
   python training/qdora_deviant.py deviant-dataset.jsonl --output adapters/deviant-qdora
   ```

## The base model is the HF checkpoint, not the GGUF

`serve.ps1` runs the **GGUF** (`...abliterated.Q6_K.gguf`). Unsloth trains from
the **safetensors** checkpoint — `--base-model huihui-ai/Huihui-Ministral-3-8B-Reasoning-2512-abliterated`
by default. They are the same model in two formats; train on the HF one, then
convert the merged result back to GGUF to serve it.

## After training: back to GGUF

Merge the adapter and convert with llama.cpp's `convert_hf_to_gguf.py`, then
requantize to match what you serve:

```bash
# in Python: model.save_pretrained_merged("merged/", tokenizer, save_method="merged_16bit")
python <llama.cpp>/convert_hf_to_gguf.py merged/ --outfile deviant.f16.gguf
<llama.cpp>/llama-quantize deviant.f16.gguf deviant.Q6_K.gguf Q6_K
```

Point `serve.ps1` at `deviant.Q6_K.gguf` to run the trained adversary — as its
own track, never dropped into a measured Solo/Critic/Adversarial arm.

## Guardrails (why it may refuse)

- **No landed rows** → refuses. A fine-tune on all-negatives learns to attempt
  nothing. Mirrors `DatasetSummary::has_positive_signal`.
- **Fewer than `--min-positive` (default 16)** → refuses unless `--allow-small`.
  One or two landings overfit to a string instead of teaching a technique.
- **Loss is on the completion only** — the attack, not the prompt.

Given the current state (one live landing, from the opening book), the gate will
refuse until more landings are gathered. That is correct: the honest next step is
more/tighter foil bouts, not a GPU afternoon on a single example. See the
`deviant-path-fidelity-limit` note for what those landings are meant to teach —
exact frozen-path spelling, the specific thing the 8B currently fumbles.
