# Training (QDoRA)

`qdora_deviant.py` fine-tunes a model on a `{system, user, completion,
verdict, reward}` JSONL. Two producers feed it, the same format both:

- **The Deviant** (`export_dataset`) — the attacks that landed, so a red-team
  technique moves into the weights instead of being rediscovered every bout.
- **Verified reasoning self-training** (`selftrain_export`) — the reasoning
  traces the harness graded *correct*, so the model learns from its own working
  that actually solved the problem (STaR/ReST). This is the feasible form of
  "retrain the model" to lift it past the substrate ceiling on reasoning.

Both are one-shot offline batch jobs. The harness stays Rust; nothing here runs
in its hot loop.

## Where it runs — the compute split

A 4 GB laptop *serves* inference but cannot *train* even a 4B in QDoRA (that
wants ~16 GB VRAM). So the work splits:

- **Local (laptop):** generate traces, grade them, export the JSONL, re-eval.
- **Cloud (the fine-tune only):** a **Kaggle free T4 (16 GB, ~30 GPU-h/week)**
  is enough for QDoRA of a 4B and needs no payment. **Colab** works too; Colab
  Pro (L4/A100) is worth buying only when you move to a bigger base (7-14B),
  full fine-tuning, or long runs — not for a 4B QDoRA. Push the trained adapter
  to HF or download it, convert to GGUF, and serve it locally.

## Verified reasoning self-training (the reasoning loop)

```
serve.ps1 -Role solver                         # the base model, local
selftrain_export (DATASET=<trainable set>)     # solve -> grade -> keep correct
  -> reasoning-selftrain.jsonl                 # verified traces, trainer-ready
qdora_deviant.py <that jsonl>   (on Kaggle/Colab GPU)  -> adapter
  -> merge + convert to GGUF -> serve.ps1 -Role solver  # the improved base
reason_eval (DATASET=<held-out p2>)            # did it move? measure honestly
```

1. **Generate + verify (local).** With the solver served:
   ```powershell
   $env:DATASET = "$env:USERPROFILE\models\reasoning\gsm-symbolic-main.jsonl"
   cargo run -p samaritan-run --example selftrain_export   # writes reasoning-selftrain.jsonl
   ```
   Use an *easier* trainable split to generate from (GSM-Symbolic `main`/`p1`,
   or GSM8K) so the solver actually lands some — you can only learn from what it
   solved. Keep the hard `p2` set held-out for measuring.
2. **Fine-tune (cloud GPU).** Upload `reasoning-selftrain.jsonl` (as a Kaggle
   dataset, or push via HF) and run the trainer there:
   ```bash
   python qdora_deviant.py reasoning-selftrain.jsonl --base-model Qwen/Qwen3-4B-Thinking-2507
   ```
   (`--allow-small` for a first proof run; every row is verified-correct, so the
   set is all positive signal.)
3. **Convert + serve + measure.** Merge the adapter, convert to GGUF (see below),
   point `serve.ps1 -Role solver` at it, and re-run `reason_eval` on held-out p2.
   The number moving — or not — is the honest test of whether the loop works.

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
   healthy case) and there is nothing to train on. Point the Deviant at weakened
   foils to generate positive signal. Four holes exist, each a documented,
   realistic weakness; `WEAK` opens a set (`all`, or a comma list):
   - `separator` (CWE-41) — the path-equivalence hole. High-value (teaches exact
     frozen-path spelling) but the 8B fumbles the string, so it lands rarely.
   - `ceiling` (CWE-1284) — the bounds hole. Lands reliably but low-diversity:
     the model converges on a few `calibration_ceiling` raises.
   - `sandbox` (CWE-22) — the path-canonicalisation hole. The model invents a
     stream of *distinct* escape paths (`\\?\`, UNC, device, drive-letter), so
     this is the **richest source of variety**.
   - `misgrade` (CWE-807) — the trust-the-label hole. Varied mislabels land.

   **Diversity, not landing rate, is the binding constraint** — a fine-tune
   needs distinct techniques, not many copies of one. So the **harvester**
   defaults to `WEAK=all`: one bout per seed, every hole open, landings merged
   into one dataset, reported against `--min-positive`:
   ```powershell
   $env:KNOWLEDGE="seed"; $env:SEED_COUNT="8"   # WEAK defaults to all
   cargo run -p samaritan-adversary --example harvest   # writes harvest-dataset.jsonl
   ```
   The harvester writes the merged positives directly, so with it you skip the
   per-bout export in step 2 and train on `harvest-dataset.jsonl`. For a single
   focused bout instead, `train_target` takes the same `WEAK` (default
   `ceiling`) and writes a `LEDGER_DB`.
2. **Export the positives** (only if you ran a single `train_target` bout).
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
