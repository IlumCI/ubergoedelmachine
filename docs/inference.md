# Inference: what runs the model, and why

## The model

`mradermacher/Huihui-Ministral-3-8B-Reasoning-2512-abliterated-GGUF`, **Q6_K**,
6.97 GB. Read from the GGUF header rather than assumed:

```
general.architecture = mistral3
general.size_label   = 8B
gguf v3, 309 tensors, 57 metadata keys
```

The repo also ships `mmproj` projectors — it is a multimodal checkpoint — but
nothing in the harness sends it an image, so they are not fetched. The VRAM is
better spent on transformer layers.

**Abliterated matters here for one practical reason.** Refusal ablation is a
blunt edit to the residual stream and reliably costs some instruction-following
along with the refusals. Every response must parse as a `DraftDecision`, so the
agent constrains decoding with a GBNF grammar rather than asking politely; see
`samaritan-agent/src/grammar.rs`. It is also the right choice for the Deviant,
which is supposed to genuinely attempt containment breaks — a refusal-trained
model would decline the adversary role and the arms race would never start.

## The hardware, and the arithmetic that follows from it

```
i7-12650H    6 P-cores + 4 E-cores, 16 threads
16 GB        DDR4-3200 dual channel, ~38 GB/s realistic
RTX 3050     Laptop, 4 GB VRAM (~3.5 GB usable), ~192 GB/s
```

Q6_K is 6.97 GB. **Nothing above Q3_K_S fits in 4 GB**, so Q6 necessarily means
a split load. Generation is memory-bandwidth-bound, so throughput follows
directly from where the weights live:

Predicted, then measured on the actual card with `llama-bench` (Q6_K, 64
generated tokens, `-t 6`):

| `-ngl` | measured tok/s |
|---|---|
| 6 | 6.11 |
| 10 | 6.72 |
| 14 | 7.59 |
| **18** | **8.64** |
| 22 | 5.39 |

The arithmetic above predicted ~9.0 tok/s at roughly 44% offload; the real
figure is 8.64 at 18 layers, so the throughput estimate held and the layer
estimate was conservative.

The shape at 22 is the point worth internalising: it does not plateau, it
**collapses** — below even the 6-layer result. One layer past what VRAM holds,
llama.cpp spills and every token pays for it. That is why the sweep stops on a
drop rather than continuing to the end of its range, and why this number has
to be re-measured whenever the context size changes, since the KV cache comes
out of the same 4 GB.

So a ~500-token decision record is around **55 s single-stream**, against 85 s
on CPU alone. The split is worth doing — roughly 1.5× — but it is not the lever
that matters most.

### The levers, in order of how much they return

1. **Parallel slots** (`--parallel 4 --cont-batching`). The weights are read
   once per token whether that token belongs to one sequence or four, so
   batching four playouts amortises the read across all of them: roughly
   2–2.5× aggregate throughput, while each individual response is no faster.
   NRPA playouts are embarrassingly parallel, which is exactly the shape this
   wants.
2. **Prefix reuse** (`--cache-reuse`). Every playout in a round shares a system
   prompt and a lesson list. `samaritan-agent/src/prompt.rs` lays prompts out in
   three bands, most stable first, so the shared prefix is processed once per
   round rather than once per episode. Interleaving one episode-specific token
   into the stable band silently costs more than every other optimisation here
   returns.
3. **Partial offload** (`-ngl 18`). ~1.4× over CPU-only, measured. Re-measure
   with `scripts/tune-ngl.ps1` after any context-size change;
   `scripts/refine.ps1` then narrows it layer by layer.
4. **Grammar-constrained decoding.** Does not speed up a token, but removes the
   retries — and a retry costs a whole generation.
5. **8-bit KV cache and flash attention.** Both cut memory traffic, which is
   the binding constraint.

### What this means for the experiment

At ~9 tok/s single-stream and ~2.2× from batching, call it **20 tok/s
aggregate**. A decision record is then ~25 s amortised. A hundred playouts is
roughly 40 minutes; a level-1 NRPA round at Rosin's N=100 is days.

That is survivable for the `Solo` arm and painful for `Adversarial`, which runs
two agents against the same budget. Two honest options when it starts to bite:

- **Split the roles by model.** The config already has `playout`, `proposer`
  and `deviant` slots. A 3B–4B checkpoint at Q6_K is ~2.5 GB, fits entirely in
  4 GB VRAM, and runs an order of magnitude faster. Using it for the thousands
  of level-0 playouts while keeping Ministral 8B for the rare proposer calls
  honours "no lower than Q6" exactly — that was a quantisation floor, not a
  parameter-count floor — and is the single largest speedup available.
- **Accept the wall clock** and size the experiment to it: fewer NRPA
  iterations, smaller level-0 batches, and a correspondingly weaker
  certificate.

## Why llama.cpp and not RustLMHub

[RustLMHub](https://github.com/IlumCI/RustLMHub) is the better engine for what
it targets, and it will not run this model today. Two separate reasons, and the
interesting one is not the obvious one.

**It does support dense models**, contrary to a first reading of `arch.rs`.
The mechanism is in `moearch.rs`: `qwen35` is registered as the *dense sibling*
of `qwen35moe` — `dense: true`, `shared_expert: false`, and otherwise identical.
The claim, which the code makes explicitly, is that dense is not a separate
family but the same stack with the feed-forward swapped, so one block
implementation serves both. It is verified by diff against llama.cpp's
`qwen35.cpp`.

**The blocker is the architecture table, not density.** `Cfg::from_meta`
accepts exactly two names:

```rust
if arch != "qwen35moe" && arch != "qwen35" {
    return Err(format!("{arch:?} is neither qwen35moe nor qwen35"));
}
```

Ministral reports `mistral3`, so it is refused — deliberately, with a comment
explaining that guessing at a config would "silently produce a DIFFERENT
model," which is the correct instinct.

**And it does not build on Windows.** `io.rs`, `st.rs` and `qwen35.rs` use
`std::os::unix::fs::FileExt` unconditionally; CI covers ubuntu and macos-14.

### What adding `mistral3` would actually take

Less than it first appears, because `uses_common_block()` is
`!hybrid && !fused_q_gate && !mla`, so a `mistral3` entry would use the
*already-verified* common block — it is structurally closer to `COMMON` than
`qwen35` is, which needs its own block for the hybrid attention and fused
q/gate.

The real work:

- a `Cfg` reader for `mistral3.*` metadata keys (the present one is
  qwen-specific);
- dense feed-forward tensor names — `blk.N.ffn_gate.weight` rather than
  `expert_names()`'s `ffn_gate_exps.weight`;
- sliding-window attention, which `Arch` has no field for today;
- a Windows positional-read shim: `read_at` → `seek_read` behind a `cfg`, plus
  the `O_DIRECT` → `FILE_FLAG_NO_BUFFERING` path, which is fiddlier because of
  its alignment requirements.

Days rather than weeks of code — but that codebase diffs every new family
against a reference before marking it `verified`, and honouring that convention
is the actual cost.

**When it becomes worth it:** the moment a huge MoE is wanted for the proposer
role. Streaming a 2.78T checkpoint at 8 GB resident is not available anywhere
else, and a strong proposer paired with a small fast playout model is exactly
the shape this design already assumes.

## What was taken from RustLMHub regardless

Its server always chooses a seed and always reports it, on the grounds that a
seed the caller cannot see is a seed that does not exist. Sampling is the only
nondeterminism in an episode, so that single decision is what makes a run
replayable — and a self-improvement result nobody can re-run is not a result.

`AgentConfig::seed` now does the same thing, with one change: the seed is
chosen *client-side* and recorded on every `Completion`, because not every
OpenAI-compatible server echoes the field back, and a seed that is sometimes
recorded is no better than one that never is.

## Running it

```powershell
.\scripts\fetch-model.ps1     # ~6.5 GB, resumable
.\scripts\tune-ngl.ps1        # measure the best split on this box
.\scripts\serve.ps1 -Ngl 14   # use the number tune-ngl printed
```

## Choosing the reasoning substrate

Measured baseline (this box, RTX 3050 Laptop 4 GB, `llama-bench -ngl 18 -t 6`):

| Huihui-Ministral-3-8B Q6_K | tok/s |
|---|---|
| prompt prefill (pp512) | 357 |
| generation (tg128) | 7.94 |

The budget is ~120 s per call. Prefill is cheap (~3 s for a 1 k-token prompt),
so the whole budget is the generation: `~117 s × 7.94 ≈ 930 tokens`. That is the
decisive fact — **the 8B can just about emit a 900-token *answer*, and has no
room left for a *thinking trace*.** A reasoning-distilled model does not answer
in 900 tokens; it thinks for one to three thousand first. At 7.94 t/s a
2 000-token trace is 4.2 minutes — over double the budget.

So the substrate that reasons well *and* fits the box must run **faster** than
the current 8B, which on a 4 GB GPU means **fewer parameters or a heavier quant**
(more layers resident on the GPU). Required generation speed:

| tokens per answer (think + answer) | needed tok/s |
|---|---|
| 900 (terse) | ≥ 7.7 |
| 1 500 (light thinking) | ≥ 12.8 |
| 2 500 (real thinking) | ≥ 21 |

**What to get.** A reasoning-distilled / native-thinking small model, **~3–4B**,
on a **Qwen** base — the family every strong small-reasoning model is built on
because Qwen bases are unusually strong at math/code per parameter.

- **First choice — Qwen3-4B-Thinking (Alibaba Qwen).** ~4B, native thinking; at
  Q5_K_M (~3 GB) most layers sit on the GPU → ~20–35 t/s → a real thinking trace
  fits the budget.
- **DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek × Qwen).** Fastest (fits the GPU
  outright), weakest — a good speed/floor reference.
- **DeepSeek-R1-Distill-Qwen-7B / OpenThinker3-7B (Qwen base), at Q4_K_M.**
  Strongest reasoning that still ~fits, but ~8–10 t/s — same "no thinking
  headroom" problem as the 8B. Pick only if per-question time may exceed 2 min.

Grab GGUFs from the usual quant providers (bartowski / unsloth / mradermacher),
then **verify on the box before trusting it** — at the model's *real* thinking
length, not the flattering 900:

```powershell
.\scripts\bench-model.ps1 -Model $env:USERPROFILE\models\<candidate>.gguf `
    -Ngl 20,24,28 -ThinkTokens 2000
```

Note the two roles want different models: the **solver** wants a strong small
Qwen reasoning model (no abliteration needed — solving math trips no refusals);
the **adversary/Deviant** keeps the abliterated checkpoint, because its job is to
attempt what a refusal-trained model would decline.
