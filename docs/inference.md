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

| offload | estimated |
|---|---|
| 0% (CPU only) | ~5.9 tok/s |
| 30% | ~7.7 tok/s |
| ~44% (the most that fits) | ~9.0 tok/s |

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
3. **Partial offload** (`-ngl`). ~1.5×. Measure it with `scripts/tune-ngl.ps1`;
   one layer past the VRAM limit llama.cpp spills and throughput falls off a
   cliff rather than plateauing.
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
