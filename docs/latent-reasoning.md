# Latent reasoning for Samaritan — what the literature says

Research note, 2026-09-16. The question: should the student reason in continuous
vectors (COCONUT-style) rather than tokens, with GRPO and eventually a
hypernetwork optimising that latent process?

**Short answer: the idea is sound, the exact combination has been published, and
it beats explicit GRPO on hard problems — but the gain is 3–4× efficiency rather
than an order of magnitude, the naive version collapses on long chains, and the
fixes matter more than the core idea.**

Sourced from search summaries and abstracts, not full-text reads. Numbers below
should be confirmed against the papers before anything is built on them.

---

## 1. The core method

**COCONUT** ([2412.06769](https://arxiv.org/abs/2412.06769), Meta). The last
hidden state is fed back as the next input embedding instead of being decoded to
a token. The "thought" is a vector.

The advantage is not compression. It is **superposition**: a continuous state can
encode several candidate next steps at once, so the model does something like
breadth-first search instead of committing to one path and backtracking in text.
Reported wins are on search-shaped logical reasoning (ProsQA), not on arithmetic.
There is now a theoretical treatment of exactly this
([2505.12514](https://arxiv.org/pdf/2505.12514), "Reasoning by Superposition").

**This predicts a split across the W7 curriculum**, and the prediction is sharp
enough to be worth testing:

| should benefit | should not |
|---|---|
| knights, sat, zebra, graph — search, backtracking, constraint propagation | modpow, crt, recurrence — deterministic computation chains |

There is nothing to hold in superposition when the work is "apply this step, then
the next". Automata and divideconquer sit somewhere between.

## 2. The failure mode, and its fix

Naive latent reasoning **collapses as the chain gets longer**. COCONUT on GPT-2
Small reportedly goes from ~99% at two latent steps to ~38% at five; CODI from
~85% to ~20%. The diagnosis is that latent representations become homogeneous —
they lose semantic diversity — because nothing supervises the individual steps.

**SIM-CoT** ([2509.20317](https://arxiv.org/abs/2509.20317), ICLR 2026,
[code](https://github.com/InternLM/SIM-CoT)) is the published fix: an auxiliary
decoder during training aligns each implicit token with its corresponding
explicit reasoning step, so latent states are forced to stay distinct. The
decoder is discarded at inference, so it costs nothing at serving time. Reported
+8.2% on Coconut/GPT-2 and +3.0% on CODI/**LLaMA-3.1 8B**, beating explicit CoT
on GPT-2 by 2.1% with 2.3× better token efficiency.

The 8B result matters here: it is evidence the approach survives past toy scale,
which was the main thing to doubt.

## 3. Latent RL — the part I expected to be an obstacle

I assumed GRPO and latent reasoning were in tension: GRPO's gradient comes
entirely from variance *within* a group of rollouts, that variance comes from
sampling, and a deterministic latent chain has nothing to sample. That tension is
real and named in the literature, but it has been solved.

**Latent-GRPO** ([2604.27998](https://arxiv.org/abs/2604.27998)) identifies three
coupled failures:

1. **off-manifold exploration** — unconstrained perturbation pushes rollouts off
   the valid latent manifold
2. **exploration–optimization misalignment** — trajectory-level rewards induce
   incorrect token-level updates
3. **latent mixture non-closure** — jointly reinforcing several correct latent
   paths produces an invalid *averaged* state, which is not any of them

Fixes: invalid-sample advantage masking, **one-sided noise sampling** (a strictly
positive perturbation margin per latent component, rather than symmetric Gumbel
noise that can push probability the wrong way), and optimal-correct-path
first-token selection.

Reported results: **+7.86 Pass@1 over its latent initialisation on easy
benchmarks, and +4.27 over explicit GRPO on hard ones (AIME among them), with
3–4× shorter reasoning chains.**

That is the proposal in this project's own terms — right *and* short, with the
model finding its own compression — already demonstrated. Worth noting the naive
"just add noise to the vectors" instinct is specifically the thing that fails
here; the one-sided construction is the non-obvious part.

## 4. How much efficiency, really

Consistently **2.7–4.4×** across methods, with outliers: Abstract-CoT
([2604.22709](https://arxiv.org/pdf/2604.22709)) reports 11× on MATH-500 at near
parity. Latent-GRPO reports 3–4×. SIM-CoT reports 2.3×.

Applied to what this project measured on generated d3:

| | tokens |
|---|---|
| base median, measured | 5,602 |
| student median, measured | 6,185 |
| at 3–4× | ~1,400–2,050 |

That would **eliminate the failure mode currently dominating both models**: 5 of
the base's 6 failures and 9 of the student's 10 were truncation at the 6,000
token cap, not wrong reasoning. So the gain lands precisely where the measured
bottleneck is — but it is a constant-factor win, not reasoning collapsing into a
handful of vectors.

## 5. The finding that most changes the plan

**SLT** ([2605.25745](https://arxiv.org/abs/2605.25745)) — selective latent
thinking — starts from the observation that *existing latent methods treat
reasoning as uniformly compressible, over-compressing precision-critical steps
and degrading accuracy*. It compresses redundant spans to latents while keeping
precision-critical spans as explicit CoT in the same trajectory, via a confidence
gate. Reported +22.7% over uniform latent baselines on four maths benchmarks.

This is the same split predicted in §1 from the superposition argument, arrived
at independently and from the opposite direction. A modular-exponentiation chain
has no redundant span to compress: every step carries a number the next step
needs. A knights puzzle is mostly search, and search is exactly what compresses.

**So the right target is not "reason in vectors" — it is "reason in vectors where
that is safe, and in tokens where it is not."**

Related: Think Silently, Think Fast
([2505.16552](https://arxiv.org/abs/2505.16552), NeurIPS 2025) on dynamic latent
compression; surveys at [2505.16782](https://arxiv.org/abs/2505.16782) and
[2507.06203](https://arxiv.org/abs/2507.06203).

## 6. What this would cost here

**Serving.** Latent reasoning cannot be served by Ollama or llama.cpp — they are
token-in, token-out. But the measurement stack survives: wrap the PyTorch model
in a shim exposing `/v1/chat/completions`, run the latent loop inside it, and
emit only the final `Answer:` / `Confidence:` lines as tokens. `reason_eval`, the
grader, W7 and the A/B all work unchanged, because the contract was always
question-in / answer-text-out and the thinking was never part of it.

**Training.** Three stages minimum, none of which exist here yet: latent SFT with
a staged curriculum (progressively replacing language steps with vectors),
step-level supervision (SIM-CoT) to stop the collapse in §2, then latent RL
(Latent-GRPO) on top. Each needs traces **with identifiable step boundaries** —
which the current 185-trace corpus does not have, and which W7 could generate,
since the generator knows the solution structure it built the problem from.

**The specific risk.** Every result above starts from a base model and teaches it
latent reasoning. Qwen3-4B-Thinking has already been heavily RL-trained to reason
in tokens, and a staged latent curriculum would be fighting that prior rather
than filling a vacuum. No source found that does this on top of an existing
long-CoT reasoning model. That is the genuine unknown, and it is not addressed by
any of the papers above.

## 7. Recommended order — unchanged by the research

1. **Plain GRPO first** (built, tested, ready). Its **per-family reward curves**
   are the gate for three separate decisions: whether per-domain specialisation
   is real (the [[hypernetwork]] question), which families are compressible
   (SLT's premise), and whether token-level RL alone closes the truncation gap.
   One run, three answers.
2. **A latent SFT probe on the search families only** — knights, sat, zebra,
   graph — where both §1 and §5 predict the win. If latent reasoning cannot beat
   token CoT *there*, it will not beat it on modpow.
3. **Latent-GRPO** only after a latent model exists to initialise it from; it is
   a post-training method, not a from-scratch one.

The sequencing matters because the alternative is three unvalidated things at
once — latent reasoning, latent RL, and a hypernetwork — with no way to attribute
a failure to any of them.
