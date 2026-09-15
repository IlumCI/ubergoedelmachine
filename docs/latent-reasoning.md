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

**The specific risk.** Every result in §1-§5 starts from a base model and teaches
it latent reasoning. Qwen3-4B-Thinking has already been heavily RL-trained to
reason in tokens, so a staged latent curriculum would be fighting that prior
rather than filling a vacuum. **§7 resolves this** - the fix is to freeze the
backbone - but it is the risk that would sink a naive attempt, and the reason not
to simply fine-tune the thinking model into latent mode.

## 7. Every obstacle in §6 is already solved — by a different paper each

The first pass called the prior-fighting risk "the genuine unknown, unaddressed
by any of the papers above". That was wrong; it is addressed, just not by the
COCONUT line.

| obstacle | solution | source |
|---|---|---|
| Fighting the RL-trained token-CoT prior | **Freeze the backbone.** SoftCoT projects instance-specific soft thought tokens into a *frozen* model's representation space, explicitly to avoid catastrophic forgetting. The prior is not overwritten because the weights holding it are never touched. | SoftCoT (via [survey](https://arxiv.org/html/2604.02029v2)) |
| Fixed, hand-chosen number of latent steps | **PLaT** models reasoning as a latent planning trajectory with a separate decoder, letting the model decide *when to stop* rather than running a preset count. | [2601.21358](https://arxiv.org/abs/2601.21358) |
| Latent collapse on long chains | **SIM-CoT** step-level supervision (§2). | [2509.20317](https://arxiv.org/abs/2509.20317) |
| Precision-critical steps over-compressed | **SLT** confidence gate, with automatic fallback to explicit CoT when the predicted span is unreliable. | [2605.25745](https://arxiv.org/abs/2605.25745) |
| Latent RL instability | **Latent-GRPO** one-sided noise + advantage masking (§3). | [2604.27998](https://arxiv.org/abs/2604.27998) |
| All-or-nothing commitment to latent mode | **ALAR** runs dual-mode: compact latent for routine steps, escalating to explicit CoT when deeper deliberation is needed. | [2606.02871](https://arxiv.org/abs/2606.02871) |

The frozen-backbone result is the important one for this project, because it
converts the largest risk into a design choice. Qwen3-4B-Thinking keeps every bit
of its token reasoning; the latent machinery is additive. It also means the
*fallback path is free* — an ALAR/SLT-style gate can escalate to the original
model, which is still intact underneath.

**Consequence: worst case stops being "we broke the model" and becomes "the gate
never fires and we wasted the training run."** That is a much cheaper failure,
and it is what makes this worth attempting at all.

## 8. What Samaritan has that none of these papers had

This is where the project is genuinely ahead, and it is not the model — it is
W7.

### 8.1 The expensive ingredient is free here

Every method in §7 needs **step-level supervision**, and the PRM literature is
largely a history of trying to obtain it affordably:

- human annotation of each step — abandoned as unscalable
- **MiPS / Monte Carlo rollouts** ([2402.02658](https://arxiv.org/pdf/2402.02658)):
  sample many completions from a partial state and use the *fraction correct* as
  a proxy label. Noisy, and costs N extra generations per step
- generator–verifier frameworks that produce labels "without ground truth"

W7 does not approximate. **The generator built each problem from an algorithm, so
it already knows the true intermediate state at every step** — the
square-and-multiply sequence for modpow, the successive congruence merges for
crt, the constraint-propagation order for zebra, the assignment order for sat.

That single fact supplies, at zero cost and zero noise, the three things the
methods above each pay dearly for:

1. **exact step boundaries** for COCONUT's staged curriculum, which otherwise has
   to be segmented heuristically out of prose
2. **exact targets for SIM-CoT's auxiliary decoder** — the mechanism that stops
   latent collapse, and the one most sensitive to label quality
3. **a perfect process reward model**, with no rollouts and no annotation

No source was found combining a procedural generator's *native* step
decomposition with latent-state supervision. Synthetic reasoning benchmarks with
process traces exist; using generator-known intermediate states to supervise
*continuous* thoughts appears to be open ground.

The work is real but bounded: each family's generator must emit its solution
trace alongside the answer. It knows it already — it computed the gold with it —
so this is plumbing, not research.

### 8.2 Compressibility is knowable a priori, not only learnable

SLT *learns* a confidence gate to decide what may be compressed. Samaritan knows
the algorithmic class of every problem by construction, and §1 and §5 agree on
the split from opposite directions. So the gate can be **initialised from
structure and then refined**, rather than discovered from scratch — and, more
valuably, W7 is a curriculum deliberately spanning both classes, which makes it
an instrument for *testing* SLT's premise rather than merely consuming it.

### 8.3 Truncation is a free compression label

This project measures something the literature does not have to hand: exactly
which items hit the token cap. Tonight's run — 5 of the base's 6 failures and 9
of the student's 10 — is a per-item label saying *this problem needs compression*.
That is a ready-made curriculum for where latent reasoning should be pointed
first, and a ready-made evaluation: does the truncation count fall?

### 8.4 A cheaper way to get GRPO's group variance

PLaT reports that when greedy decoding from a latent state gives a wrong answer,
**the correct path is often still encoded in that same state**. If that holds,
then decoding the answer several times from *one* latent chain yields a group
with genuine variance — at the cost of the short answer decode rather than G full
reasoning chains. GRPO's group could become ~G× cheaper, which matters because
rollouts are the binding cost of every GRPO run.

Two honest caveats. The gradient then flows mostly through the decode, and the
shared latent chain receives an averaged signal — which is exactly Latent-GRPO's
**latent mixture non-closure** failure, so it would need that paper's masking.
And it is an inference from one reported observation, not a result. But it is
cheap to test: decode k times from one latent state, grade each against a
computed gold, and measure the recovery rate directly. W7 makes that a one-
afternoon experiment, and a negative result is just as informative.

## 9. Revised plan

1. **Plain GRPO first** — built, tested, ready. Its per-family reward curves
   answer three questions at once: is per-domain specialisation real (the
   hypernetwork gate), which families are compressible (SLT's premise, §5), and
   does token-level RL alone close the truncation gap.
2. **Teach the W7 generators to emit solution traces** (§8.1). Independent of
   everything else, cheap, and it is the prerequisite for every latent method as
   well as for a process reward model. Highest value per hour of anything here.
3. **Frozen-backbone latent probe on the search families only** — knights, sat,
   zebra, graph, where §1 and §5 both predict the win, with the token model
   intact underneath as the fallback. If latent reasoning cannot beat token CoT
   there, it will not on modpow.
4. **Latent-GRPO** only once a latent model exists to initialise from; it is a
   post-training method, not a from-scratch one.

The ordering is not conservatism. Steps 1 and 2 produce the evidence and the data
that steps 3 and 4 consume, and each is independently useful if the latent track
is abandoned — the reward curves settle the hypernetwork question either way, and
generator solution traces give a process reward model regardless of whether a
single vector is ever fed back into the model.
