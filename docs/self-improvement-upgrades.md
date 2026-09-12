# Recursive self-improvement, borrowed from outside computer science

> Status: proposal. A menu of mechanisms drawn from fields *other than* ML, each
> mapped onto Samaritan's existing machinery, with a recommendation and an honest
> account of what any of it can and cannot do to an HLE score.

## The honest ceiling first — because the rest is worthless without it

The brief is a recursive self-improvement method that could reach **>30% on HLE**.
The base model scores ~4.3%. No wrapper, search, or self-reference trick turns an
8B into a 30% HLE model, and saying otherwise would betray the one thing this
project is built on — measuring rather than asserting. HLE tests graduate-level
knowledge and reasoning the 8B does not contain, and **you cannot self-improve
your way to knowledge that is nowhere in your training signal.** Samaritan's
level-1 loop tunes lessons, prompts, and thresholds — the agent's *policy over
repo tasks* — which is orthogonal to what HLE asks.

So separate two things the brief conflates:

1. **What recursion can genuinely improve**: generalisation, sample-efficiency,
   capability-per-unit-compute, and the *reliability* with which the model
   applies what it already knows. These are real, measurable, and the honest
   target of the mechanisms below. The out-of-arena yardstick and HLE-as-a-
   dashboard-signal will move here.
2. **What actually raises an HLE number**: (a) a stronger base model; (b) a
   training signal that *contains* HLE-like reasoning (not git commits); (c)
   test-time tool use — retrieval over a real corpus, code execution, search.
   Only these change what the model *is* or what it can *reach*.

The one bridge between (1) and (2) that this architecture can build is the
**Baldwin effect** — assimilating repeatedly-useful learned behaviour into the
weights — because that is the only mechanism here that moves base capability
rather than policy. Even so: honest gains are single-digit-to-low-double-digit
percentage points on a *reasoning* benchmark with tools, not a leap to 30% on a
knowledge benchmark from a small base. The design should chase the real gains and
report the HLE number as what it is.

With that fixed, the interesting part.

## The mechanisms

Each is: the source field, the precise mechanism, where it lands in Samaritan,
and what it should actually do.

### 1. Iterated learning — the transmission bottleneck (linguistics / cognitive science)

Kirby's iterated-learning work shows that when knowledge is passed through a
*compression bottleneck* across generations of learners, it becomes
compositional and general — structure that does not survive the bottleneck is
selected out. This is a mechanism for **generalisation as a consequence of
transmission**, and nothing in mainstream self-improvement uses it.

- **Where**: `samaritan-reflect` (lesson memory) + `samaritan-cert`.
- **What changes**: each generation, the lesson corpus is not inherited whole. It
  is passed through a bottleneck — re-derived from a *bounded* budget, or
  compressed to a capacity limit — so only lessons general enough to be
  reconstructed survive. The Ville certificate then tests the *compressed* lesson
  set against held-out tasks: if the bottlenecked lessons generalise at least as
  well as the fat ones, the compression is kept.
- **Effect**: selects compositional, transferable lessons over overfit ones —
  directly the thing the yardstick rewards. The most on-theme and most novel.

### 2. Zone of proximal development — the curriculum (developmental psychology)

Vygotsky: learning is fastest on tasks just beyond current independent ability,
with scaffolding. Samaritan already has a frontier set with solvability
witnesses; it does not yet *select* by proximal difficulty.

- **Where**: `samaritan-corpus` (frontier set) + the episode sampler.
- **What changes**: sample the next episode from the band where current solve
  probability is ~0.3–0.7 — the proximal zone — estimated from the ledger's
  recent outcomes, rather than uniformly or by static difficulty. Scaffolding
  (a lesson, a hint) is added exactly when a task sits just above the zone.
- **Effect**: capability-per-compute. The cheapest, most grounded win; curriculum
  learning with a principled, non-arbitrary difficulty target.

### 3. Baldwin assimilation — learning becomes weights (evolutionary biology)

The Baldwin effect: behaviour learned within a lifetime, if repeatedly adaptive,
is assimilated into the genome. Here: a lesson that keeps earning its keep across
generations is *assimilated* into the model by a QDoRA fine-tune, then retired
from the prompt — it is now known, not looked up.

- **Where**: `samaritan-reflect` → `training/` (QDoRA) → the served model.
- **What changes**: track per-lesson lifetime utility across generations; when a
  lesson clears a certificate for *durable* usefulness, it becomes training data,
  is assimilated into weights, and drops out of the injected lesson set. The
  freed prompt budget makes room for new learning.
- **Effect**: the only lever here that moves base capability, so the only one that
  honestly touches HLE. Heaviest to build (needs the GPU training loop wired).

### 4. Kelly-criterion bet sizing on the certificate (economics / information theory)

Kelly (1956): to maximise long-run growth, size each bet proportional to your
edge. Samaritan's α-investing spends the error budget; it does not yet *size* the
spend by the strength of the evidence.

- **Where**: `samaritan-cert` (α-investing).
- **What changes**: allocate the α (and the compute) for evaluating a candidate
  by a Kelly fraction of the current e-value / edge, instead of a fixed slice.
  Strong candidates get more budget and commit sooner; weak ones are starved.
- **Effect**: faster, more compute-efficient acceptance without raising the
  false-certification rate. Precise, small, and principled.

### 5. Clonal selection — affinity maturation (immunology)

A B-cell that binds an antigen proliferates and *hypermutates locally*, and the
higher-affinity variants are selected. Mapped to the arena: an attack or defence
that lands proliferates and is hypermutated in its neighbourhood, concentrating
search where it is paying off — a local-search intensification the flat NRPA
rollout lacks.

- **Where**: `samaritan-search` / `samaritan-adversary`.
- **What changes**: on a landing, spawn a burst of local mutations of that
  attack/lesson (hypermutation) and select the best, rather than returning to
  breadth. Balances the arms race's explore/exploit.
- **Effect**: sharper exploitation of a found gradient; complements ZPD's
  exploration.

## Recommendation

Build in this order, because it front-loads the honest, cheap, measurable wins
and defers the heavy one:

1. **ZPD curriculum** (#2) — cheapest, most grounded, immediate capability-per-
   compute. Pure Rust in corpus + sampler.
2. **Iterated-learning bottleneck** (#1) — the unique contribution; targets
   generalisation, which is what the yardstick and any honest HLE gain need.
   Pure Rust in reflect + cert.
3. **Kelly sizing** (#4) — sharpen the certificate. Small, self-contained.
4. **Baldwin assimilation** (#3) — the base-capability lever, once the QDoRA
   loop is wired end to end (it is built but needs the GPU environment).
5. **Clonal selection** (#5) — if the arms race needs more intensification.

Every one of these ships behind the existing discipline: a change is a
*candidate*, it earns its place only through the Ville certificate on held-out
tasks, and the frozen core is untouched. The novelty is the **synthesis** — an
iterated-learning bottleneck feeding a Baldwin assimilation loop, curriculum-
scheduled at the zone of proximal development, with bets sized by Kelly and
gated by an anytime-valid certificate. That combination does not exist in the
literature. It will not make an 8B a 30% HLE model; it is a genuinely novel
recursive-self-improvement architecture, and it will move the numbers it can
honestly move.
