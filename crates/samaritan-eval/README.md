# samaritan-eval — the out-of-arena capability probe

Everything else Samaritan measures is on the repo surface it trains against.
This measures something it *cannot* tune to: Humanity's Last Exam, closed-form
reasoning it has never seen. A rising HLE score is capability that generalised
out of the environment — the signal the three-arm experiment is trying to read,
and the capability half of the public-repo milestone.

It is a **dashboard signal, not a grant**. The kernel makes HLE necessary but
never sufficient for a capability tier; a score alone unlocks nothing.

## What's here

- `lib.rs` — the pure, tested scoring core: `load`, `grade`, `tally`.
- `examples/hle.rs` — the live runner: queries the local model, scores, and
  records an `HleEvaluated` row the milestone gate reads.

## Running it

```powershell
$env:HLE_DATASET = "hle.jsonl"    # you supply this — see below
$env:LIMIT = "50"                 # the model is slow; cap the count
$env:LEDGER_DB = "run.db"         # optional: record the score
cargo run -p samaritan-eval --example hle
```

## You supply the dataset

HLE is gated and licensed, so this crate ships the **grader, not the
questions** — the same rule as the knowledge base and the task corpus. Provide
a JSONL, one object per line:

```json
{"id": "q1", "question": "…", "answer": "42", "answer_type": "exactMatch"}
{"id": "q2", "question": "…", "answer": "C", "answer_type": "multipleChoice"}
```

Extra columns are ignored, so a full HLE export loads as-is. The text-only
subset is the right starting point; this grader does not read images.

## The grading caveat — read this before trusting the number

Official HLE grades free-form answers with an **LLM judge**. This uses
normalised string / token matching: lowercase, strip a "the answer is" preface,
compare. It is cheap, deterministic, and fine for a *directional* signal — is
the score moving as the model changes — but it will **under-credit** a correct
answer phrased unusually, so the absolute number reads low versus an official
run. The recorded `HleEvaluated` event names the method, so the ledger never
passes this off as an official score.

If you want the official number, run the real HLE judge harness separately and
append the `HleEvaluated` row yourself; the milestone gate reads whatever score
is in the ledger.

## Why the score won't move on its own

Samaritan's level-1 self-improvement tunes the *agent's repo-task policy* —
lessons, prompts, thresholds. That barely touches HLE, which is about what the
base model *knows*. Two levers move it: **weight training** (the QDoRA path) and
**tool/knowledge use** at eval time. If HLE = 10% is the milestone, the pathway
that targets it has to be real work, not a number that drifts up by itself.
