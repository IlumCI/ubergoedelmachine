#!/usr/bin/env python3
"""GRPO with no training framework - torch, transformers, peft, nothing else.

WHY THIS EXISTS. The TRL/unsloth version failed four separate times in a Colab
runtime, never once on the algorithm: an unconditional `mergekit` import from
trl/trainer/callbacks.py, a lazy loader that hid it until first use, mergekit's
own transitive chain, and torchao wheels built for cpython-310 loading under
3.13. Three layers of other people's dependency decisions, to run something that
fits on two pages.

GRPO is simple. For each prompt, sample G completions; grade them; the advantage
of a completion is how far its reward sits from its GROUP's mean, in units of the
group's spread; push up the log-probability of completions that beat their
siblings and down the ones that lose. No critic, no value head, no reference
model. That is the whole method.

What is deliberately NOT here:
  * a KL penalty to a frozen reference. It needs a second model in memory and
    matters over long runs; for a few hundred steps on a LoRA it is not what
    decides whether this works. Add it when a run is long enough to drift.
  * vLLM-accelerated rollouts. Correctness first, on a stack that imports.

The reward comes from the harness's own Rust grader, exactly as in
grpo_reasoning.py - the trainer never decides correctness itself.

    python training/grpo_standalone.py train.jsonl \\
        --grader ./target/release/examples/grade_batch \\
        --output adapters/reasoning-grpo

Every piece of maths here is unit-tested in training/test_grpo_standalone.py,
which needs no GPU and no model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Must match the prompt the model is EVALUATED under, or training optimises one
# format while scoring reads another.
REASONING_SYSTEM = (
    "You are answering a single hard exam question. Reason carefully but be "
    "EFFICIENT: do not re-derive or re-verify the same result over and over, and "
    "do not pad your working. Your output space is limited - if your reasoning is "
    "running long, stop and commit to your best answer instead of continuing to "
    "deliberate, and always leave room to finish. End your reply with two lines, "
    "exactly:\nAnswer: <your final answer, as short as the question allows>\n"
    "Confidence: <a number from 0 to 1>\n"
    "State a low confidence when unsure. A confident wrong answer is worse than an "
    "honest low-confidence one - but running out of space with no answer is worst "
    "of all, so always give the two closing lines."
)


# --------------------------------------------------------------- the maths ---
# Kept free of torch imports at module scope so the tests can exercise it on any
# machine. These three functions are the entire algorithm; everything else in
# this file is plumbing around them.


def group_advantages(
    rewards: list[float], eps: float = 1e-4, min_spread: float = 0.0
) -> list[float]:
    """How far each completion's reward sits from its group's mean, scaled.

    This is the whole of "group relative". There is no critic estimating a
    baseline - the siblings ARE the baseline, which is what makes GRPO cheap.

    A unanimous group returns all zeros, and that is correct rather than a
    degenerate case to paper over: if every completion earned the same reward,
    nothing in the group is evidence that one behaviour beat another. Dividing by
    a near-zero spread would manufacture enormous advantages out of floating
    point noise, so the epsilon floor matters.

    `min_spread` is the same argument in absolute units, and it exists because
    partial credit reintroduced the problem the epsilon floor was built to stop.
    Dividing by the standard deviation is scale-free: eight wrong rollouts where
    one reached four intermediates and another three differ by about 0.02, and
    normalising turns that into a full-sized +/-1.2 advantage. The model would
    then train as hard on one accidental number as on getting the answer right.
    A correct/incorrect split is worth 0.75 or more, so a floor here costs the
    binary signal nothing and discards only the amplified noise.
    """
    n = len(rewards)
    if n == 0:
        return []
    if min_spread and max(rewards) - min(rewards) < min_spread:
        return [0.0] * n
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def problem_weight(
    history: list[int],
    generations: int,
    age: int,
    *,
    exploit: float = 5.0,
    unseen: float = 0.3,
    floor: float = 0.02,
    half_life: float = 400.0,
) -> float:
    """How much a rollout spent on this problem is likely to teach.

    Uniform sampling spends most of the budget on problems that are already
    solved or entirely hopeless, because on this curriculum most of them are:
    the first d3 run got one gradient step in forty. Retiring a problem the
    moment it comes back unanimous is the crude fix, and it throws away two
    things - that a problem can BECOME trainable as the policy moves, and that a
    problem which disagrees every single time may be a plateau rather than a
    frontier.

    This follows Schmidhuber's curiosity formulation instead: the thing worth
    rewarding is learning PROGRESS, the first derivative of success, not the
    success rate itself. Three parts, and the weight is whichever is largest:

      spread     the last group disagreed, so a rollout here carries a gradient
                 right now
      progress   the success rate MOVED between the last two visits, so this is
                 where the policy is actually changing
      staleness  weight recovers as a problem goes unvisited, because "unanimous
                 once" is evidence about the policy that drew it, not a permanent
                 fact about the problem

    `history` is pass counts out of `generations`, oldest first. `age` is
    attempts since it was last drawn. Nothing ever reaches zero, so no problem is
    permanently dead.

    Movement is only counted above the binomial noise floor. At 8 rollouts a
    one-sample swing has standard deviation sqrt(8)/2 = 1.4, so a change of one
    correct answer is indistinguishable from resampling the same policy - and
    chasing it would be chasing the sampler.

    THE DEFAULTS ARE MEASURED, not chosen. Simulated over 520 draws on a
    450-problem population shaped like the observed run (52% solved, 26% out of
    reach, 22% in the band), against uniform sampling at 36.9%:

        exploit  unseen   gradient   distinct problems   top 5 got
            0.6     1.0      36.2%        138                 9%
            5.0     0.3      64.1%         80                20%
           20.0     0.3      75.2%         53                31%
           20.0     0.1      81.7%         39                46%

    The first row was the obvious parameterisation and it is worth NOTHING - an
    unexplored problem outranking a proven one means the sampler explores forever
    and never exploits, which on 450 problems is uniform sampling with extra
    steps. The last row doubles the yield again but puts nearly half the updates
    on five problems, and this run is measured on held-out problems, so that
    trade is not free. 5.0 nearly doubles the gradient per GPU-hour while keeping
    the spread of problems within sight of uniform.

    What the simulation could NOT test is the progress term: its latent
    difficulties are fixed, so nothing ever genuinely moves and only sampling
    noise does. Progress is kept because it is the part that distinguishes a
    problem the policy is actually learning from one stuck at a coin flip
    forever, and it costs nothing - but it is unvalidated, unlike `exploit`.
    """
    if not history:
        return unseen          # never tried: the only way to find the band at all
    g = max(generations, 1)
    last = history[-1]
    in_band = 1.0 if 0 < last < g else 0.0
    progress = 0.0
    if len(history) >= 2:
        moved = abs(history[-1] - history[-2]) - (g ** 0.5) / 2
        progress = max(0.0, moved) / g
    recovered = unseen * (1.0 - 0.5 ** (max(age, 0) / max(half_life, 1e-9)))
    return max(floor, exploit * (in_band + progress), recovered)


def completion_mask(prompt_len: int, total_len: int, pad_from: int | None = None) -> list[int]:
    """1 for tokens the policy generated, 0 for prompt and padding.

    The loss must not touch prompt tokens. Including them would train the model
    to predict the QUESTION - rewarding it for text it never chose, and diluting
    the signal from the tokens it did.
    """
    end = total_len if pad_from is None else min(pad_from, total_len)
    return [1 if prompt_len <= i < end else 0 for i in range(total_len)]


def sequence_logprob(token_logprobs: list[float], mask: list[int]) -> float:
    """Total log-probability of the completion, prompt and padding excluded."""
    return sum(lp for lp, m in zip(token_logprobs, mask) if m)


def policy_gradient_loss(
    seq_logprobs: list[float], advantages: list[float], mask_sizes: list[int]
) -> float:
    """-(advantage * logprob), length-normalised, averaged over the group.

    Length normalisation is not cosmetic. Without it a 6,000-token completion
    contributes a hundred times the gradient of a 60-token one purely for being
    long, so the optimiser learns length rather than correctness - and length is
    exactly the failure the SFT student already has.
    """
    if not seq_logprobs:
        return 0.0
    terms = []
    for lp, adv, n in zip(seq_logprobs, advantages, mask_sizes):
        terms.append(-adv * lp / max(n, 1))
    return sum(terms) / len(terms)


# ------------------------------------------------------------------ reward ---


def check_grader(grader_cmd: str) -> None:
    """Fail loudly when the grader command cannot run at all.

    shell=True on a path that does not exist exits 127 with EMPTY stdout, which
    arrives as "0 verdicts" - indistinguishable from a grader that ran and
    disagreed. The usual cause is a rebuilt working tree: re-extracting the
    bundle deletes target/, so a previously-good absolute path silently stops
    existing while the variable holding it does not.
    """
    # shlex, not split(): a naive split breaks any path containing a space and
    # reports a "missing" binary that is sitting right there.
    import shlex
    parts = shlex.split(grader_cmd, posix=(os.name != "nt"))
    if not parts:
        sys.exit("--grader is empty")
    first = parts[0]
    if "/" in first or "\\" in first:
        from pathlib import Path as _P
        if not _P(first).exists():
            sys.exit(
                f"grader not found: {first}\n"
                "The binary is gone but the path is not - usually because the "
                "bundle was re-extracted, which removes target/. Rebuild it "
                "(the cell that runs cargo build) and try again."
            )
        if not os.access(first, os.X_OK):
            sys.exit(f"grader is not executable: {first}")


def shaped_rewards(
    verdicts: list[tuple[bool, float | None]], shaping: float
) -> list[float]:
    """Correctness, plus partial credit among the completions that got it wrong.

    WHY. A binary reward carries no gradient when the whole group agrees, and on
    this curriculum it usually does: a 40-step run on d3 produced one update,
    because eight rollouts on a problem the model can do come back eight-correct
    and on one it cannot come back eight-wrong. But eight wrong answers are not
    equally wrong - one may have derived four of the five intermediate quantities
    before losing the thread. `step_recall` measures that against the generator's
    own recorded path, so the doomed group becomes a ranking instead of a tie.

    Partial credit applies ONLY to failures, which is the point rather than an
    optimisation. Among completions that are all correct there is no evidence one
    is better, and preferring whichever happened to match the generator's route
    would teach route-imitation with no gain in correctness. An all-correct group
    stays flat, gets retired, and its rollout budget goes somewhere it can learn.

    The invariant: recall is in [0,1] and shaping is well below 1, so the best
    possible wrong answer scores below the worst possible right one. Partial
    credit reorders failures; it never outranks being right.
    """
    out = []
    for ok, recall in verdicts:
        if ok:
            out.append(1.0)
        elif shaping and recall is not None:
            out.append(shaping * recall)
        else:
            out.append(0.0)
    return out


def grade(grader_cmd: str, items: list) -> list[tuple[bool, float | None]]:
    """Score completions through the harness's Rust grader.

    `items` are (given, row) pairs: the grader needs the question and the
    generator's recorded steps as well as the gold answer, because it scores
    partial progress along that path as well as the final answer.

    Returns (correct, step_recall) per item, where recall is None when the
    problem has too few distinct derived quantities to score progress against.

    One subprocess per group rather than per completion: the cost is process
    startup, and a step grades every rollout at once.
    """
    if not items:
        return []
    payload = "\n".join(
        json.dumps({
            "given": g,
            "answer": row["answer"],
            "answer_kind": row.get("answer_kind", "exactMatch"),
            "question": row.get("question", ""),
            "steps": row.get("steps") or [],
        })
        for g, row in items
    )
    try:
        proc = subprocess.run(
            grader_cmd, shell=True, input=payload,
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("warning: grader timed out; scoring this group 0", file=sys.stderr)
        return [(False, None)] * len(items)
    verdicts = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            recall = rec.get("step_recall")
            verdicts.append((
                bool(rec.get("correct", False)),
                float(recall) if recall is not None else None,
            ))
        except (json.JSONDecodeError, TypeError, ValueError):
            verdicts.append((False, None))
    if len(verdicts) != len(items):
        # Never silently misalign rewards with completions - that trains noise.
        # Print what the grader actually said. Swallowing stderr here is what
        # made a missing binary look like a disagreeing one.
        detail = (proc.stderr or "").strip()[-400:]
        print(
            f"warning: grader returned {len(verdicts)} verdicts for {len(items)} "
            f"completions (exit {proc.returncode}); scoring this group 0"
            + (f"\n  grader said: {detail}" if detail else "\n  grader said nothing"),
            file=sys.stderr,
        )
        return [(False, None)] * len(items)
    return verdicts


def reward_health(
    texts: list[str],
    groups: list[list[float]],
    budget: int,
    min_spread: float = 0.0,
    right: list[int] | None = None,
) -> str:
    """Whether the opening groups carry any gradient, and if not, why.

    A run of unanimous groups is not slow progress, it is no progress - and from
    the outside it looks exactly like a healthy run. Spread is counted PER GROUP,
    never pooled: five all-right groups and five all-wrong ones pool to a
    perfectly mixed set while producing zero advantage everywhere.
    """
    answered = sum(1 for t in texts if "answer:" in t.lower()) / max(len(texts), 1)
    # Spread on the REWARDS, not the verdicts: with step-recall shaping a group
    # of eight wrong answers carries a gradient whenever they got different
    # distances along the path, and counting booleans would call that dead. Uses
    # the same floor the trainer uses, or this would report gradient the trainer
    # then declines to take.
    floor = max(min_spread, 1e-9)
    mixed = sum(1 for g in groups if g and max(g) - min(g) >= floor)
    rate = mixed / max(len(groups), 1)

    # Shaping makes almost every group technically mixed, so "mixed" alone stops
    # being the interesting number. What matters is whether the model is ever
    # getting these right: a run where every group is all-wrong is ranking
    # failures forever and will never learn to answer, however healthy the
    # gradient looks.
    note = ""
    if right is not None and right:
        split = sum(1 for g, r in zip(groups, right) if 0 < r < len(g))
        allwrong = sum(1 for r in right if r == 0)
        note = (
            # Counted over ALL the groups, not over the mixed ones - an all-wrong
            # group can be flat as well as ranked, so these do not sum to `mixed`
            # and saying "of these" made them look like they should.
            f" Across all {len(groups)}: {split} had a correct/incorrect split, "
            f"{allwrong} got nothing right."
        )
        if allwrong == len(groups):
            return (
                f"*** every one of {len(groups)} opening groups was entirely wrong. "
                f"Partial credit still gives a gradient, so this will train - but "
                f"toward reaching intermediates, never toward finishing, because "
                f"nothing here has ever shown it what finishing looks like. Mix in "
                f"an easier DIFFICULTY so some groups land correct."
            )
    # A THIRD, not one. `if mixed:` passed a run where 1 group in 10 had spread,
    # and it went on to spend five hours producing a single gradient step -
    # every other step skipped as unanimous. Spread has to be common enough that
    # most rollout compute buys a gradient, or the run is mostly a generator.
    if rate >= 0.3:
        return (
            f"reward health: {mixed}/{len(groups)} opening groups had spread, "
            f"{answered:.0%} of rollouts finished with an answer. Training has "
            "something to learn from." + note
        )
    if mixed and answered >= 0.5:
        return (
            f"*** almost no gradient: only {mixed}/{len(groups)} groups had spread, "
            f"and {answered:.0%} of rollouts finished. The problems are mostly "
            "SOLVED-OR-DOOMED rather than uncertain - 8 rollouts on an easy item "
            "give 8 correct, on a hard one 8 wrong, and neither teaches anything. "
            "Check --shaping is non-zero and the dataset carries 'steps': partial "
            "credit is what turns a doomed group into a ranking. Otherwise mix in "
            "harder DIFFICULTY, or select problems that actually produce spread."
        )
    if answered < 0.5:
        return (
            f"*** no gradient: {len(groups)} unanimous groups, and only "
            f"{answered:.0%} of rollouts emitted an 'Answer:' line. They are being "
            f"CUT OFF, not getting it wrong. Raise --max-new above {budget}."
        )
    return (
        f"*** no gradient: {len(groups)} unanimous groups, but {answered:.0%} "
        "of rollouts finished. The problems are uniformly too easy or too hard - "
        "regenerate at a different DIFFICULTY."
    )


# ------------------------------------------------------------------- data ----


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"dataset not found: {path}")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"malformed JSON on line {i} of {path}: {e}")
        if r.get("question") and r.get("answer"):
            rows.append(r)
    if not rows:
        sys.exit(f"no usable rows in {path}")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", type=Path)
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--output", type=Path, default=Path("adapters/reasoning-grpo"))
    p.add_argument(
        "--grader",
        default="cargo run -q -p samaritan-corpus --example grade_batch",
        help="command reading grading JSONL on stdin, writing verdicts on stdout",
    )
    # 8192 covers the base model's full measured demand on generated d3 - its
    # longest correct answer was 7,121 tokens - so truncation is not part of the
    # reward. A budget near the median would make roughly half the signal "did it
    # fit" rather than "was it right".
    p.add_argument("--max-new", type=int, default=8192)
    p.add_argument("--max-prompt", type=int, default=2048)
    p.add_argument("--generations", type=int, default=8,
                   help="completions per prompt; a group needs SPREAD to teach anything")
    # Rollouts dominate, and without vLLM they are plain HF generate: 8 completions
    # of up to 8k tokens is ~65k tokens per step, so expect MINUTES per step rather
    # than seconds. 40 steps is a first run that shows whether reward moves; it is
    # not a finished model. Raise it once the loop is proven.
    p.add_argument("--steps", type=int, default=40)
    # 5e-6 is a FULL-MODEL GRPO rate. On a rank-16 LoRA whose B matrix starts at
    # zero, forty steps at 5e-6 move the adapter by roughly nothing: the run
    # finishes, the log looks healthy, and the weights are indistinguishable from
    # the base. 2e-5 is still conservative - there is no KL penalty to a frozen
    # reference here, so a rate high enough to matter is also a rate high enough
    # to drift - but it is high enough for forty steps to leave a mark.
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument(
        "--exploit", type=float, default=5.0,
        help="how hard to favour problems already shown to sit in the trainable "
             "band. Simulated on a 450-problem population shaped like the "
             "measured run: 0.6 is worth nothing (36%% against uniform's 37%%), "
             "5.0 gives 64%% over 80 distinct problems, 20.0 gives 75%% over 53. "
             "Higher buys gradient per GPU-hour and pays in concentration, and "
             "this run is scored on held-out problems.",
    )
    p.add_argument(
        "--pool", type=int, default=0, metavar="N",
        help="train on a stratified N-problem subset (0 = all of it). No longer "
             "needed - at --exploit 5 the sampler works on the full 450 - but it "
             "raises the revisit rate if you want learning progress measured "
             "within a single session rather than across four.",
    )
    p.add_argument(
        "--min-spread", type=float, default=0.05,
        help="a group whose rewards differ by less than this is treated as "
             "unanimous. Stops normalisation from amplifying a one-anchor "
             "difference into a full-sized advantage; a correct/incorrect split "
             "is 0.75 or more, so the binary signal is untouched.",
    )
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="rollout temperature; too low collapses the group to one answer")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    # Everything below exists so an UNATTENDED run is worth starting. Colab
    # preempts, GPUs run out of memory, and a flat reward burns hours looking
    # exactly like a healthy run. None of those should cost the whole session.
    p.add_argument("--resume", action="store_true",
                   help="continue from the adapter already in --output, if one is there")
    p.add_argument("--abort-if-flat", action="store_true", default=True,
                   help="stop when the opening groups carry no gradient, rather than "
                        "spending the rest of the run on a zero signal")
    p.add_argument("--no-abort-if-flat", dest="abort_if_flat", action="store_false")
    p.add_argument("--attempts-per-step", type=int, default=6,
                   help="how many prompts may be tried per gradient step before "
                        "giving up. A unanimous group teaches nothing but costs "
                        "the same rollouts, so this bounds the waste")
    p.add_argument("--min-generations", type=int, default=2,
                   help="on CUDA OOM the group is halved and retried, down to this; "
                        "a smaller group still teaches something, a dead run does not")
    p.add_argument(
        "--shaping", type=float, default=0.25,
        help="weight on step-recall partial credit for WRONG completions "
             "(0 disables). Must stay below 1 so a wrong answer can never "
             "outscore a right one.",
    )
    p.add_argument(
        "--probe", type=int, default=0, metavar="N",
        help="generate and grade N groups, report the reward spread, and stop "
             "without training. Answers the one question the offline tests "
             "cannot: whether REAL rollouts differ from each other enough to "
             "carry a gradient. Cheap next to finding out an hour into a run.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="after the first group, force one backward pass with synthetic "
             "advantages and stop. Proves checkpointing + LoRA + masking "
             "actually produce a gradient - the part a flat-group smoke test "
             "never touches.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="verify the dataset and reward path, then stop before the GPU")
    return p.parse_args()


# ------------------------------------------------------------------- train ---


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset)
    print(f"dataset: {len(rows)} problems from {args.dataset}")
    if args.pool and args.pool < len(rows):
        # Deterministic in --seed, and taken BEFORE anything else touches the
        # rows, so a resumed session draws the same pool and its stored history
        # still refers to problems that are in it. Stratified by family, because
        # a random 60 of 450 can easily miss a family entirely and then the run
        # silently trains on nine tenths of the curriculum.
        import random
        by_fam: dict[str, list[dict]] = {}
        for r in rows:
            by_fam.setdefault(r["id"].split("-")[1], []).append(r)
        rng = random.Random(args.seed)
        for v in by_fam.values():
            rng.shuffle(v)
        picked, fams = [], sorted(by_fam)
        while len(picked) < args.pool and any(by_fam[f] for f in fams):
            for f in fams:
                if by_fam[f] and len(picked) < args.pool:
                    picked.append(by_fam[f].pop())
        rows = sorted(picked, key=lambda r: r["id"])
        print(f"  --pool {args.pool}: training on {len(rows)} of them, "
              f"{len(fams)} families, so learning progress has revisits to measure")

    # Prove the reward path BEFORE any GPU time. A reward function that silently
    # returns 0 trains the model to do nothing, slowly and expensively.
    check_grader(args.grader)
    probe = [
        ("<think>x</think>\nAnswer: " + rows[0]["answer"], rows[0]),
        ("<think>x</think>\nAnswer: __definitely_wrong__", rows[0]),
    ]
    got = [ok for ok, _ in grade(args.grader, probe)]
    if got != [True, False]:
        sys.exit(
            f"grader sanity check failed: expected [True, False], got {got}.\n"
            f"Command was: {args.grader}\nFix this first - a broken reward trains nothing."
        )
    print(f"grader OK via: {args.grader}")

    # Shaping is only as good as the traces behind it, and a dataset generated
    # before the W7 families recorded their solution paths has none. Say so here
    # rather than letting every recall come back None and the run look flat for
    # an unrelated reason.
    if args.shaping:
        traced = sum(1 for r in rows if r.get("steps"))
        print(f"step traces: {traced}/{len(rows)} problems, shaping weight {args.shaping}")
        if traced == 0:
            sys.exit(
                "--shaping is on but no problem carries 'steps'. This dataset was "
                "generated before the families recorded their solution paths - "
                "regenerate it, or pass --shaping 0 to train on binary reward "
                "alone (and expect unanimous groups)."
            )

    if args.dry_run:
        print("--dry-run: dataset and reward path verified; stopping before model load.")
        return

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        sys.exit("no GPU visible - refusing to start; this needs one")

    # Which GPU, said out loud. Generation dominates the wall clock here - a
    # measured run spent 11 minutes per prompt - and the first question about a
    # slow run is whether it landed on the accelerator it asked for. A T4 has no
    # native bfloat16 at all, so it would emulate every matmul in this script;
    # that is a 5-10x difference and no amount of tuning elsewhere recovers it.
    props = torch.cuda.get_device_properties(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.0f} GB, "
          f"bfloat16 {'native' if bf16 else 'EMULATED - expect it to crawl'}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so generated tokens are a contiguous suffix

    # .to(device), NOT device_map=device. `device_map` routes the load through
    # accelerate, which attaches a hook to every submodule to check and move
    # tensors; on a model that fits on one GPU those hooks buy nothing and run
    # hundreds of times per forward. Generation is essentially the whole wall
    # clock here - a measured A100 run managed 68 tok/s across 8 streams, 104 ms
    # per decode step, when the weight read alone should cost about 5 ms - so
    # per-forward overhead is the first thing to remove.
    #
    # attn_implementation is named rather than left to the default, because the
    # default is a function of the installed torch and transformers and this run
    # should not silently change speed when Colab updates a wheel.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
    except Exception as e:   # noqa: BLE001 - any refusal of the argument, not just ours
        # Broad on purpose. This runtime is on transformers 5.x and the exact
        # exception an unsupported attn_implementation raises is a moving target;
        # the fallback costs one extra load and the alternative is the run dying
        # at model load after the grader and the dataset already checked out.
        print(f"note: sdpa attention unavailable ({type(e).__name__}: {e}); "
              "falling back to the default", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16
        ).to(device)
    print(f"attention: {getattr(model.config, '_attn_implementation', 'unknown')}")

    # Resume before creating a fresh adapter, or a preempted run starts over from
    # random weights while looking like it continued.
    resumed = args.resume and (args.output / "adapter_config.json").exists()
    if resumed:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.output), is_trainable=True)
        print(f"resumed the adapter in {args.output}")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ),
        )
    model.print_trainable_parameters()

    # Activation memory is the binding constraint in the backward pass: a full
    # forward WITH gradients over prompt plus up to 8k completion tokens does not
    # fit alongside the weights and the generation KV cache otherwise. Checkpointing
    # trades recomputation for memory, which is the right trade when the alternative
    # is not running.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # or checkpointing detaches the LoRA graph
    model.config.use_cache = True        # keep KV caching for generation

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # Adam's moments are part of the run's state, not scratch. A session that
    # resumes without them restarts the optimiser cold - the first few steps
    # after every preemption take badly scaled updates, and on a run that only
    # fits in a Colab session three times over, that is most of the run.
    opt_state = args.output / "optimizer.pt"
    if resumed and opt_state.exists():
        try:
            opt.load_state_dict(torch.load(opt_state, map_location=device))
            print(f"resumed optimiser state from {opt_state}")
        except Exception as e:  # a corrupt half-written file must not end the run
            print(f"warning: could not load {opt_state} ({e}); "
                  "continuing with a fresh optimiser", file=sys.stderr)
    elif resumed:
        print("note: no optimizer.pt beside the adapter - this resumes the "
              "weights but restarts Adam cold")

    gens = args.generations
    attempts = 0      # prompts tried
    trained = 0       # prompts that actually produced a gradient
    seen_texts: list[str] = []
    seen_groups: list[list[float]] = []
    seen_right: list[int] = []
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    # --steps counts GRADIENT steps, not prompts tried. A unanimous group costs
    # the same rollouts and teaches nothing, so letting it consume a step is how
    # a 40-step run produced one update. Problems that come back unanimous are
    # also retired: on this curriculum a prompt is usually solved-or-doomed
    # rather than uncertain, so re-drawing it buys the same nothing again.
    step = 0
    probed: list[tuple[str, int, list[float]]] = []

    # Per-problem visit history, keyed by problem id rather than row index so a
    # regenerated dataset does not silently attach one problem's history to
    # another. It lives beside the adapter and the optimiser state, because the
    # whole point of measuring learning progress is that it accumulates - and
    # this run only fits in a Colab session three or four times over. Within one
    # 40-group session almost nothing is revisited; across four, it is.
    hist_path = args.output / "curriculum.json"
    seen: dict[str, list[int]] = {}
    last_drawn: dict[str, int] = {}
    if args.resume and hist_path.exists():
        try:
            saved = json.loads(hist_path.read_text(encoding="utf-8"))
            seen = {k: list(v) for k, v in saved.get("history", {}).items()}
            known = {r["id"] for r in rows}
            stale = [k for k in seen if k not in known]
            for k in stale:
                del seen[k]
            print(f"curriculum: {len(seen)} problems with history"
                  + (f" ({len(stale)} dropped - not in this dataset)" if stale else ""))
        except (json.JSONDecodeError, OSError, TypeError) as e:
            print(f"warning: could not read {hist_path} ({e}); starting fresh",
                  file=sys.stderr)

    def save_history() -> None:
        args.output.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(
            json.dumps({"generations": args.generations, "history": seen}, indent=1),
            encoding="utf-8",
        )
    max_attempts = args.probe if args.probe else args.steps * args.attempts_per_step
    while trained < args.steps and attempts < max_attempts:
        # Sample by learning progress rather than uniformly. No pool to exhaust
        # and no reopening: every weight stays positive, and a problem that came
        # back unanimous recovers on its own as the policy moves away from the
        # one that drew it.
        weights = [
            problem_weight(seen.get(r["id"], []), args.generations,
                           attempts - last_drawn.get(r["id"], 0),
                           exploit=args.exploit)
            for r in rows
        ]
        idx = int(torch.multinomial(torch.tensor(weights), 1, generator=g).item())
        row = rows[idx]
        past = seen.get(row["id"], [])
        chat = [
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": row["question"]},
        ]
        prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.max_prompt).to(device)
        prompt_len = enc.input_ids.shape[1]

        # --- rollouts: G samples from the same prompt --------------------------
        # Generation wants the KV cache; the checkpointed training forward below
        # cannot use it. Toggling explicitly beats letting transformers warn and
        # change the setting underneath us.
        model.eval()
        model.config.use_cache = True
        why = "new" if not past else f"seen {len(past)}x {past[-3:]}"
        print(f"step {step:>4}  generating {gens} x <={args.max_new} tok "
              f"({row['domain']}, {why}) ...", end="", flush=True)
        t_gen = time.time()
        # OOM is a survivable condition, not the end of the run. The KV cache
        # scales with the group size, so halving it is the one knob that reliably
        # helps - and a smaller group still teaches something, where a crashed run
        # at step 3 of an unattended night teaches nothing.
        out = None
        while out is None:
            try:
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        num_return_sequences=gens,
                        pad_token_id=tok.pad_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if gens <= args.min_generations:
                    print(f"step {step}: OOM even at {gens} generations - skipping",
                          file=sys.stderr, flush=True)
                    break
                gens = max(args.min_generations, gens // 2)
                print(f"step {step}: OOM, retrying with {gens} generations",
                      file=sys.stderr, flush=True)
        if out is None:
            continue
        completions = out[:, prompt_len:]
        texts = tok.batch_decode(completions, skip_special_tokens=True)
        n_tok = int((completions != tok.pad_token_id).sum())
        print(f" {n_tok:,} tok in {time.time()-t_gen:.0f}s, grading ...",
              end="", flush=True)

        # --- reward ------------------------------------------------------------
        verdicts = grade(args.grader, [(t, row) for t in texts])
        rewards = shaped_rewards(verdicts, args.shaping)
        advantages = group_advantages(rewards, min_spread=args.min_spread)
        n_right = sum(1 for ok, _ in verdicts if ok)
        seen.setdefault(row["id"], []).append(n_right)
        last_drawn[row["id"]] = attempts

        if len(seen_groups) < 10:
            seen_texts += texts
            seen_groups.append(rewards)
            seen_right.append(n_right)
            if len(seen_groups) == 10:
                msg = reward_health(seen_texts, seen_groups, args.max_new,
                                    args.min_spread, seen_right)
                dead = msg.startswith("***")
                print(f"\n{msg}\n",
                      file=sys.stderr if dead else sys.stdout, flush=True)
                # Not during a probe: its whole job is to report, and exiting
                # here would suppress the summary that says what to change.
                if dead and args.abort_if_flat and not args.probe:
                    # Ten unanimous groups means the remaining steps would train on
                    # a zero gradient. Unattended, that silently spends the whole
                    # session; stopping here leaves the reason on screen and the
                    # GPU free. --no-abort-if-flat to override.
                    print("aborting: the message above says what to change. "
                          "Pass --no-abort-if-flat to run anyway.",
                          file=sys.stderr, flush=True)
                    sys.exit(2)

        # --smoke: the old smoke test generated two 512-token completions, both
        # truncated, got a flat group, skipped the backward pass and printed
        # success. It proved generation and grading and never once touched the
        # gradient path - which is the part most likely to break, because
        # checkpointing, LoRA and masking all interact there. Synthetic
        # advantages force that path to run. It is deliberately NOT learning:
        # the numbers are made up, and the adapter it produces is discarded.
        if args.smoke:
            advantages = [1.0 if i % 2 == 0 else -1.0 for i in range(len(texts))]
            print(f"\n--smoke: forcing one backward with synthetic advantages "
                  f"{advantages} (not learning - testing the machinery)", flush=True)

        attempts += 1

        # --probe: record and move on. No backward, no optimiser, nothing saved.
        if args.probe:
            lo, hi = min(rewards), max(rewards)
            probed.append((row["id"], n_right, rewards))
            print(f" {n_right}/{len(rewards)} right, reward {lo:.3f}..{hi:.3f} "
                  f"(spread {hi - lo:.3f})  [{attempts}/{args.probe}]", flush=True)
            continue

        # A flat group carries no information; skip the backward pass entirely
        # rather than spending it on a zero gradient.
        if all(a == 0.0 for a in advantages):
            print(f" {n_right}/{len(verdicts)} right, reward "
                  f"{sum(rewards)/len(rewards):.2f} (unanimous - no gradient; "
                  f"{trained}/{args.steps} trained, "
                  f"{attempts}/{max_attempts} tried)", flush=True)
            continue
        trained += 1
        step = trained - 1

        # --- policy gradient ---------------------------------------------------
        model.train()
        model.config.use_cache = False   # incompatible with gradient checkpointing
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(len(advantages)):
            seq = out[i : i + 1]
            attn = (seq != tok.pad_token_id).long()
            logits = model(input_ids=seq, attention_mask=attn).logits[:, :-1]
            targets = seq[:, 1:]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]

            # Completion tokens only: prompt tokens were never chosen by the
            # policy, and padding is not a choice at all.
            keep = torch.zeros_like(tok_lp, dtype=torch.bool)
            keep[prompt_len - 1 :] = True
            keep &= targets[0] != tok.pad_token_id
            n = int(keep.sum())
            if n == 0:
                continue

            seq_lp = tok_lp[keep].sum() / n   # length-normalised, see the loss note
            loss = -advantages[i] * seq_lp / len(advantages)
            loss.backward()
            total += float(loss.detach())   # detached: this is for the log only

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        if args.smoke:
            # A zero gradient norm here means the backward ran and reached
            # nothing: enable_input_require_grads missing, checkpointing having
            # detached the LoRA graph, or every token masked out. That failure is
            # silent in a real run - the loss prints, the step counts, and the
            # adapter never moves.
            if not float(gnorm) > 0:
                sys.exit(
                    f"--smoke FAILED: loss {total:+.4f} but gradient norm is "
                    f"{float(gnorm)}. The backward pass reached no trainable "
                    "parameter. Do not start a real run."
                )
            print(f"\n--smoke PASSED: gradient norm {float(gnorm):.4f} reached "
                  f"the LoRA weights; backward, checkpointing and masking all "
                  f"work. Nothing saved.", flush=True)
            return
        # Correct count and mean reward are now different numbers: with shaping a
        # group can move the policy while getting nothing right, which is exactly
        # the case this was built for. Printing only the mean would hide it.
        print(f" {n_right}/{len(verdicts)} right, reward "
              f"{sum(rewards)/len(rewards):.2f}  loss {total:+.4f}  "
              f"adv {min(advantages):+.2f}..{max(advantages):+.2f}  "
              f"[{time.time()-t_gen:.0f}s]", flush=True)

        if args.save_every and trained % args.save_every == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(args.output))
            torch.save(opt.state_dict(), opt_state)
            save_history()
            print(f"  checkpoint -> {args.output}", flush=True)

    if args.probe:
        # The question this answers: do real rollouts differ from each other?
        # Every offline test used completions I wrote by hand, which are
        # guaranteed to differ. Nothing before this point proves the model
        # produces spread on its own.
        n = len(probed)
        trainable = sum(1 for _, _, r in probed if max(r) - min(r) >= args.min_spread)
        split = sum(1 for _, k, r in probed if 0 < k < len(r))
        ranked = sum(1 for _, k, r in probed
                     if k == 0 and max(r) - min(r) >= args.min_spread)
        flat = n - trainable
        print(f"\n--- probe: {n} groups, no training ---")
        print(f"  would carry a gradient : {trainable}/{n}  ({trainable/max(n,1):.0%})")
        print(f"    of which correct/wrong splits : {split}")
        print(f"    of which all-wrong, ranked by partial credit : {ranked}")
        print(f"  flat (spread < {args.min_spread}) : {flat}")
        if trainable / max(n, 1) < 0.3:
            print("\n*** under a third would train. A real run would abort at step "
                  "10, and should. Do not start it.")
        elif split == 0:
            print("\n*** nothing was ever answered correctly. Partial credit will "
                  "train it toward reaching intermediates and never toward "
                  "finishing. Mix in an easier difficulty first.")
        else:
            print(f"\nlooks workable: {trainable}/{n} groups carry a gradient and "
                  f"{split} of them contain a correct answer to learn from.")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    torch.save(opt.state_dict(), opt_state)
    save_history()
    tok.save_pretrained(str(args.output))
    print(f"\nsaved GRPO adapter to {args.output} "
          f"({trained} gradient steps this session, {attempts} prompts tried)")

    # What the curriculum knows now. Across one session this is mostly a list of
    # first impressions; it earns its keep over the three or four sessions this
    # run takes, which is why it is stored beside the adapter.
    if seen:
        visits = [len(v) for v in seen.values()]
        revisited = sum(1 for v in visits if v > 1)
        band = sum(1 for v in seen.values() if 0 < v[-1] < args.generations)
        dead = sum(1 for v in seen.values() if len(v) > 1 and all(
            x == 0 or x == args.generations for x in v))
        print(f"curriculum: {len(seen)}/{len(rows)} problems tried, "
              f"{revisited} more than once, {sum(visits)} visits total")
        print(f"  {band} last came back mixed - these are the trainable band")
        print(f"  {dead} were unanimous every visit - solved or out of reach")
        if revisited < 5:
            print("  (too few revisits to measure learning progress yet; it needs "
                  "several sessions, or a smaller --pool)")


if __name__ == "__main__":
    main()
