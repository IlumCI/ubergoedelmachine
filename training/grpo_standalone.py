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


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """How far each completion's reward sits from its group's mean, scaled.

    This is the whole of "group relative". There is no critic estimating a
    baseline - the siblings ARE the baseline, which is what makes GRPO cheap.

    A unanimous group returns all zeros, and that is correct rather than a
    degenerate case to paper over: if every completion earned the same reward,
    nothing in the group is evidence that one behaviour beat another. Dividing by
    a near-zero spread would manufacture enormous advantages out of floating
    point noise, so the epsilon floor matters.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


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


def grade(grader_cmd: str, items: list) -> list[bool]:
    """Score (given, gold, kind) triples through the harness's Rust grader.

    One subprocess per group rather than per completion: the cost is process
    startup, and a step grades every rollout at once.
    """
    if not items:
        return []
    payload = "\n".join(
        json.dumps({"given": g, "answer": a, "answer_kind": k}) for g, a, k in items
    )
    try:
        proc = subprocess.run(
            grader_cmd, shell=True, input=payload,
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("warning: grader timed out; scoring this group 0", file=sys.stderr)
        return [False] * len(items)
    verdicts = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            verdicts.append(bool(json.loads(line).get("correct", False)))
        except json.JSONDecodeError:
            verdicts.append(False)
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
        return [False] * len(items)
    return verdicts


def reward_health(texts: list[str], groups: list[list[bool]], budget: int) -> str:
    """Whether the opening groups carry any gradient, and if not, why.

    A run of unanimous groups is not slow progress, it is no progress - and from
    the outside it looks exactly like a healthy run. Spread is counted PER GROUP,
    never pooled: five all-right groups and five all-wrong ones pool to a
    perfectly mixed set while producing zero advantage everywhere.
    """
    answered = sum(1 for t in texts if "answer:" in t.lower()) / max(len(texts), 1)
    mixed = sum(1 for g in groups if len(set(g)) > 1)
    rate = mixed / max(len(groups), 1)
    # A THIRD, not one. `if mixed:` passed a run where 1 group in 10 had spread,
    # and it went on to spend five hours producing a single gradient step -
    # every other step skipped as unanimous. Spread has to be common enough that
    # most rollout compute buys a gradient, or the run is mostly a generator.
    if rate >= 0.3:
        return (
            f"reward health: {mixed}/{len(groups)} opening groups had spread, "
            f"{answered:.0%} of rollouts finished with an answer. Training has "
            "something to learn from."
        )
    if mixed and answered >= 0.5:
        return (
            f"*** almost no gradient: only {mixed}/{len(groups)} groups had spread, "
            f"and {answered:.0%} of rollouts finished. The problems are mostly "
            "SOLVED-OR-DOOMED rather than uncertain - 8 rollouts on an easy item "
            "give 8 correct, on a hard one 8 wrong, and neither teaches anything. "
            "Mix in harder DIFFICULTY, or select problems that actually produce "
            "spread."
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
    p.add_argument("--lr", type=float, default=5e-6)
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
    p.add_argument("--dry-run", action="store_true",
                   help="verify the dataset and reward path, then stop before the GPU")
    return p.parse_args()


# ------------------------------------------------------------------- train ---


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset)
    print(f"dataset: {len(rows)} problems from {args.dataset}")

    # Prove the reward path BEFORE any GPU time. A reward function that silently
    # returns 0 trains the model to do nothing, slowly and expensively.
    check_grader(args.grader)
    kind = rows[0].get("answer_kind", "exactMatch")
    probe = [
        ("<think>x</think>\nAnswer: " + rows[0]["answer"], rows[0]["answer"], kind),
        ("<think>x</think>\nAnswer: __definitely_wrong__", rows[0]["answer"], kind),
    ]
    got = grade(args.grader, probe)
    if got != [True, False]:
        sys.exit(
            f"grader sanity check failed: expected [True, False], got {got}.\n"
            f"Command was: {args.grader}\nFix this first - a broken reward trains nothing."
        )
    print(f"grader OK via: {args.grader}")

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

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so generated tokens are a contiguous suffix

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map=device
    )

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

    gens = args.generations
    attempts = 0      # prompts tried
    trained = 0       # prompts that actually produced a gradient
    seen_texts: list[str] = []
    seen_groups: list[list[bool]] = []
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    # --steps counts GRADIENT steps, not prompts tried. A unanimous group costs
    # the same rollouts and teaches nothing, so letting it consume a step is how
    # a 40-step run produced one update. Problems that come back unanimous are
    # also retired: on this curriculum a prompt is usually solved-or-doomed
    # rather than uncertain, so re-drawing it buys the same nothing again.
    step = 0
    retired: set[int] = set()
    max_attempts = args.steps * args.attempts_per_step
    while trained < args.steps and attempts < max_attempts:
        live = [i for i in range(len(rows)) if i not in retired]
        if not live:
            print("every problem has come back unanimous at least once; "
                  "reopening the pool", flush=True)
            retired.clear()
            live = list(range(len(rows)))
        idx = live[int(torch.randint(len(live), (1,), generator=g).item())]
        row = rows[idx]
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
        print(f"step {step:>4}  generating {gens} x <={args.max_new} tok "
              f"({row['domain']}) ...", end="", flush=True)
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
        gold, kind = row["answer"], row.get("answer_kind", "exactMatch")
        verdicts = grade(args.grader, [(t, gold, kind) for t in texts])
        rewards = [1.0 if v else 0.0 for v in verdicts]
        advantages = group_advantages(rewards)

        if len(seen_groups) < 10:
            seen_texts += texts
            seen_groups.append(verdicts)
            if len(seen_groups) == 10:
                msg = reward_health(seen_texts, seen_groups, args.max_new)
                dead = msg.startswith("***")
                print(f"\n{msg}\n",
                      file=sys.stderr if dead else sys.stdout, flush=True)
                if dead and args.abort_if_flat:
                    # Ten unanimous groups means the remaining steps would train on
                    # a zero gradient. Unattended, that silently spends the whole
                    # session; stopping here leaves the reason on screen and the
                    # GPU free. --no-abort-if-flat to override.
                    print("aborting: the message above says what to change. "
                          "Pass --no-abort-if-flat to run anyway.",
                          file=sys.stderr, flush=True)
                    sys.exit(2)

        # A flat group carries no information; skip the backward pass entirely
        # rather than spending it on a zero gradient.
        attempts += 1
        if all(a == 0.0 for a in advantages):
            retired.add(idx)
            print(f" reward {sum(rewards)/len(rewards):.2f} (unanimous - no "
                  f"gradient; {trained}/{args.steps} trained, "
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
            total += float(loss)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        print(f" reward {sum(rewards)/len(rewards):.2f}  loss {total:+.4f}  "
              f"adv {min(advantages):+.2f}..{max(advantages):+.2f}  "
              f"[{time.time()-t_gen:.0f}s]", flush=True)

        if args.save_every and trained % args.save_every == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(args.output))
            print(f"  checkpoint -> {args.output}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tok.save_pretrained(str(args.output))
    print(f"\nsaved GRPO adapter to {args.output}")


if __name__ == "__main__":
    main()
