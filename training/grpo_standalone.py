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
import subprocess
import sys
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
        print(
            f"warning: grader returned {len(verdicts)} verdicts for {len(items)} "
            "completions; scoring this group 0",
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
    if mixed:
        return (
            f"reward health: {mixed}/{len(groups)} opening groups had spread, "
            f"{answered:.0%} of rollouts finished with an answer. Training has "
            "something to learn from."
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
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="rollout temperature; too low collapses the group to one answer")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
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
    model.config.use_cache = True

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    seen_texts: list[str] = []
    seen_groups: list[list[bool]] = []
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    for step in range(args.steps):
        row = rows[int(torch.randint(len(rows), (1,), generator=g).item())]
        chat = [
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": row["question"]},
        ]
        prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.max_prompt).to(device)
        prompt_len = enc.input_ids.shape[1]

        # --- rollouts: G samples from the same prompt --------------------------
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                num_return_sequences=args.generations,
                pad_token_id=tok.pad_token_id,
            )
        completions = out[:, prompt_len:]
        texts = tok.batch_decode(completions, skip_special_tokens=True)

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
                print(f"\n{msg}\n",
                      file=sys.stderr if msg.startswith("***") else sys.stdout, flush=True)

        # A flat group carries no information; skip the backward pass entirely
        # rather than spending it on a zero gradient.
        if all(a == 0.0 for a in advantages):
            print(f"step {step:>4}  reward {sum(rewards)/len(rewards):.2f}  "
                  f"(unanimous - skipped)", flush=True)
            continue

        # --- policy gradient ---------------------------------------------------
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(args.generations):
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
            loss = -advantages[i] * seq_lp / args.generations
            loss.backward()
            total += float(loss)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        print(f"step {step:>4}  reward {sum(rewards)/len(rewards):.2f}  "
              f"loss {total:+.4f}  adv {min(advantages):+.2f}..{max(advantages):+.2f}",
              flush=True)

        if args.save_every and (step + 1) % args.save_every == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(args.output))
            print(f"  checkpoint -> {args.output}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tok.save_pretrained(str(args.output))
    print(f"\nsaved GRPO adapter to {args.output}")


if __name__ == "__main__":
    main()
