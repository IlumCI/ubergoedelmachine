#!/usr/bin/env python3
"""GRPO on verifiable rewards - the lever SFT distillation could not pull.

The distillation run taught the student the teacher's *style*: it reached the
same answers in 27% fewer tokens, which at a fixed budget read as +20pp accuracy.
But when the budget was lifted the base recovered 25/26 of its failures, so
imitation bought efficiency, not capability. That is what imitation buys.

GRPO rewards being RIGHT rather than sounding like the teacher, and that is the
technique behind every small model that actually gained reasoning ability. Three
things this project already has make it runnable:

  * a reward that cannot be argued with - the harness's own grader, reached
    through `grade_batch` (--grader). The trainer never decides correctness
    itself; a Python reimplementation would be a second oracle that silently
    disagrees with every eval we have run, and a reward that disagrees with the
    measurement is how a model gets optimised toward the wrong thing.
  * an unbounded, contamination-free problem source at tunable difficulty -
    samaritan-curriculum (W7). Generated problems are post-cutoff by
    construction, so the model cannot be scored on what it memorised.
  * verified headroom: the base solves these when given tokens, so the reward is
    neither all-zero (nothing to learn from) nor all-one (nothing to push on).

Usage:
    python training/grpo_reasoning.py generated-d3.jsonl \
        --base-model Qwen/Qwen3-4B-Thinking-2507 \
        --output adapters/reasoning-grpo

The dataset is the reasoning JSONL the rest of the harness reads
({question, answer, answer_kind, domain}); gold answers are used ONLY by the
grader and never shown to the model.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", type=Path, help="reasoning JSONL (question/answer/answer_kind)")
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="HF checkpoint to train (NOT a GGUF - that has no trainable weights)",
    )
    p.add_argument("--output", type=Path, default=Path("adapters/reasoning-grpo"))
    p.add_argument(
        "--grader",
        default="cargo run -q -p samaritan-run --example grade_batch",
        help="command reading grading JSONL on stdin, writing verdicts on stdout",
    )
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-completion", type=int, default=2048,
                   help="tokens per rollout; the binding cost of GRPO")
    p.add_argument("--generations", type=int, default=4,
                   help="rollouts per prompt; GRPO needs >1 to have a group to rank")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="RL wants a far smaller lr than SFT")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="verify the dataset and the reward path, then stop before the GPU")
    return p.parse_args()


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
        if not r.get("question") or not r.get("answer"):
            continue
        rows.append(r)
    if not rows:
        sys.exit(f"no usable rows in {path}")
    return rows


def grade(grader_cmd: str, items: list) -> list:
    """Score (given, gold, kind) triples through the harness's grader.

    One subprocess per batch rather than per item: the cost is process startup,
    and a GRPO step grades batch * generations completions at once.
    """
    payload = "\n".join(
        json.dumps({"given": g, "answer": a, "answer_kind": k}) for g, a, k in items
    )
    try:
        proc = subprocess.run(
            grader_cmd, shell=True, input=payload,
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("warning: grader timed out; scoring this batch 0", file=sys.stderr)
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
            "items; scoring this batch 0",
            file=sys.stderr,
        )
        return [False] * len(items)
    return verdicts


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset)
    print(f"dataset: {len(rows)} problems from {args.dataset}")

    # Prove the reward path BEFORE any GPU time: a reward function that silently
    # returns 0 would train the model to do nothing, slowly and expensively.
    kind = rows[0].get("answer_kind", "exactMatch")
    probe = [
        ("<think>x</think>\nAnswer: " + rows[0]["answer"], rows[0]["answer"], kind),
        ("<think>x</think>\nAnswer: __definitely_wrong__", rows[0]["answer"], kind),
    ]
    got = grade(args.grader, probe)
    if got != [True, False]:
        sys.exit(
            f"grader sanity check failed: expected [True, False], got {got}.\n"
            f"Command was: {args.grader}\n"
            "Fix this first - a broken reward trains nothing."
        )
    print(f"grader OK via: {args.grader}")

    if args.dry_run:
        print("--dry-run: dataset and reward path verified; stopping before model load.")
        return

    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        sys.exit(
            f"\nmissing a training dependency ({e.name}). In a CUDA venv:\n"
            "  pip install -r training/requirements.txt\n"
            "  pip install -U 'trl>=0.14'   # GRPOTrainer lives in newer TRL\n"
            "Use --dry-run to verify the dataset and reward path without a GPU."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
        fast_inference=True,   # rollouts dominate GRPO's cost
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        random_state=args.seed,
    )

    ds = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": REASONING_SYSTEM},
                {"role": "user", "content": r["question"]},
            ],
            "gold": r["answer"],
            "kind": r.get("answer_kind", "exactMatch"),
        }
        for r in rows
    ])

    def reward_correct(completions, gold, kind, **_):
        """1.0 for a verified-correct answer, 0.0 otherwise.

        The harness decides, through the same extraction and matching a graded
        episode uses. The model is never asked whether it was right, so there is
        nothing here for it to talk its way around.
        """
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        verdicts = grade(args.grader, list(zip(texts, gold, kind)))
        return [1.0 if v else 0.0 for v in verdicts]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correct],
        train_dataset=ds,
        args=GRPOConfig(
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_generations=args.generations,
            max_completion_length=args.max_completion,
            max_prompt_length=args.max_seq_len - args.max_completion,
            max_steps=args.steps,
            logging_steps=1,
            save_steps=25,          # Colab preempts; checkpoint often
            optim="adamw_8bit",
            seed=args.seed,
            output_dir=str(args.output / "checkpoints"),
            report_to="none",
        ),
    )
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\nsaved GRPO adapter to {args.output}")
    print("next: merge + export to GGUF, then measure against the SFT student and the base.")


if __name__ == "__main__":
    main()
