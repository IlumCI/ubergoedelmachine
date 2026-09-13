#!/usr/bin/env python3
"""QDoRA fine-tune of the Deviant on its own landed attacks.

This is the Deviant's channel for learning — implicit, in the weights — as
against the Warden's explicit, certificate-gated lessons. It reads the JSONL the
Rust exporter writes (`samaritan-adversary --example export_dataset`), trains a
QDoRA adapter on the attacks that *landed*, and saves it for merging back to the
GGUF the harness serves.

Why Python, and why Unsloth: QDoRA's speed lives in Triton/CUDA kernels, and
Unsloth's are among the fastest on a single consumer GPU. Python is only the
driver that launches them. Training is a one-shot offline batch over the
exporter's file — it is not in the harness's hot loop, which stays Rust.

Three honest guardrails, carried over from the exporter so they are not
rediscovered after a GPU afternoon:

1. **It refuses to train on nothing.** A run where the Warden held everything is
   all-negative, and a fine-tune on it learns to attempt nothing. This mirrors
   `DatasetSummary::has_positive_signal`: no landed rows, no training.
2. **It refuses to train on almost nothing.** One or two landed rows overfit to a
   single string. `--min-positive` is the floor; `--allow-small` overrides it
   with eyes open.
3. **It trains on the completion only.** The loss is on the attack the model
   should emit, not on the prompt it was given — otherwise it learns to
   recite the instructions back.

A QDoRA'd Deviant must stay out of the *measured* arms (Solo/Critic/Adversarial):
its improvement lives in weights the ledger cannot see, which breaks the
token-matched comparison. Train it as its own track.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="JSONL from the Rust exporter (rows: system,user,completion,verdict,reward,round)")
    p.add_argument(
        "--base-model",
        default="huihui-ai/Huihui-Ministral-3-8B-Reasoning-2512-abliterated",
        help="HF checkpoint to fine-tune (NOT the served GGUF — Unsloth trains from safetensors)",
    )
    p.add_argument("--output", type=Path, default=Path("adapters/deviant-qdora"), help="where to save the QDoRA adapter")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=66600)
    p.add_argument("--min-positive", type=int, default=16, help="refuse to train with fewer landed rows than this")
    p.add_argument("--allow-small", action="store_true", help="train anyway below --min-positive (will overfit)")
    p.add_argument(
        "--reward-weighted",
        action="store_true",
        help="oversample each landed row in proportion to its reward, instead of plain SFT on distinct landed rows",
    )
    p.add_argument("--dry-run", action="store_true", help="inspect and gate the dataset, then stop before loading the model")
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
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"malformed JSON on line {i} of {path}: {e}")
    return rows


def select_training_rows(rows: list[dict], reward_weighted: bool) -> list[dict]:
    """The landed rows, shaped for the chosen objective.

    Plain SFT trains on the distinct landed attacks — one lesson per real hole.
    Reward-weighted oversamples each by round(reward * 4), so a fully-novel
    landing (reward ~1.0) is seen more often than a marginal one, an honest
    stand-in for a true reward objective without pretending SFT is one.
    """
    landed = [r for r in rows if float(r.get("reward", 0.0)) > 0.0]
    if not reward_weighted:
        # Distinct (user, completion) pairs, so a trick that recurred across
        # rounds does not dominate the set by sheer repetition.
        seen = set()
        distinct = []
        for r in landed:
            key = (r.get("user", ""), r.get("completion", ""))
            if key not in seen:
                seen.add(key)
                distinct.append(r)
        return distinct
    weighted = []
    for r in landed:
        n = max(1, round(float(r["reward"]) * 4))
        weighted.extend([r] * n)
    return weighted


def report(rows: list[dict], training: list[dict]) -> None:
    verdicts = Counter(r.get("verdict", "?") for r in rows)
    print(f"dataset: {len(rows)} rows")
    for v in ("landed", "repelled", "inert"):
        print(f"  {v}: {verdicts.get(v, 0)}")
    print(f"training rows selected: {len(training)}")


def gate(training: list[dict], min_positive: int, allow_small: bool) -> None:
    if len(training) == 0:
        sys.exit(
            "\nNo attempt landed. This set has no positive signal — a fine-tune on it would\n"
            "learn to attempt nothing. Run more bouts, or point the Deviant at a weakened\n"
            "target (train_target example), before training."
        )
    if len(training) < min_positive and not allow_small:
        sys.exit(
            f"\nOnly {len(training)} landed row(s) — below --min-positive={min_positive}. A fine-tune\n"
            "this small overfits to a handful of strings rather than learning the technique.\n"
            "Gather more landings, or pass --allow-small to proceed with eyes open."
        )


def response_only_markers(tokenizer) -> tuple[str, str] | None:
    """The (instruction, response) header strings for this base's chat template.

    `train_on_responses_only` masks the loss to the assistant turn by splitting the
    *rendered* prompt on the exact substrings that separate the user turn from the
    assistant turn — and those are template-specific. Hardcoding one vendor's pair
    is the bug this replaces: the Deviant's Ministral uses Mistral `[INST]`/`[/INST]`,
    but a Qwen (ChatML) student uses `<|im_start|>` headers, and feeding the wrong
    pair means the markers are never found, the mask covers the prompt too, and the
    model learns to recite questions instead of answering them.

    Detected from a rendered probe rather than the model name, so it follows the
    template the tokenizer actually applies. Returns None for an unknown template,
    which the caller turns into a loud warning rather than a silent full-text train.
    """
    try:
        probe = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "PROBE_USER"},
                {"role": "assistant", "content": "PROBE_ASSISTANT"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:  # noqa: BLE001 — a template that won't render is an unknown one
        return None
    # ChatML — Qwen (the reasoning student) and many others.
    if "<|im_start|>assistant" in probe:
        return ("<|im_start|>user\n", "<|im_start|>assistant\n")
    # Mistral / Ministral — the Deviant's base.
    if "[/INST]" in probe:
        return ("[INST]", "[/INST]")
    # Llama-3 family.
    if "<|start_header_id|>" in probe:
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    return None


def format_and_train(args: argparse.Namespace, training: list[dict]) -> None:
    # Imported here so --dry-run and the gate work without a GPU stack installed.
    try:
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        sys.exit(
            f"\nmissing a training dependency ({e.name}). Install into a CUDA venv:\n"
            "  pip install -r training/requirements.txt\n"
            "then rerun. Use --dry-run to inspect the dataset without the GPU stack."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,  # the Q in QDoRA
        dtype=None,
    )
    # use_dora=True over the 4-bit base is exactly QDoRA.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        use_dora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def to_text(row: dict) -> dict:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = Dataset.from_list([to_text(r) for r in training])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            logging_steps=1,
            optim="adamw_8bit",
            seed=args.seed,
            output_dir=str(args.output / "checkpoints"),
            report_to="none",
        ),
    )

    # Train on the assistant completion only — the answer, not the prompt it was
    # given — or the model learns to recite the prompt back (guardrail #3). The
    # marker strings are template-specific, so derive them from this base's own chat
    # template: Qwen (ChatML) and the Deviant's Ministral (Mistral) need different
    # pairs, and the wrong pair silently masks nothing.
    markers = response_only_markers(tokenizer)
    if markers is None:
        print(
            "warning: unrecognised chat template — cannot restrict loss to the response; "
            "training on the FULL text (the model may learn to echo prompts). Add this "
            "template's markers to response_only_markers() before trusting the result.",
            file=sys.stderr,
        )
    else:
        instruction_part, response_part = markers
        print(f"masking loss to responses: instruction={instruction_part!r} response={response_part!r}")
        try:
            trainer = train_on_responses_only(
                trainer,
                instruction_part=instruction_part,
                response_part=response_part,
            )
        except Exception as e:  # noqa: BLE001 — a template mismatch should warn, not crash
            print(f"warning: could not restrict loss to responses ({e}); training on the full text", file=sys.stderr)

    if torch.cuda.is_available():
        print(f"training on {torch.cuda.get_device_name(0)}")
    else:
        print("warning: no CUDA device visible; this will be extremely slow", file=sys.stderr)

    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\nsaved QDoRA adapter to {args.output}")
    print("next: merge and export to GGUF for llama.cpp — see training/README.md")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset)
    training = select_training_rows(rows, args.reward_weighted)
    report(rows, training)
    gate(training, args.min_positive, args.allow_small)
    if args.dry_run:
        print("\n--dry-run: dataset passes the gate; stopping before model load.")
        return
    format_and_train(args, training)


if __name__ == "__main__":
    main()
