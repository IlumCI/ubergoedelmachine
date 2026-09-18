#!/usr/bin/env python3
"""Paired eval that can actually detect something, and screens the set as it goes.

WHY THIS EXISTS. The first GRPO eval came back 2 fixed, 0 broken on 40 problems -
p about 0.5, a coin flip. The reason was not the adapter. It was that **38 of the
40 problems were decided identically by both arms**: 30 both-right, 8 both-wrong.
One sample per arm on problems the base model already solves 75% of the time
leaves almost nothing that can move. An eval with 5% informative items cannot
resolve a 5% improvement, whatever the model does.

Two changes fix that, and the second is worth more than the first:

  * HARDER PROBLEMS. An item the base solves every time can only get worse and an
    item it never solves can only get better; the information is concentrated
    where its success rate is near a half. Difficulty is the cheap lever.

  * K SAMPLES PER ARM instead of one. A single sample turns a problem into a coin
    flip and throws away everything except which way it landed. K samples measure
    a RATE, and rates can move by less than one whole problem. Rollouts are
    nearly free here - the per-step cost is fixed, so 8 samples cost barely more
    than 1 (see the table in grpo_standalone.py) - which makes this close to the
    cheapest power available.

And the screen comes out as a by-product: once you know each problem's base rate,
the next run can keep the ones near a half and skip the decided ones. That is
written to the result file, so this is a one-time cost that pays forever.

Selecting on the BASE model's rate is a legitimate power boost and not a thumb on
the scale, as long as the selection happens before the adapter is loaded: both
arms are then measured fresh on the same items, so regression to the mean pulls
on both equally.

    python training/eval_paired.py --adapter adapters/reasoning-grpo \\
        --grader ./target/release/examples/grade_batch \\
        --generate ./target/release/examples/generate

The base arm is disable_adapter(): identical weights and kernels with the LoRA
switched off. A disagreement is therefore the adapter and cannot be anything
else - unlike comparing against a number from a different engine and quant.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--grader", required=True)
    p.add_argument("--generate", required=True, help="the curriculum generator binary")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--problems", type=int, default=32)
    p.add_argument(
        "--samples", type=int, default=8,
        help="rollouts per problem PER ARM. The whole point: one sample measures "
             "a coin flip, K samples measure a rate.",
    )
    p.add_argument(
        "--difficulty", default="5",
        help="harder than the training eval's d4, because an item the base "
             "always solves carries no information",
    )
    p.add_argument("--seed", default="883", help="an EVAL seed, never trained on")
    p.add_argument("--max-new", type=int, default=6144)
    p.add_argument(
        "--batch", type=int, default=32,
        help="sequences per generate call, so problems-per-call = batch // samples",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--out", type=Path, default=Path("eval-result.json"))
    p.add_argument(
        "--screen", type=Path, default=None,
        help="a previous result file; keep only the problems whose BASE rate was "
             "strictly between 0 and 1 there. This is the payoff of the last run.",
    )
    return p.parse_args()


# ------------------------------------------------------------------ stats ----
# Kept free of torch so they can be tested on any machine.


def paired_difference(base: list[int], trained: list[int], k: int) -> float:
    """Mean per-problem change in success rate, trained minus base."""
    if not base:
        return 0.0
    return sum((t - b) / k for b, t in zip(base, trained)) / len(base)


def bootstrap_ci(
    base: list[int], trained: list[int], k: int, rounds: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for the mean difference, resampling PROBLEMS not samples.

    The unit of independence is the problem: K rollouts of one problem are not K
    independent observations of the model, and treating them as such would
    narrow the interval by roughly sqrt(K) and manufacture significance.
    """
    n = len(base)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum((trained[i] - base[i]) / k for i in idx) / n)
    means.sort()
    return (means[int(0.025 * rounds)], means[int(0.975 * rounds)])


def informative(base: list[int], k: int) -> int:
    """Problems the base solves SOMETIMES - the only ones that can show a change."""
    return sum(1 for b in base if 0 < b < k)


# ------------------------------------------------------------------- main ----


def main() -> None:
    args = parse_args()
    if args.batch < args.samples:
        sys.exit(f"--batch {args.batch} must be at least --samples {args.samples}")
    per_call = args.batch // args.samples

    # Problems first, and no model loaded yet: a bad generator or grader should
    # cost seconds, not a model load.
    pool = "/tmp/eval-pool.jsonl" if os.name != "nt" else "eval-pool.jsonl"
    env = dict(os.environ, SEED=str(args.seed), DIFFICULTY=str(args.difficulty),
               COUNT=str(args.problems * 3), SPLIT_LABEL="eval", OUT=pool)
    subprocess.run([args.generate], env=env, check=True, capture_output=True)
    rows = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]

    if args.screen and args.screen.exists():
        prev = json.loads(args.screen.read_text(encoding="utf-8"))
        keep = {p["id"] for p in prev.get("problems", [])
                if 0 < p["base"] < prev.get("samples", args.samples)}
        before = len(rows)
        rows = [r for r in rows if r["id"] in keep]
        print(f"screened {before} -> {len(rows)} problems the base solved "
              f"sometimes in {args.screen}")
    rows = rows[: args.problems]
    if not rows:
        sys.exit("no problems left after screening - widen the pool or drop --screen")
    print(f"{len(rows)} problems, d{args.difficulty} seed {args.seed}, "
          f"{args.samples} samples per arm, {per_call} problems per generate call")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grpo_standalone import REASONING_SYSTEM, grade

    if not torch.cuda.is_available():
        sys.exit("no GPU visible - this needs one")
    print("GPU:", torch.cuda.get_device_name(0))

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # so completions are a contiguous suffix
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    print(f"adapter: {args.adapter}")

    def generate(chunk, with_adapter, n):
        chats = [tok.apply_chat_template(
            [{"role": "system", "content": REASONING_SYSTEM},
             {"role": "user", "content": r["question"]}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to("cuda")
        # The SAME seed for both arms, so the two differ by the adapter and not
        # by which way the sampler happened to fall.
        torch.manual_seed(20260918)

        def run():
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=True,
                    temperature=args.temperature, top_p=0.95,
                    num_return_sequences=n, pad_token_id=tok.pad_token_id,
                )
            return tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                    skip_special_tokens=True)

        if with_adapter:
            return run()
        with model.disable_adapter():
            return run()

    base_hits = [0] * len(rows)
    trained_hits = [0] * len(rows)
    t0 = time.time()
    for start in range(0, len(rows), per_call):
        chunk = rows[start:start + per_call]
        n = args.samples
        while True:
            try:
                texts = {arm: generate(chunk, arm, n) for arm in (False, True)}
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if n <= 1:
                    sys.exit("OOM even at one sample per problem")
                n //= 2
                print(f"  OOM, retrying at {n} samples per arm", file=sys.stderr)
        for arm, hits in ((False, base_hits), (True, trained_hits)):
            # batch_decode returns n consecutive completions per input, in order.
            pairs = [(t, chunk[i // n]) for i, t in enumerate(texts[arm])]
            for i, (ok, _) in enumerate(grade(args.grader, pairs)):
                hits[start + i // n] += int(ok)
        done = start + len(chunk)
        print(f"  {done}/{len(rows)} problems  [{(time.time()-t0)/60:.0f} min]  "
              f"base {sum(base_hits)}/{done*n} trained {sum(trained_hits)}/{done*n}",
              flush=True)

    # ----------------------------------------------------------- the result ---
    k, n_prob = args.samples, len(rows)
    total = k * n_prob
    diff = paired_difference(base_hits, trained_hits, k)
    lo, hi = bootstrap_ci(base_hits, trained_hits, k)
    better = sum(1 for b, t in zip(base_hits, trained_hits) if t > b)
    worse = sum(1 for b, t in zip(base_hits, trained_hits) if t < b)
    info = informative(base_hits, k)

    print(f"\n  base    {sum(base_hits)}/{total}  ({sum(base_hits)/total:.1%})")
    print(f"  trained {sum(trained_hits)}/{total}  ({sum(trained_hits)/total:.1%})")
    print(f"\n  mean per-problem change: {diff:+.1%}")
    print(f"  95% interval:            {lo:+.1%} to {hi:+.1%}")
    print(f"  problems better {better}, worse {worse}, unchanged {n_prob-better-worse}")
    print(f"\n  informative items (base solved sometimes): {info}/{n_prob}")
    if info < n_prob // 3:
        print("  *** most problems were decided either way regardless of the "
              "adapter. Raise --difficulty; this set cannot measure much.")
    if lo > 0:
        print("\n  The interval excludes zero: the adapter helped.")
    elif hi < 0:
        print("\n  The interval excludes zero on the wrong side: it hurt.")
    else:
        print("\n  The interval spans zero. No effect this eval can resolve - "
              "which is not the same as no effect.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "samples": k,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "base_total": sum(base_hits),
        "trained_total": sum(trained_hits),
        "mean_difference": diff,
        "ci95": [lo, hi],
        # Per problem, so the next run can --screen on it and skip the decided ones.
        "problems": [{"id": r["id"], "base": b, "trained": t}
                     for r, b, t in zip(rows, base_hits, trained_hits)],
    }, indent=1), encoding="utf-8")
    print(f"\nsaved to {args.out} - pass it to --screen next time")


if __name__ == "__main__":
    main()
