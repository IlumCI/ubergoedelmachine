#!/usr/bin/env python3
"""Why is generation slow? Measure it, do not guess at it.

A run on an A100 spent 850 seconds producing 65,536 tokens across 8 streams -
68 tok/s aggregate, 104 ms per decode step. A 4B model in bfloat16 reads about
8 GB of weights per step, and an A100 moves 1.5 TB/s, so the weight read alone
should take 5 ms. Even allowing for transformers' per-step Python overhead, that
is five to ten times slower than it should be, and generation is essentially the
entire wall clock of a GRPO run.

Guessing at that is how an afternoon disappears. This times the same generation
under the configurations that plausibly explain it, one variable at a time:

  device_map=      what the trainer does today. `device_map` routes the load
                   through accelerate, which attaches a hook to every submodule
                   to check and move tensors. On a model that fits on one GPU
                   those hooks buy nothing and run hundreds of times per forward.
  .to(cuda)        the same weights, placed directly, no hooks.
  attn sdpa        scaled_dot_product_attention rather than the eager path.
  + LoRA           what the adapter costs: r=16 adds two small matmuls to each
                   of 7 modules across 36 layers, which is ~500 extra kernel
                   launches per forward, each with its own Python overhead.
  + checkpointing  gradient checkpointing is enabled for the backward pass and
                   is supposed to be inert under eval(). "Supposed to" is worth
                   fifteen seconds to check.

It also reports the first quarter of the generation against the last, because
the two explanations look completely different there: fixed overhead per step is
flat, while a cache that is not being reused grows with the context.

    python training/bench_generate.py                    # ~5 minutes
    python training/bench_generate.py --new-tokens 128   # quicker
"""
from __future__ import annotations

import argparse
import gc
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--batch", type=int, default=8, help="matches --generations")
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible - this measures GPU generation")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.0f} GB, "
          f"bfloat16 {'native' if torch.cuda.is_bf16_supported() else 'EMULATED'}")
    print(f"torch {torch.__version__}")
    import transformers
    print(f"transformers {transformers.__version__}")
    print(f"\nbatch {args.batch}, {args.new_tokens} new tokens, "
          f"{args.prompt_len}-token prompt\n")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # A real prompt, not zeros: token ids that never occur can take different
    # paths through the embedding and give a flattering number.
    text = ("Solve this step by step, showing every intermediate value. " * 40)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.prompt_len)
    enc = {k: v.cuda() for k, v in enc.items()}

    def timed(model, label):
        """tok/s overall, and first quarter against last quarter."""
        model.eval()
        with torch.no_grad():   # warm up the kernels; the first call is not the cost
            model.generate(**enc, max_new_tokens=8, do_sample=False,
                           num_return_sequences=args.batch,
                           pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()

        marks = []
        quarter = max(args.new_tokens // 4, 1)

        class Mark(torch.nn.Module):
            """A stopping criterion that never stops, used as a per-step clock."""
            def __call__(self, input_ids, scores, **kw):
                marks.append(time.perf_counter())
                return torch.zeros(input_ids.shape[0], dtype=torch.bool,
                                   device=input_ids.device)

        from transformers import StoppingCriteriaList
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.new_tokens, do_sample=False,
                num_return_sequences=args.batch, pad_token_id=tok.pad_token_id,
                stopping_criteria=StoppingCriteriaList([Mark()]),
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        made = (out.shape[1] - enc["input_ids"].shape[1]) * args.batch
        rate = made / dt
        head = tail = float("nan")
        if len(marks) > 2 * quarter:
            head = quarter / (marks[quarter - 1] - t0) * args.batch
            tail = quarter / (marks[-1] - marks[-1 - quarter]) * args.batch
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {label:<34} {rate:>7.0f} tok/s   "
              f"first quarter {head:>6.0f}  last {tail:>6.0f}   "
              f"{dt:>5.1f}s   {peak:.0f} GB")
        return rate

    def load(**kw):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, **kw
        )

    def drop(m):
        del m
        gc.collect()
        torch.cuda.empty_cache()

    results = {}

    m = load(device_map="cuda")
    results["device_map"] = timed(m, "device_map='cuda'  (trainer today)")
    drop(m)

    m = load().to("cuda")
    results["to_cuda"] = timed(m, ".to('cuda')        (no accelerate hooks)")
    drop(m)

    m = load(attn_implementation="sdpa").to("cuda")
    results["sdpa"] = timed(m, ".to('cuda') + attn_implementation='sdpa'")

    from peft import LoraConfig, get_peft_model
    m = get_peft_model(m, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    results["lora"] = timed(m, "  + LoRA r=16        (what the adapter costs)")

    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    m.config.use_cache = True
    results["ckpt"] = timed(m, "  + gradient checkpointing  (should be inert)")
    drop(m)

    # ------------------------------------------------------------- verdict ---
    print()
    best = max(results, key=results.get)
    now = results["device_map"]
    print(f"fastest: {best} at {results[best]:.0f} tok/s, "
          f"against {now:.0f} for what the trainer does now "
          f"({results[best] / max(now, 1):.1f}x)")

    if results["to_cuda"] > now * 1.3:
        print("\n-> device_map is the problem. It routes the load through accelerate,")
        print("   which hooks every submodule; on a model that fits on one GPU those")
        print("   hooks run hundreds of times per forward and buy nothing.")
    if results["lora"] < results["sdpa"] * 0.7:
        print("\n-> LoRA is costing more than a third of generation. Merging the adapter")
        print("   into the base weights for the rollout phase and unmerging to train")
        print("   would remove it - peft can do that with merge_adapter/unmerge_adapter.")
    if results["ckpt"] < results["lora"] * 0.9:
        print("\n-> gradient checkpointing is NOT inert under eval() here, which it is")
        print("   supposed to be. Disable it around generation and re-enable to train.")
    for k, v in results.items():
        if v < 200:
            print(f"\n-> {k} is under 200 tok/s, which no configuration here explains.")
            print("   Compare the first quarter against the last: if the last is much")
            print("   slower the KV cache is not being reused; if they match, the cost")
            print("   is fixed per-step overhead and the answer is a real inference")
            print("   engine (vLLM) rather than transformers' generate loop.")
            break


if __name__ == "__main__":
    main()
