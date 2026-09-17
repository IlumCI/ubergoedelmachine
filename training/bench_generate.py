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

    # The per-step clock is a nice-to-have, and its API is the most likely thing
    # here to move between transformers versions. The tok/s number is the answer;
    # losing the head-against-tail split must not lose that too.
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
        HAVE_CLOCK = True
    except ImportError as e:
        print(f"note: no per-step clock ({e}); reporting overall rate only\n")
        HAVE_CLOCK = False

    # Sampling, not greedy: greedy with num_return_sequences > 1 is rejected
    # outright, and rightly - eight identical sequences would measure nothing the
    # trainer does. These match the trainer's rollout settings exactly, because
    # the point is to time THAT workload and not a tidier one.
    GEN = dict(do_sample=True, temperature=1.0, top_p=0.95)

    def timed(model, label):
        """tok/s overall, and first quarter against last quarter."""
        model.eval()
        with torch.no_grad():   # warm up the kernels; the first call is not the cost
            model.generate(**enc, max_new_tokens=8, **GEN,
                           num_return_sequences=args.batch,
                           pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()

        marks: list[float] = []
        quarter = max(args.new_tokens // 4, 1)
        extra = {}
        if HAVE_CLOCK:
            class Mark(StoppingCriteria):
                """A stopping criterion that never stops - a per-step clock."""
                def __call__(self, input_ids, scores, **kw):
                    marks.append(time.perf_counter())
                    return torch.zeros(input_ids.shape[0], dtype=torch.bool,
                                       device=input_ids.device)
            extra["stopping_criteria"] = StoppingCriteriaList([Mark()])

        # Same seed for every configuration, so they generate the same text and
        # the comparison is of speed rather than of luck.
        torch.manual_seed(1234)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.new_tokens, **GEN,
                num_return_sequences=args.batch, pad_token_id=tok.pad_token_id,
                **extra,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # Count real tokens, not the padded rectangle: a sequence that hit EOS
        # early would otherwise be credited with tokens it never generated.
        comp = out[:, enc["input_ids"].shape[1]:]
        made = int((comp != tok.pad_token_id).sum())
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

    results: dict[str, float] = {}

    def measure(key, model, label):
        """One configuration, and a failure in it must not cost the rest.

        The first version of this file died on the very first config and the
        whole five minutes bought nothing. Every row is independent, so a row
        that cannot run should print why and let the others answer the question.
        """
        try:
            results[key] = timed(model, label)
        except Exception as e:                      # noqa: BLE001 - report anything
            print(f"  {label:<34} FAILED: {type(e).__name__}: {e}")

    def attempt(label, build):
        """Build a model for one row, or report why the row is missing.

        The loads are guarded as well as the timings: `attn_implementation` and
        `device_map` are exactly the arguments most likely to have moved between
        transformers versions, and this runtime is on transformers 5.x - so the
        argument that is being tested is also the argument that could kill the
        script before any of the other rows run.
        """
        try:
            return build()
        except Exception as e:                      # noqa: BLE001 - report anything
            print(f"  {label:<34} could not load: {type(e).__name__}: {e}")
            return None

    m = attempt("device_map='cuda'", lambda: load(device_map="cuda"))
    if m is not None:
        measure("device_map", m, "device_map='cuda'  (trainer before)")
        drop(m)

    m = attempt(".to('cuda')", lambda: load().to("cuda"))
    if m is not None:
        measure("to_cuda", m, ".to('cuda')        (no accelerate hooks)")
        drop(m)

    m = attempt(".to('cuda') + sdpa",
                lambda: load(attn_implementation="sdpa").to("cuda"))
    if m is None:
        # Fall back so the LoRA and checkpointing rows still have a model. They
        # answer different questions and should not be lost with this one.
        m = attempt(".to('cuda') (sdpa unavailable)", lambda: load().to("cuda"))
    else:
        measure("sdpa", m, ".to('cuda') + sdpa  (trainer now)")

    if m is not None:
        from peft import LoraConfig, get_peft_model
        m = get_peft_model(m, LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        measure("lora", m, "  + LoRA r=16        (what the adapter costs)")

        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        m.config.use_cache = True
        measure("ckpt", m, "  + gradient checkpointing  (should be inert)")
        drop(m)

    if not results:
        raise SystemExit("every configuration failed - nothing measured")

    # ------------------------------------------------------------- verdict ---
    # Every comparison is guarded: a row that failed must not take the verdict
    # down with it, and a conclusion drawn from a missing number is worse than
    # no conclusion.
    def got(*keys):
        return all(k in results for k in keys)

    print()
    best = max(results, key=results.get)
    print(f"fastest: {best} at {results[best]:.0f} tok/s")
    if got("device_map"):
        now = results["device_map"]
        print(f"  against {now:.0f} for the old load path "
              f"({results[best] / max(now, 1e-9):.1f}x)")

    if got("device_map", "to_cuda"):
        if results["to_cuda"] > results["device_map"] * 1.3:
            print("\n-> device_map WAS the problem. It routes the load through accelerate,")
            print("   which hooks every submodule; on a model that fits on one GPU those")
            print("   hooks run hundreds of times per forward and buy nothing. Already")
            print("   fixed in the trainer.")
        else:
            print("\n-> device_map is NOT the problem; the two load paths are within noise.")
    if got("lora", "sdpa") and results["lora"] < results["sdpa"] * 0.7:
        print("\n-> LoRA is costing more than a third of generation. merge_adapter()")
        print("   before the rollouts and unmerge_adapter() before the backward would")
        print("   remove it - at the cost of a bf16 round-trip each step, which is why")
        print("   it is not already done.")
    if got("ckpt", "lora") and results["ckpt"] < results["lora"] * 0.9:
        print("\n-> gradient checkpointing is NOT inert under eval() here, which it is")
        print("   supposed to be. Disable it around generation and re-enable to train.")

    if max(results.values()) < 200:
        print("\n-> nothing here breaks 200 tok/s, so no configuration explains the")
        print("   slowness. Read the first-quarter against last-quarter columns: if the")
        print("   last is much slower the KV cache is not being reused; if they match,")
        print("   the cost is fixed per-step overhead, transformers' generate loop is")
        print("   the ceiling, and the answer is a real inference engine (vLLM).")
    elif got("sdpa"):
        gain = results["sdpa"] / 68.0     # the measured rate of the run that died
        print(f"\n-> the run that died managed 68 tok/s. This config is {gain:.1f}x that,")
        print(f"   which would turn 11 minutes a prompt into roughly "
              f"{11 / max(gain, 1e-9):.1f}.")


if __name__ == "__main__":
    main()
