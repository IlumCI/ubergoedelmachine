#!/usr/bin/env python3
"""Is vLLM worth it, and does it survive contact with this runtime? Measure first.

WHY. Generation is the entire wall clock of a GRPO run, and it is bound by fixed
per-step overhead rather than by the GPU: measured on an A100-80GB, a decode step
costs 106 ms at a batch of 8 and 115 ms at 64, so throughput scales with batch
and the card is nearly idle either way. Raising --generations to 32 already
claimed the easy 4x. What is left is the ~110 ms itself, which is transformers'
Python generate loop, and no amount of batching touches it. vLLM's CUDA graphs
and fused kernels are aimed at exactly that number.

WHY A PROBE AND NOT AN INTEGRATION. TRL failed four separate times in this
runtime and never once on the algorithm - an unconditional `mergekit` import, a
lazy loader that hid it, mergekit's own transitive chain, and torchao wheels
built for the wrong cpython. vLLM pins torch hard. Installing it can quietly
replace the torch that the trainer is built against, and then nothing works and
the reason is three layers down. So this answers three questions and writes no
integration code:

  1. does it install without moving torch or transformers underneath us?
  2. how fast is it really, on THIS model, at the batch GRPO actually uses?
  3. can a LoRA adapter be hot-swapped between steps? GRPO changes the policy
     every step, so an engine that cannot reload the adapter must be rebuilt
     each time, and a 30 s rebuild against a 60 s step is not a speedup.

Question 3 is the one that decides the design. If hot-swap works, rollouts move
to vLLM and the backward stays in HF. If it does not, the adapter has to be
merged and the engine restarted per step, and the win has to beat that overhead.

    python training/bench_vllm.py                 # measure only, assumes installed
    python training/bench_vllm.py --install       # pip install vllm first

HF BASELINE, for comparison: 289 tok/s aggregate at batch 32 (110.7 ms/step),
558 at batch 64. Anything under ~1000 here is not worth the dependency.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HF_BASELINE = {8: 75.0, 16: 149.0, 32: 289.0, 64: 558.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--adapter", type=Path, default=None,
                   help="a real LoRA adapter, to test loading and hot-swap")
    p.add_argument("--batch", type=int, default=32, help="matches --generations")
    p.add_argument("--new-tokens", type=int, default=512)
    p.add_argument("--install", action="store_true")
    p.add_argument(
        "--gpu-frac", type=float, default=0.55,
        help="vLLM preallocates its KV cache up front and would take the whole "
             "card. Training needs room on the same GPU for weights, Adam and "
             "activations, so it does not get the whole card.",
    )
    return p.parse_args()


def versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "peft", "vllm"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:                       # noqa: BLE001
            out[mod] = f"absent ({type(e).__name__})"
    return out


def main() -> None:
    args = parse_args()

    before = versions()
    print("before:", json.dumps(before))

    if args.install:
        # Recorded, not trusted. The thing that matters is whether torch moved.
        print("\ninstalling vllm - this is the step that can break everything\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"],
                       check=False)
        after = versions()
        print("after: ", json.dumps(after))
        moved = [k for k in ("torch", "transformers")
                 if before.get(k) != after.get(k) and "absent" not in before.get(k, "")]
        if moved:
            print(f"\n*** {', '.join(moved)} CHANGED VERSION. The trainer is built "
                  f"against the old one. Re-run the GRPO maths tests and the smoke "
                  f"test before trusting any training in this runtime.")
        else:
            print("\ntorch and transformers unchanged - the install was survivable")

    try:
        from vllm import LLM, SamplingParams
    except Exception as e:                           # noqa: BLE001
        raise SystemExit(f"vllm will not import: {type(e).__name__}: {e}\n"
                         "Re-run with --install, or give up on it cheaply.")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible")
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # A prompt shaped like the real ones, not a toy: the per-step cost depends on
    # the KV cache, and the KV cache depends on how much context there is.
    prompt = ("Solve this step by step, showing every intermediate value. " * 40)
    sp = SamplingParams(temperature=1.0, top_p=0.95,
                        max_tokens=args.new_tokens, n=args.batch)

    results = {}

    def timed(llm, label, lora=None):
        llm.generate([prompt], sp, **({"lora_request": lora} if lora else {}))  # warm
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, **({"lora_request": lora} if lora else {}))
        dt = time.perf_counter() - t0
        made = sum(len(c.token_ids) for c in out[0].outputs)
        rate = made / dt
        print(f"  {label:<34} {rate:>8.0f} tok/s   {dt:>5.1f}s   {made:,} tokens")
        results[label] = rate
        return out

    print(f"\nbatch {args.batch}, {args.new_tokens} new tokens\n")
    kw = dict(model=args.base_model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_frac, max_model_len=4096)
    if args.adapter:
        kw.update(enable_lora=True, max_lora_rank=32)

    llm = LLM(**kw)
    timed(llm, "vLLM, base weights")

    if args.adapter:
        from vllm.lora.request import LoRARequest
        a = timed(llm, "vLLM + LoRA adapter",
                  LoRARequest("grpo", 1, str(args.adapter)))

        # THE question for GRPO. The policy changes every step, so the adapter
        # must be replaceable without rebuilding the engine. vLLM caches by
        # integer id, so a changed adapter at the same path needs a NEW id -
        # reusing the id would silently serve the old weights, which is the kind
        # of bug that looks like "training does nothing".
        b = timed(llm, "  same adapter, new id (hot-swap)",
                  LoRARequest("grpo", 2, str(args.adapter)))
        same = a[0].outputs[0].text == b[0].outputs[0].text
        print(f"\n  identical text across ids: {same} "
              f"({'expected - same weights, same seed' if same else 'investigate'})")

    # ------------------------------------------------------------- verdict ---
    print()
    base = HF_BASELINE.get(args.batch)
    best = max(results.values()) if results else 0.0
    if base:
        print(f"HF transformers at batch {args.batch}: {base:.0f} tok/s")
        print(f"vLLM best:                    {best:>8.0f} tok/s   "
              f"({best / base:.1f}x)")
        step_min = 6144 / (best / args.batch) / 60
        print(f"\nA {args.batch}-rollout group to a 6144-token cap would take "
              f"~{step_min:.1f} min (it is ~11 now).")
        if best < base * 2:
            print("\n-> under 2x. Not worth the dependency risk; stay on transformers.")
        elif best < base * 5:
            print("\n-> worth having, but the integration has to be cheap to justify it.")
        else:
            print("\n-> this changes the economics of the whole run. Integrate it.")


if __name__ == "__main__":
    main()
