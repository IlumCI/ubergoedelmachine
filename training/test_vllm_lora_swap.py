#!/usr/bin/env python3
"""Can vLLM swap a LoRA adapter between steps? The question the design hangs on.

GRPO changes the policy every step. If the rollout engine can load a new adapter
without being rebuilt, rollouts move to vLLM (10.2x measured) and the backward
pass stays in transformers. If it cannot, the engine has to be torn down and
rebuilt each step, and a ~90 s rebuild against a ~70 s step is not a speedup.

THE TRAP THIS EXISTS TO FIND. vLLM caches adapters by INTEGER ID, not by path or
by content. Re-requesting the same id after the file on disk has changed is not
an error - it quietly serves the weights it cached the first time. In a training
loop that reads as a healthy run whose policy silently never updates, and the
reward curve would look no different. So the test is not "does it load" but "does
requesting a CHANGED adapter actually change the output".

HOW IT TELLS. A LoRA adds `B·A·x`, and `B = 0` makes the adapter an exact
identity - the model behaves as if no adapter were attached at all. So zeroing
every lora_B gives a second adapter with the SAME shapes and a KNOWN effect, and
no model has to be loaded to make it:

    real    = the trained adapter          -> must differ from base
    zeroed  = same file, every lora_B = 0  -> must equal base, exactly

Greedy decoding, so "equal" means byte-equal text rather than a judgement call.

    python training/test_vllm_lora_swap.py --adapter adapters/reasoning-grpo
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--adapter", type=Path, default=None,
        help="a trained adapter. Omit to build a synthetic one instead - the "
             "swap mechanism does not care whether the weights learned anything, "
             "only that they are not zero, and a synthetic adapter needs no "
             "Drive and no prior run.",
    )
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--work", type=Path, default=Path("/content/lora-swap"))
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--gpu-frac", type=float, default=0.55)
    return p.parse_args()


def zero_lora_b(src: Path, dst: Path) -> int:
    """Copy an adapter with every `lora_B` zeroed, making it an exact identity."""
    import torch
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copy(f, dst / f.name)
    t = load_file(str(src / "adapter_model.safetensors"))
    n = 0
    for k in t:
        if "lora_B" in k:
            t[k] = torch.zeros_like(t[k])
            n += 1
    save_file(t, str(dst / "adapter_model.safetensors"))
    return n


def synthesise(base_model: str, dst: Path) -> Path:
    """Build a LoRA with NON-ZERO B, from the config alone - no model load.

    peft initialises B to zero, which is right for training and useless here: a
    zero-B adapter is the identity, so it could not tell "the swap worked" from
    "the adapter was ignored". Filling B with noise makes the effect loud.

    The shapes come from `config.json`, not from an instantiated model. Loading
    8 GB of weights to find out that `q_proj` is 2560 wide is minutes of wall
    clock and gigabytes of RAM for numbers that are sitting in a 700-byte file -
    and the weights are never read, since every tensor written here is random.
    """
    import torch
    from peft import LoraConfig
    from safetensors.torch import save_file
    from transformers import AutoConfig

    print(f"synthesising an adapter from {base_model}'s config", flush=True)
    c = AutoConfig.from_pretrained(base_model)
    h = c.hidden_size
    head = getattr(c, "head_dim", None) or h // c.num_attention_heads
    q_out = c.num_attention_heads * head
    kv_out = getattr(c, "num_key_value_heads", c.num_attention_heads) * head
    ff = c.intermediate_size
    # (module, in_features, out_features) for each projection LoRA targets.
    shapes = {
        "self_attn.q_proj": (h, q_out), "self_attn.k_proj": (h, kv_out),
        "self_attn.v_proj": (h, kv_out), "self_attn.o_proj": (q_out, h),
        "mlp.gate_proj": (h, ff), "mlp.up_proj": (h, ff), "mlp.down_proj": (ff, h),
    }
    assert h > 0 and ff > 0, f"nonsense shapes from config: hidden={h} ff={ff}"

    r = 16
    g = torch.Generator().manual_seed(0)
    t = {}
    for i in range(c.num_hidden_layers):
        for mod, (fan_in, fan_out) in shapes.items():
            p = f"base_model.model.model.layers.{i}.{mod}"
            t[f"{p}.lora_A.weight"] = (
                torch.randn(r, fan_in, generator=g) * 0.02).to(torch.bfloat16)
            # B is the one that must NOT be zero, or the adapter is the identity
            # and the test cannot tell a working swap from an ignored adapter.
            t[f"{p}.lora_B.weight"] = (
                torch.randn(fan_out, r, generator=g) * 0.02).to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    LoraConfig(r=r, lora_alpha=r, lora_dropout=0.0, bias="none",
               task_type="CAUSAL_LM", base_model_name_or_path=base_model,
               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]
               ).save_pretrained(str(dst))
    save_file(t, str(dst / "adapter_model.safetensors"))
    print(f"  {len(t)} tensors over {c.num_hidden_layers} layers", flush=True)
    return dst


def main() -> None:
    args = parse_args()
    shutil.rmtree(args.work, ignore_errors=True)
    src = args.adapter
    if src is None:
        src = synthesise(args.base_model, args.work / "synth")
    if not (src / "adapter_model.safetensors").exists():
        sys.exit(f"no adapter_model.safetensors in {src}")

    real, zeroed = args.work / "real", args.work / "zeroed"
    real.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, real / f.name)
    n = zero_lora_b(src, zeroed)
    print(f"built two adapters: real, and one with {n} lora_B tensors zeroed")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Greedy: "the same" then means byte-identical text, not a judgement call.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompt = "Compute 17 * 23 step by step, showing every intermediate value."

    llm = LLM(model=args.base_model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_frac, max_model_len=2048,
              enable_lora=True, max_lora_rank=32)

    def gen(lora=None):
        out = llm.generate([prompt], sp, **({"lora_request": lora} if lora else {}))
        return out[0].outputs[0].text

    base = gen()
    t_real = gen(LoRARequest("real", 1, str(real)))
    t_zero = gen(LoRARequest("zero", 2, str(zeroed)))

    ok = True
    print("\n--- is the adapter applied at all? ---")
    if t_zero == base:
        print("  zeroed adapter == base            PASS (B=0 is the identity)")
    else:
        ok = False
        print("  zeroed adapter != base            FAIL - an identity adapter "
              "changed the output, so something other than the LoRA differs")
    if t_real != base:
        print("  trained adapter != base           PASS (it is being applied)")
    else:
        ok = False
        print("  trained adapter == base           FAIL - either vLLM ignored the "
              "adapter, or 49 steps changed nothing this prompt can show")

    # ---- THE hot-swap question ---------------------------------------------
    # Overwrite the ZEROED adapter's files with the REAL ones. Same path, new
    # content. Then ask for it twice: once under the id it was already cached
    # with, once under a fresh id.
    print("\n--- hot-swap: same path, new content ---")
    for f in real.iterdir():
        shutil.copy(f, zeroed / f.name)

    t_old_id = gen(LoRARequest("zero", 2, str(zeroed)))     # id 2 again
    t_new_id = gen(LoRARequest("zero", 3, str(zeroed)))     # fresh id

    if t_new_id == t_real:
        print("  fresh id picks up new content     PASS - increment the id each step")
    else:
        ok = False
        print("  fresh id did NOT pick it up       FAIL - the engine must be "
              "rebuilt per step, and the 10x has to beat a ~90 s rebuild")
    if t_old_id == t_zero:
        print("  reused id serves STALE weights    (expected - never reuse an id)")
    elif t_old_id == t_real:
        print("  reused id also picked it up       (fine, but do not rely on it)")

    print("\nVERDICT:", "vLLM can back the rollouts, swapping by a fresh id each step"
          if ok else "something is wrong above - do not wire this in yet")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
