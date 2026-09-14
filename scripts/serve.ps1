<#
.SYNOPSIS
  Serve Samaritan's local model — the reasoning solver by default, or the
  adversary on demand.

.DESCRIPTION
  Two roles want two different models, and a 4 GB GPU cannot hold both at once,
  so this serves one at a time under a single alias (`samaritan-playout`) — the
  name the Rust config and the examples ask for, so nothing there has to change
  when you switch roles.

      -Role solver   (default)  Qwen3-4B-Thinking-2507 Q8_0, -ngl 30
      -Role deviant             Huihui-Ministral-3-8B abliterated Q6_K, -ngl 18

  Why the solver is a 4B thinking model, not the 8B: benched on this box
  (see docs/inference.md), the 8B generates at 7.9 t/s — the whole ~2-min budget
  buys one 900-token answer with no room to *think*. The Qwen 4B at -ngl 30 does
  18.9 t/s (2.4x) with 1003 t/s prefill, so a full ~2 000-token reasoning trace
  fits the budget. Reasoning needs a model fast enough to reason out loud.

  Why the adversary stays the abliterated Ministral: its job is to attempt what
  a refusal-trained model would decline, so refusal ablation is the point there
  and a liability in a solver.

  Tuned for this machine specifically:

      i7-12650H   6 P-cores + 4 E-cores, 16 threads
      16 GB       DDR4-3200, dual channel (~38 GB/s real)
      RTX 3050    Laptop, 4 GB VRAM (~3.5 GB usable), ~192 GB/s

  Both models are a split load: as many layers as fit go to the GPU, the rest
  run on CPU out of system RAM.

  Why each setting is what it is:

  -ngl        Partial offload. GPU memory is ~5x faster per byte than system
              RAM here, so every layer moved across is a real win — but only
              until VRAM runs out, after which llama.cpp spills and it gets
              *worse*, not better. For the Qwen 4B the measured cliff is sharp:
              generation climbs to 18.9 t/s at -ngl 30, then prefill collapses
              1003 -> 215 t/s at 31. Do not exceed the role's default without
              re-running bench-model.ps1.

  -t 6        Threads for generation: P-cores only. Alder Lake is
              heterogeneous and llama.cpp splits work evenly across threads,
              so a batch finishes when the slowest thread does. Adding the
              four E-cores makes every batch wait on them.

  -tb 10      Threads for prompt processing. That phase is compute-bound
              rather than latency-bound, so the E-cores do help there.

  --parallel 2 -cb
              Generation is memory-bandwidth-bound: the weights are read once
              per token whether it belongs to one sequence or several, so
              batching amortises the read and aggregate tokens/sec rises.
              Dialled to 2 after a thermal scare; raise once cooling is proven.

  --cache-reuse 256
              Keeps the KV state of a matching prompt prefix. Samaritan lays
              its prompts out stable-part-first precisely so this hits; see
              samaritan-agent/src/prompt.rs.

  -fa on / -ctk q8_0 / -ctv q8_0
              Flash attention plus an 8-bit KV cache. Both cut memory traffic,
              which is the binding constraint.

.PARAMETER Role
  solver (default) or deviant. Picks the model and its measured -ngl.

.PARAMETER Ngl
  Override the role's default GPU-layer split. 0 (default) uses the measured
  best for the role.

.PARAMETER ModelFile
  Serve a specific GGUF from -ModelDir instead of the role's default. This is
  what makes an A/B honest: a trained student and the base it was trained from
  must be served through the SAME alias, settings, and -ngl, so the only thing
  that differs between two eval runs is the weights.

      scripts\serve.ps1 -ModelFile samaritan-student-v1-185trace-q4_k_m.gguf
#>
param(
    [ValidateSet("solver", "deviant")][string]$Role = "solver",
    [string]$ModelFile = "",
    [int]$Ngl = 0,
    [int]$Port = 8080,
    [string]$ModelDir = "$env:USERPROFILE\models",
    [int]$Parallel = 2,
    [int]$CtxPerSlot = 4096
)

$ErrorActionPreference = "Stop"

# Model and measured operating point per role. One at a time — the 4 GB GPU
# cannot hold both.
if ($Role -eq "solver") {
    $modelFile = "Qwen3-4B-Thinking-2507-Qwen3.8-Max-Distillation-Detrax-q8_0.gguf"
    $defaultNgl = 30
    $label = "Qwen3-4B-Thinking Q8_0 (solver)"
} else {
    $modelFile = "Huihui-Ministral-3-8B-Reasoning-2512-abliterated.Q6_K.gguf"
    $defaultNgl = 18
    $label = "Ministral-3 8B Q6_K abliterated (deviant)"
}
# An explicit -ModelFile wins over the role's default, so a variant (a trained
# student, a different quant) serves through the identical path and alias.
if ($ModelFile) {
    $modelFile = $ModelFile
    $label = "$ModelFile (override)"
}
if ($Ngl -le 0) { $Ngl = $defaultNgl }

$model = Join-Path $ModelDir $modelFile
if (-not (Test-Path $model)) {
    Write-Host "Model not found at $model" -ForegroundColor Yellow
    if ($Role -eq "solver") {
        Write-Host "Fetch the Qwen3-4B-Thinking Q8_0 GGUF into $ModelDir first." -ForegroundColor Yellow
    } else {
        Write-Host "Run scripts\fetch-model.ps1 first." -ForegroundColor Yellow
    }
    exit 1
}

$server = Get-Command llama-server -ErrorAction SilentlyContinue
if (-not $server) {
    Write-Host "llama-server is not on PATH." -ForegroundColor Yellow
    Write-Host "Get a CUDA build from https://github.com/ggml-org/llama.cpp/releases" -ForegroundColor Yellow
    Write-Host "(llama-<ver>-bin-win-cuda-x64.zip), unzip it, and add it to PATH." -ForegroundColor Yellow
    exit 1
}

# Total context is shared across slots, so ask for per-slot x slots.
$ctx = $CtxPerSlot * $Parallel

Write-Host "$label  |  ngl=$Ngl  parallel=$Parallel  ctx=$ctx ($CtxPerSlot/slot)" -ForegroundColor Cyan

& llama-server `
    --model $model `
    --alias samaritan-playout `
    --host 127.0.0.1 --port $Port `
    --n-gpu-layers $Ngl `
    --threads 6 `
    --threads-batch 10 `
    --ctx-size $ctx `
    --parallel $Parallel `
    --cont-batching `
    --cache-reuse 256 `
    --batch-size 512 `
    --ubatch-size 512 `
    --flash-attn on `
    --cache-type-k q8_0 `
    --cache-type-v q8_0 `
    --metrics
