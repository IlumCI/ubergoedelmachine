<#
.SYNOPSIS
  Serve Huihui-Ministral-3-8B-Reasoning-2512-abliterated Q6_K for Samaritan.

.DESCRIPTION
  Tuned for this machine specifically:

      i7-12650H   6 P-cores + 4 E-cores, 16 threads
      16 GB       DDR4-3200, dual channel (~38 GB/s real)
      RTX 3050    Laptop, 4 GB VRAM (~3.5 GB usable), ~192 GB/s

  Q6_K is 6.97 GB and does not fit in 4 GB, so this is a split load: as many
  layers as fit go to the GPU, the rest run on CPU out of system RAM.

  Why each setting is what it is:

  -ngl        Partial offload. GPU memory is ~5x faster per byte than system
              RAM here, so every layer moved across is a real win — but only
              until VRAM runs out, after which llama.cpp spills and it gets
              worse, not better. The right number is measured, not guessed:
              run tune-ngl.ps1 and put the winner here.

  -t 6        Threads for generation: P-cores only. Alder Lake is
              heterogeneous and llama.cpp splits work evenly across threads,
              so a batch finishes when the slowest thread does. Adding the
              four E-cores makes every batch wait on them.

  -tb 10      Threads for prompt processing. That phase is compute-bound
              rather than latency-bound, so the E-cores do help there.

  --parallel 4 -cb
              The largest single throughput lever on this box. Generation is
              memory-bandwidth-bound: the weights are read once per token,
              whether that token belongs to one sequence or four. Batching
              four playouts amortises the read across all of them, so
              aggregate tokens/sec goes up roughly 2-2.5x even though each
              individual response is no faster. NRPA playouts are
              embarrassingly parallel, which is exactly the shape this wants.

  --cache-reuse 256
              Keeps the KV state of a matching prompt prefix. Samaritan lays
              its prompts out stable-part-first precisely so this hits; see
              samaritan-agent/src/prompt.rs.

  -fa on / -ctk q8_0 / -ctv q8_0
              Flash attention plus an 8-bit KV cache. Both cut memory traffic,
              which is the binding constraint.

  --no-mmap   Read the whole model into RAM up front. With 16 GB total and a
              7 GB model there is room, and it avoids page-cache eviction
              stalling generation later in a long run.

.PARAMETER Ngl
  GPU layers to offload. Default 14; run tune-ngl.ps1 to find the real best.
#>
param(
    [int]$Ngl = 14,
    [int]$Port = 8080,
    [string]$ModelDir = "$env:USERPROFILE\models",
    [int]$Parallel = 4,
    [int]$CtxPerSlot = 4096
)

$ErrorActionPreference = "Stop"
$model = Join-Path $ModelDir "Huihui-Ministral-3-8B-Reasoning-2512-abliterated.Q6_K.gguf"

if (-not (Test-Path $model)) {
    Write-Host "Model not found at $model" -ForegroundColor Yellow
    Write-Host "Run scripts\fetch-model.ps1 first." -ForegroundColor Yellow
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

Write-Host "Ministral-3 8B Q6_K  |  ngl=$Ngl  parallel=$Parallel  ctx=$ctx ($CtxPerSlot/slot)" -ForegroundColor Cyan

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
    --no-mmap `
    --metrics
