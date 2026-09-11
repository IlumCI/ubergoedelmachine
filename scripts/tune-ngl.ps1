<#
.SYNOPSIS
  Find how many layers actually belong on the GPU.

.DESCRIPTION
  Q6_K is 6.97 GB against 4 GB of VRAM, so some layers run on the GPU and the
  rest on the CPU. Where to put the line is not a thing to reason about from
  a spec sheet: the useful number depends on the KV cache, the context size,
  whatever else is holding VRAM, and the driver's own overhead.

  So measure it. This sweeps -ngl, records tokens/sec, and stops when it stops
  improving — which is the real signal, because one layer past the limit
  llama.cpp spills and throughput falls off a cliff rather than plateauing.

  Expect the winner to be somewhere near 14 on a 4 GB card with nothing else
  running, and lower if a browser is open. Run it again if you change the
  context size, because KV cache comes out of the same 4 GB.
#>
param(
    [string]$ModelDir = "$env:USERPROFILE\models",
    [int]$Start = 6,
    [int]$Max = 26,
    [int]$Step = 4,
    [int]$Tokens = 64,
    [int]$CtxPerSlot = 4096,
    [int]$Parallel = 4
)

$ErrorActionPreference = "Stop"
$model = Join-Path $ModelDir "Huihui-Ministral-3-8B-Reasoning-2512-abliterated.Q6_K.gguf"
if (-not (Test-Path $model)) { throw "model not found at $model; run fetch-model.ps1" }

$bench = Get-Command llama-bench -ErrorAction SilentlyContinue
if (-not $bench) {
    throw "llama-bench is not on PATH. It ships in the same llama.cpp release archive as llama-server."
}

Write-Host "Sweeping -ngl for Q6_K. Close other GPU consumers first." -ForegroundColor Cyan
Write-Host ""

$results = @()
$best = [pscustomobject]@{ Ngl = 0; Tps = 0.0 }

for ($ngl = $Start; $ngl -le $Max; $ngl += $Step) {
    Write-Host ("  ngl={0,-3} " -f $ngl) -NoNewline

    # -n generation only: prompt processing scales differently and would
    # flatter a setting that is bad for the phase we spend our time in.
    $raw = & llama-bench -m $model -ngl $ngl -n $Tokens -p 0 -t 6 -o json 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "failed (likely out of VRAM) - stopping" -ForegroundColor DarkYellow
        break
    }

    try {
        $tps = ([string]::Join("", $raw) | ConvertFrom-Json)[0].avg_ts
    } catch {
        Write-Host "unparseable output - stopping" -ForegroundColor DarkYellow
        break
    }

    Write-Host ("{0,6:N2} tok/s" -f $tps)
    $results += [pscustomobject]@{ Ngl = $ngl; Tps = [double]$tps }

    if ($tps -gt $best.Tps) {
        $best = [pscustomobject]@{ Ngl = $ngl; Tps = [double]$tps }
    } elseif ($tps -lt $best.Tps * 0.92) {
        # A real drop rather than noise: past the VRAM limit, and it only gets
        # worse from here.
        Write-Host "  throughput fell off - past the limit, stopping" -ForegroundColor DarkYellow
        break
    }
}

Write-Host ""
if ($best.Ngl -eq 0) { throw "no configuration completed; check that llama-bench runs at all" }

Write-Host ("Best: -ngl {0} at {1:N2} tok/s single-stream" -f $best.Ngl, $best.Tps) -ForegroundColor Green
Write-Host ""
Write-Host ("  .\scripts\serve.ps1 -Ngl {0} -Parallel {1}" -f $best.Ngl, $Parallel) -ForegroundColor Cyan
Write-Host ""
Write-Host ("Aggregate throughput with {0} slots will be roughly 2-2.5x that number:" -f $Parallel) -ForegroundColor DarkGray
Write-Host ("generation is bandwidth-bound, so batched sequences share one read of the weights." ) -ForegroundColor DarkGray
