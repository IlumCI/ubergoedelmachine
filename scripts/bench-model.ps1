<#
.SYNOPSIS
  Decide whether a GGUF is fast enough to be Samaritan's reasoning substrate on
  this machine.

.DESCRIPTION
  Runs llama-bench at one or more -ngl values and reports the two numbers that
  actually decide latency — prompt-processing throughput (pp, the prefill) and
  generation throughput (tg) — then turns them into the wall-clock a *real*
  Samaritan call costs and grades it against a budget.

  Why not just "tok/s": a Samaritan reasoning call is `prefill(prompt) +
  generate(answer)`. The prompt is not tiny — system instructions, lessons, a
  retrieved snippet — and a *reasoning-distilled* model does not emit a 900-token
  answer, it emits a long thinking trace first. So the honest cost is
  `PromptTokens/pp + ThinkTokens/tg`, and a model that looks fast on a 128-token
  bench can still blow the budget once it thinks for two thousand tokens. This
  script makes that visible.

  The budget default (120 s) is the user's line: 900 tokens should not take more
  than ~2 minutes. Set -ThinkTokens to what a reasoning model actually spends
  (often 1500-3000 on a hard question) to see the real picture, not the
  flattering one.

  Optionally (-MeasureLoad) it also times how long llama-server takes to load
  the weights and answer — "how the server handles model loading" — by starting
  it, polling /v1/models, and shutting it down.

.EXAMPLE
  scripts\bench-model.ps1 -Model $env:USERPROFILE\models\some-reasoning-7b.Q5_K_M.gguf -Ngl 12,16,20
#>
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$ToolsDir = "$env:USERPROFILE\tools\llama.cpp",
    [int[]]$Ngl = @(18),
    [int]$Threads = 6,
    # A representative Samaritan reasoning prompt: system + question + a couple of
    # lessons + one retrieved snippet.
    [int]$PromptTokens = 1024,
    # What the model actually generates per answer. 900 is the floor (answer
    # only); raise it to a reasoning model's real thinking length to be honest.
    [int]$ThinkTokens = 900,
    # Wall-clock budget per call, seconds.
    [double]$BudgetSecs = 120,
    [switch]$MeasureLoad
)

$ErrorActionPreference = "Stop"
$env:PATH = "$ToolsDir;$env:PATH"
$bench = Join-Path $ToolsDir "llama-bench.exe"
if (-not (Test-Path $bench)) { throw "llama-bench not found at $bench" }
if (-not (Test-Path $Model)) { throw "model not found at $Model" }

$sizeGb = (Get-Item $Model).Length / 1GB
Write-Host ("model: {0} ({1:N2} GB)" -f (Split-Path $Model -Leaf), $sizeGb) -ForegroundColor Cyan
Write-Host ("budget: {0:N0} s per call  |  prompt {1} tok, generate {2} tok" -f $BudgetSecs, $PromptTokens, $ThinkTokens) -ForegroundColor DarkGray
Write-Host ""

function Bench-At([int]$n) {
    # Two rows: pp (n_gen=0) and tg (n_prompt=0). avg_ts is tok/s for each.
    $raw = & $bench -m $Model -ngl $n -t $Threads -p 512 -n 128 -o json 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    try { $rows = ([string]::Join("", $raw) | ConvertFrom-Json) } catch { return $null }
    $pp = ($rows | Where-Object { $_.n_prompt -gt 0 -and $_.n_gen -eq 0 } | Select-Object -First 1).avg_ts
    $tg = ($rows | Where-Object { $_.n_gen -gt 0 -and $_.n_prompt -eq 0 } | Select-Object -First 1).avg_ts
    if (-not $pp -or -not $tg) { return $null }
    [pscustomobject]@{ Ngl = $n; Pp = [double]$pp; Tg = [double]$tg }
}

$results = @()
foreach ($n in $Ngl) {
    Write-Host ("  benchmarking -ngl {0} ..." -f $n) -ForegroundColor DarkGray
    $r = Bench-At $n
    if ($null -eq $r) {
        Write-Host ("  -ngl {0}: did not load or produced no result (likely out of VRAM)" -f $n) -ForegroundColor Yellow
        continue
    }
    $prefill = $PromptTokens / $r.Pp
    $gen = $ThinkTokens / $r.Tg
    $total = $prefill + $gen
    $verdict = if ($total -le $BudgetSecs) { "PASS" } else { "OVER" }
    $results += [pscustomobject]@{
        Ngl = $r.Ngl
        "pp tok/s" = [math]::Round($r.Pp, 1)
        "tg tok/s" = [math]::Round($r.Tg, 1)
        "prefill s" = [math]::Round($prefill, 1)
        "gen s" = [math]::Round($gen, 1)
        "call s" = [math]::Round($total, 1)
        verdict = $verdict
    }
}

if ($results.Count -eq 0) { Write-Host "no -ngl value produced a result." -ForegroundColor Red; exit 1 }

Write-Host ""
$results | Format-Table -AutoSize

$best = $results | Sort-Object "call s" | Select-Object -First 1
$bestColor = if ($best.verdict -eq "PASS") { "Green" } else { "Yellow" }
Write-Host ("best: -ngl {0} -> {1:N0} s/call ({2})" -f $best.Ngl, $best."call s", $best.verdict) -ForegroundColor $bestColor
if ($best.verdict -ne "PASS") {
    Write-Host ("  over the {0:N0}s budget. Options: a smaller/more-quantized model, a shorter thinking budget (-ThinkTokens), or accept the longer per-call time." -f $BudgetSecs) -ForegroundColor DarkGray
}

if ($MeasureLoad) {
    Write-Host "`nmeasuring server load time..." -ForegroundColor Cyan
    $server = Join-Path $ToolsDir "llama-server.exe"
    $nBest = $best.Ngl
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $server -ArgumentList @(
        "-m", $Model, "--n-gpu-layers", $nBest, "-t", $Threads, "--port", "8099", "--host", "127.0.0.1"
    ) -PassThru -WindowStyle Hidden
    try {
        $ready = $false
        while ($sw.Elapsed.TotalSeconds -lt 300 -and -not $proc.HasExited) {
            try {
                $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8099/v1/models" -TimeoutSec 2 -UseBasicParsing
                if ($resp.StatusCode -eq 200) { $ready = $true; break }
            } catch { Start-Sleep -Milliseconds 500 }
        }
        if ($ready) {
            Write-Host ("  weights loaded and serving in {0:N1} s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
        } else {
            Write-Host "  server did not become ready within 300 s" -ForegroundColor Yellow
        }
    } finally {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    }
}
