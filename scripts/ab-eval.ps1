<#
.SYNOPSIS
  A/B a trained student against the base it was trained from, on one held-out set.

.DESCRIPTION
  Serves each model in turn through LM Studio under the SAME alias
  (samaritan-playout), the same context, the same GPU offload and the same token
  budget, so the only thing differing between the two eval runs is the weights.
  Each model gets its own progress file: sharing one would let the second model
  inherit the first's answers and report them as its own.

  Defaults to the W7 generated set, which is the right instrument here for two
  reasons: it is contamination-free by construction (generated, post-cutoff), and
  it is calibrated to a 4B. The teacher-calibrated blend is NOT usable for this -
  a 4B truncates on those without answering, so both models floor at ~0 and the
  difference measures nothing.

.PARAMETER Limit
  Items per model. 60 fits a night; higher is better powered but linearly slower.

.PARAMETER MaxTokens
  Per-item generation budget. Must stay under the served context (4096) with room
  for the prompt.

.PARAMETER Dataset
  Reasoning JSONL to evaluate. Defaults to the W7 generated held-out set.

.PARAMETER Models
  Which model keys to run. Pass a single one to probe just that model - e.g. to
  ask whether the baseline's cap-failures are rescued by a bigger budget.

.PARAMETER Ctx
  Served context window. Must exceed MaxTokens plus the prompt, or the model is
  capped by the window instead of the budget you set.

.PARAMETER Tag
  Suffix for log/progress filenames. Use it when re-running the SAME questions
  under a different budget - reusing the capped run's progress file would serve
  its stale answers instead of re-measuring.

.PARAMETER NoStream
  Suppress the live token echo. Streaming is ON by default so you can watch the
  model reason; output goes straight to the console (no pipeline) because
  PowerShell buffers pipelines by line and token output has no newlines.

.PARAMETER RemoteUrl
  Serve from somewhere else (docs/colab/samaritan_serve_gguf.ipynb) instead of
  loading into local LM Studio. The local side is then just reason_eval making
  HTTP calls - a few MB resident - which is the difference between a 4 GB laptop
  GPU being unusable for a night and not being touched at all.

  Serving is one model at a time, so pass a single -Models entry naming whatever
  the notebook has loaded. Do NOT split one model's items across two backends:
  LM Studio and Ollama differ on the sampling knobs the harness does not pin
  (top_k, top_p, min_p), so a progress file half-filled by each measures the
  backend as much as the weights. Use a fresh -Tag when you change hosts.
#>
param(
    [int]$Limit = 60,
    [int]$MaxTokens = 3000,
    [int]$Ctx = 4096,
    [string]$Tag = "",
    [string]$Dataset = "",
    [string[]]$Models = @("base-q4km","student-v1-q4km"),
    [switch]$NoStream,
    [string]$RemoteUrl = ""
)
$ErrorActionPreference = "Continue"
$LMS  = "C:\Users\ilum\.lmstudio\bin\lms.exe"
$repo = "C:\Users\ilum\Projects\memetoken"
$out  = "$env:USERPROFILE\models\reasoning"

# `powershell -File script.ps1 -Models a,b` hands the binder ONE string "a,b",
# not two elements - unlike dot-sourcing or -Command. Left alone, that becomes a
# model key no server has, every item fails, and the run looks like a broken
# tunnel rather than a broken argument. Split here so both call styles agree.
$Models = $Models | ForEach-Object { $_ -split ',' } | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }

if ($RemoteUrl) {
    if ($Models.Count -gt 1) {
        Write-Host "-RemoteUrl serves one model at a time; got $($Models.Count): $($Models -join ', ')" -ForegroundColor Red
        exit 1
    }
    $env:SAMARITAN_URL = $RemoteUrl.TrimEnd('/')
    Write-Host "remote serving: $env:SAMARITAN_URL" -ForegroundColor Cyan

    # Probe once before evaluating. Without this an unreachable server - a stale
    # tunnel, a stopped runtime, or a placeholder left in the URL - fails every
    # item in turn and reads as 40 model failures rather than one bad argument.
    $probe = "$($env:SAMARITAN_URL)/models"
    try {
        $r = Invoke-WebRequest -Uri $probe -Headers @{ Authorization = "Bearer ollama" } `
                               -TimeoutSec 30 -UseBasicParsing
        if ($r.Content -notmatch "samaritan-playout") {
            Write-Host "reachable, but 'samaritan-playout' is not served there." -ForegroundColor Red
            Write-Host "Re-run the notebook cell that creates the alias." -ForegroundColor Red
            exit 1
        }
        Write-Host "preflight OK: samaritan-playout is served" -ForegroundColor Green
    } catch {
        Write-Host "cannot reach $probe" -ForegroundColor Red
        Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Check the tunnel URL is current and the Colab keep-alive cell is running." -ForegroundColor Yellow
        exit 1
    }
} else {
    & $LMS server start --port 1234 | Out-Null
    $env:SAMARITAN_URL = "http://127.0.0.1:1234/v1"
}
$env:SAMARITAN_MODEL = "samaritan-playout"
if (-not $Dataset) { $Dataset = "$out\generated-d2-s777.jsonl" }
$env:DATASET   = $Dataset
$env:LIMIT     = "$Limit"
$env:MAX_TOKENS = "$MaxTokens"
if ($NoStream) { Remove-Item Env:SAMARITAN_STREAM -ErrorAction SilentlyContinue }
else           { $env:SAMARITAN_STREAM = "1" }
Set-Location $repo

foreach ($m in $Models) {
    Write-Host "`n=================== $m ===================" -ForegroundColor Cyan
    if (-not $RemoteUrl) {
        & $LMS unload --all 2>$null | Out-Null
        Start-Sleep -Seconds 3
        & $LMS load $m --identifier samaritan-playout --gpu max -c $Ctx -y 2>$null | Out-Null
        Start-Sleep -Seconds 3
    }
    $env:RESUME = "$out\ab$Tag-$m.progress.jsonl"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # Transcript rather than Tee-Object: a pipeline buffers by line, which would
    # hold back token-level output until a newline arrives.
    Start-Transcript -Path "$out\ab$Tag-$m.log" -Append | Out-Null
    cargo run -q -p samaritan-run --example reason_eval
    Stop-Transcript | Out-Null
    $sw.Stop()
    Write-Host ("[$m] wall clock: {0:N1} min" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Yellow
}
if (-not $RemoteUrl) { & $LMS unload --all 2>$null | Out-Null }
Write-Host "`nDONE. Logs: $out\ab$Tag-*.log" -ForegroundColor Green
if ($RemoteUrl) {
    Write-Host "Remote runtime is still billing - stop the Colab keep-alive cell." -ForegroundColor Yellow
}
