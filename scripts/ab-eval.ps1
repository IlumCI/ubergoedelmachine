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

.PARAMETER NoStream
  Suppress the live token echo. Streaming is ON by default so you can watch the
  model reason; output goes straight to the console (no pipeline) because
  PowerShell buffers pipelines by line and token output has no newlines.
#>
param(
    [int]$Limit = 60,
    [int]$MaxTokens = 3000,
    [switch]$NoStream
)
$ErrorActionPreference = "Continue"
$LMS  = "C:\Users\ilum\.lmstudio\bin\lms.exe"
$repo = "C:\Users\ilum\Projects\memetoken"
$out  = "$env:USERPROFILE\models\reasoning"

& $LMS server start --port 1234 | Out-Null
$env:SAMARITAN_URL   = "http://127.0.0.1:1234/v1"
$env:SAMARITAN_MODEL = "samaritan-playout"
$env:DATASET   = "$out\generated-d2-s777.jsonl"
$env:LIMIT     = "$Limit"
$env:MAX_TOKENS = "$MaxTokens"
if ($NoStream) { Remove-Item Env:SAMARITAN_STREAM -ErrorAction SilentlyContinue }
else           { $env:SAMARITAN_STREAM = "1" }
Set-Location $repo

foreach ($m in @("base-q4km","student-v1-q4km")) {
    Write-Host "`n=================== $m ===================" -ForegroundColor Cyan
    & $LMS unload --all 2>$null | Out-Null
    Start-Sleep -Seconds 3
    & $LMS load $m --identifier samaritan-playout --gpu max -c 4096 -y 2>$null | Out-Null
    Start-Sleep -Seconds 3
    $env:RESUME = "$out\ab-$m.progress.jsonl"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # Transcript rather than Tee-Object: a pipeline buffers by line, which would
    # hold back token-level output until a newline arrives.
    Start-Transcript -Path "$out\ab-$m.log" -Append | Out-Null
    cargo run -q -p samaritan-run --example reason_eval
    Stop-Transcript | Out-Null
    $sw.Stop()
    Write-Host ("[$m] wall clock: {0:N1} min" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Yellow
}
& $LMS unload --all 2>$null | Out-Null
Write-Host "`nDONE. Logs: $out\ab-*.log" -ForegroundColor Green
