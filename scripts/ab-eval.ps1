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

.PARAMETER Shards
  Run N reason_eval workers concurrently over interleaved slices of the items,
  then merge. reason_eval is one request deep, so a single run leaves a serving
  GPU mostly idle no matter how large it is - decode for one stream is bound by
  memory bandwidth, and the weights are shared across a batch. Four shards is
  worth more than four times the silicon, and costs less.

  Remote only: parallel streams need the server to hold N contexts at once,
  which a 4 GB laptop card cannot do. Set OLLAMA_NUM_PARALLEL on the server to
  at least this value, or the requests queue and nothing is gained.

  Keep Shards IDENTICAL across the models being compared. Batched kernels are
  not bit-identical to single-stream ones, so a pair split across two batching
  regimes has one more difference in it than the weights.
#>
param(
    [int]$Limit = 60,
    [int]$MaxTokens = 3000,
    [int]$Ctx = 4096,
    [string]$Tag = "",
    [string]$Dataset = "",
    [string[]]$Models = @("base-q4km","student-v1-q4km"),
    [switch]$NoStream,
    [string]$RemoteUrl = "",
    [int]$Shards = 1
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

if ($Shards -gt 1 -and -not $RemoteUrl) {
    Write-Host "-Shards needs -RemoteUrl: N concurrent streams need a server holding N contexts," -ForegroundColor Red
    Write-Host "which the local 4 GB card cannot do. Serve from Colab instead." -ForegroundColor Red
    exit 1
}

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

    if ($Shards -gt 1) {
        # reason_eval is strictly sequential, so throughput is one stream deep no
        # matter how big the GPU is. A server holding a 4B model runs several
        # streams at nearly the same per-stream speed - the weights are shared
        # across the batch - so N shards is a far bigger lever than faster
        # silicon. Split the items, run N processes, concatenate the results.
        $shardDir = Join-Path $out "shards$Tag-$m"
        New-Item -ItemType Directory -Force -Path $shardDir | Out-Null
        $Utf8NoBom = New-Object System.Text.UTF8Encoding $false

        # Build once. N concurrent `cargo run`s would serialise on the build lock
        # and race to write the same exe.
        cargo build -q -p samaritan-run --example reason_eval
        if ($LASTEXITCODE -ne 0) { Write-Host "build failed" -ForegroundColor Red; exit 1 }
        $exe = Join-Path $repo "target\debug\examples\reason_eval.exe"

        $rows = Get-Content $Dataset | Where-Object { $_.Trim() } | Select-Object -First $Limit
        Write-Host "sharding $($rows.Count) items across $Shards workers" -ForegroundColor Cyan

        $jobs = @()
        for ($s = 0; $s -lt $Shards; $s++) {
            $slice = @(); for ($j = $s; $j -lt $rows.Count; $j += $Shards) { $slice += $rows[$j] }
            if (-not $slice) { continue }
            $dsPath = Join-Path $shardDir "items-$s.jsonl"
            # NOT Set-Content -Encoding utf8: on PowerShell 5.1 that writes a BOM,
            # and three stray bytes before the first '{' make every shard fail to
            # parse as "expected value at line 1 column 1".
            [System.IO.File]::WriteAllLines($dsPath, $slice, $Utf8NoBom)
            $jobs += Start-Job -ScriptBlock {
                param($exe, $repo, $ds, $resume, $url, $model, $maxTok, $seed)
                Set-Location $repo
                $env:SAMARITAN_URL = $url; $env:SAMARITAN_MODEL = $model
                $env:DATASET = $ds; $env:RESUME = $resume
                $env:LIMIT = "100000"; $env:MAX_TOKENS = $maxTok
                if ($seed) { $env:SEED = $seed }
                Remove-Item Env:SAMARITAN_STREAM -ErrorAction SilentlyContinue
                & $exe 2>&1
            } -ArgumentList $exe, $repo, $dsPath, (Join-Path $shardDir "progress-$s.jsonl"),
                            $env:SAMARITAN_URL, $env:SAMARITAN_MODEL, "$MaxTokens", $env:SEED
        }

        $done = 0
        while ($jobs | Where-Object { $_.State -eq 'Running' }) {
            Start-Sleep -Seconds 20
            $n = 0
            Get-ChildItem "$shardDir\progress-*.jsonl" -ErrorAction SilentlyContinue |
                ForEach-Object { $n += @(Get-Content $_ | Where-Object { $_.Trim() }).Count }
            if ($n -ne $done) {
                $done = $n
                Write-Host ("  {0}/{1} items  ({2:N1} min)" -f $done, $rows.Count, $sw.Elapsed.TotalMinutes)
            }
        }
        $jobs | ForEach-Object { Receive-Job $_ | Out-File -Append -Encoding utf8 "$out\ab$Tag-$m.log" }
        $jobs | Remove-Job

        # Merge into the single progress file the analysis expects - again BOM-free,
        # so downstream readers see JSON on line 1 rather than three stray bytes.
        $merged = @(Get-ChildItem "$shardDir\progress-*.jsonl" -ErrorAction SilentlyContinue |
            ForEach-Object { Get-Content $_ | Where-Object { $_.Trim() } })
        [System.IO.File]::WriteAllLines($env:RESUME, $merged, $Utf8NoBom)
        $total = @(Get-Content $env:RESUME | Where-Object { $_.Trim() }).Count
        Write-Host "merged $total items -> $env:RESUME" -ForegroundColor Green
    } else {
        # Transcript rather than Tee-Object: a pipeline buffers by line, which would
        # hold back token-level output until a newline arrives.
        Start-Transcript -Path "$out\ab$Tag-$m.log" -Append | Out-Null
        cargo run -q -p samaritan-run --example reason_eval
        Stop-Transcript | Out-Null
    }
    $sw.Stop()
    Write-Host ("[$m] wall clock: {0:N1} min" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Yellow
}
if (-not $RemoteUrl) { & $LMS unload --all 2>$null | Out-Null }
Write-Host "`nDONE. Logs: $out\ab$Tag-*.log" -ForegroundColor Green
if ($RemoteUrl) {
    Write-Host "Remote runtime is still billing - stop the Colab keep-alive cell." -ForegroundColor Yellow
}
