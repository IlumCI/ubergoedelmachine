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
  Per-item generation budget. Must stay under the served context with room for
  the prompt.

  Set it ABOVE the ~95th percentile of what the model actually needs, never near
  the median. Measured 2026-09-16: at 6000 tokens, against a median demand of
  5,548, two runs of the SAME weights on the SAME questions disagreed on 18 of 40
  items - and 15 of those 18 flips were an answer crossing the cap, not a change
  of mind. A budget at the median turns half the set into coin tosses and the
  eval reports the sampler rather than the model.

  Accuracy then measures reasoning; token counts measure efficiency. Conflating
  them into accuracy-at-a-tight-budget measures neither.

.PARAMETER Seed
  Sampling seed, passed to every worker. Fixed by default so a rerun is
  reproducible and two models face identical dice - which cancels most of the
  sampling variance from their DIFFERENCE, the quantity actually under test.
  Vary it deliberately to measure the spread rather than one draw from it.

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

  A server holding ONE model takes a single -Models entry. A server holding
  several (the notebook can load both) takes several, with -RemoteAliases naming
  what each is served as, and they are evaluated concurrently in one launch.

  Do NOT split one model's items across two backends:
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

.PARAMETER RemoteAliases
  Served model names, matched position-by-position to -Models. Supply these when
  the server holds several models at once (see the serving notebook) and both
  halves of a comparison can run in ONE launch, with no unload-edit-reload step
  between them.

  Defaults to samaritan-playout when a single model is run, which is what a
  one-model-at-a-time server serves.

  Running both together does not halve the GPU time - the same questions are
  answered either way - but it removes the human-in-the-middle step, and both
  models then face the same server, the same batching and the same moment, which
  is one fewer difference between them than a sequential pair has.
#>
param(
    [int]$Limit = 60,
    [int]$MaxTokens = 16000,
    [int]$Ctx = 20480,
    [string]$Tag = "",
    [string]$Dataset = "",
    [string[]]$Models = @("base-q4km","student-v1-q4km"),
    [switch]$NoStream,
    [string]$RemoteUrl = "",
    [int]$Shards = 1,
    [int]$Seed = 1234,
    [string[]]$RemoteAliases = @()
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

# The window has to hold the prompt AND the whole generation. When it does not,
# neither server errors: LM Studio stops early and Ollama context-shifts, both
# producing a plausible reply that silently lost its beginning. That is
# indistinguishable from a wrong answer in the results, so refuse up front.
$ctxNeeded = $MaxTokens + 1024
if ($Ctx -lt $ctxNeeded) {
    Write-Host "-Ctx $Ctx is too small for -MaxTokens $MaxTokens." -ForegroundColor Red
    Write-Host "Generation plus prompt must fit the window, or the trace is trimmed with no error." -ForegroundColor Red
    Write-Host "Use -Ctx $ctxNeeded or higher (and match num_ctx on the server)." -ForegroundColor Yellow
    exit 1
}

if ($Shards -gt 1 -and -not $RemoteUrl) {
    Write-Host "-Shards needs -RemoteUrl: N concurrent streams need a server holding N contexts," -ForegroundColor Red
    Write-Host "which the local 4 GB card cannot do. Serve from Colab instead." -ForegroundColor Red
    exit 1
}

if ($RemoteUrl) {
    $env:SAMARITAN_URL = $RemoteUrl.TrimEnd('/')
    Write-Host "remote serving: $env:SAMARITAN_URL" -ForegroundColor Cyan

    $RemoteAliases = $RemoteAliases | ForEach-Object { $_ -split ',' } |
                     Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }
    if (-not $RemoteAliases) { $RemoteAliases = @("samaritan-playout") }
    if ($Models.Count -gt 1 -and $RemoteAliases.Count -le 1) {
        Write-Host "-Models names $($Models.Count) models but no -RemoteAliases were given." -ForegroundColor Red
        Write-Host "All of them would be sent to the same served alias and return the same" -ForegroundColor Red
        Write-Host "answers under different filenames. Pass one alias per model." -ForegroundColor Red
        exit 1
    }
    if ($RemoteAliases.Count -ne $Models.Count) {
        Write-Host "-RemoteAliases has $($RemoteAliases.Count) entries but -Models has $($Models.Count)." -ForegroundColor Red
        Write-Host "They are matched position by position, so the counts must agree." -ForegroundColor Red
        exit 1
    }

    # Probe once before evaluating. Without this an unreachable server - a stale
    # tunnel, a stopped runtime, or a placeholder left in the URL - fails every
    # item in turn and reads as 40 model failures rather than one bad argument.
    $probe = "$($env:SAMARITAN_URL)/models"
    try {
        $r = Invoke-WebRequest -Uri $probe -Headers @{ Authorization = "Bearer ollama" } `
                               -TimeoutSec 30 -UseBasicParsing
        # Check EVERY alias. Finding out at item 1 that only one of two models
        # was registered means the other half of the comparison runs against
        # nothing, and reads as a total model failure rather than a setup slip.
        $absent = @($RemoteAliases | Where-Object { $r.Content -notmatch [regex]::Escape($_) })
        if ($absent) {
            Write-Host "reachable, but not serving: $($absent -join ', ')" -ForegroundColor Red
            Write-Host "Re-run the notebook cell that registers the aliases." -ForegroundColor Red
            exit 1
        }
        Write-Host "preflight OK: $($RemoteAliases -join ', ')" -ForegroundColor Green
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
$env:SEED      = "$Seed"
if ($NoStream) { Remove-Item Env:SAMARITAN_STREAM -ErrorAction SilentlyContinue }
else           { $env:SAMARITAN_STREAM = "1" }
Set-Location $repo

# Several models, served together, evaluated in one launch. Every (model, shard)
# pair is its own worker; they all start, all finish, then each model's shards
# are merged back into the one progress file the analysis reads.
if ($RemoteUrl -and $Models.Count -gt 1) {
    $Utf8NoBom = New-Object System.Text.UTF8Encoding $false
    cargo build -q -p samaritan-run --example reason_eval
    if ($LASTEXITCODE -ne 0) { Write-Host "build failed" -ForegroundColor Red; exit 1 }
    $exe = Join-Path $repo "target\debug\examples\reason_eval.exe"
    $rows = Get-Content $Dataset | Where-Object { $_.Trim() } | Select-Object -First $Limit
    $shardCount = [Math]::Max($Shards, 1)

    Write-Host ("`nrunning {0} model(s) x {1} shard(s) = {2} workers over {3} items each" -f `
        $Models.Count, $shardCount, ($Models.Count * $shardCount), $rows.Count) -ForegroundColor Cyan
    for ($i = 0; $i -lt $Models.Count; $i++) {
        Write-Host ("  {0,-22} -> {1}" -f $Models[$i], $RemoteAliases[$i]) -ForegroundColor Cyan
    }

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $jobs = @()
    $dirs = @{}
    for ($i = 0; $i -lt $Models.Count; $i++) {
        $m = $Models[$i]
        $shardDir = Join-Path $out "shards$Tag-$m"
        New-Item -ItemType Directory -Force -Path $shardDir | Out-Null
        $dirs[$m] = $shardDir
        for ($sIdx = 0; $sIdx -lt $shardCount; $sIdx++) {
            $slice = @(); for ($j = $sIdx; $j -lt $rows.Count; $j += $shardCount) { $slice += $rows[$j] }
            if (-not $slice) { continue }
            $dsPath = Join-Path $shardDir "items-$sIdx.jsonl"
            [System.IO.File]::WriteAllLines($dsPath, $slice, $Utf8NoBom)
            $jobs += Start-Job -ScriptBlock {
                param($exe, $repo, $ds, $resume, $url, $alias, $maxTok, $seed)
                Set-Location $repo
                $env:SAMARITAN_URL = $url; $env:SAMARITAN_MODEL = $alias
                $env:DATASET = $ds; $env:RESUME = $resume
                $env:LIMIT = "100000"; $env:MAX_TOKENS = $maxTok
                if ($seed) { $env:SEED = $seed }
                Remove-Item Env:SAMARITAN_STREAM -ErrorAction SilentlyContinue
                & $exe 2>&1
            } -ArgumentList $exe, $repo, $dsPath, (Join-Path $shardDir "progress-$sIdx.jsonl"),
                            $env:SAMARITAN_URL, $RemoteAliases[$i], "$MaxTokens", "$Seed"
        }
    }

    $last = ""
    while ($jobs | Where-Object { $_.State -eq 'Running' }) {
        Start-Sleep -Seconds 20
        $parts = @()
        foreach ($m in $Models) {
            $n = 0
            Get-ChildItem "$($dirs[$m])\progress-*.jsonl" -ErrorAction SilentlyContinue |
                ForEach-Object { $n += @(Get-Content $_ | Where-Object { $_.Trim() }).Count }
            $parts += "{0} {1}/{2}" -f $m, $n, $rows.Count
        }
        $line = $parts -join "  |  "
        if ($line -ne $last) {
            $last = $line
            Write-Host ("  {0}   ({1:N1} min)" -f $line, $sw.Elapsed.TotalMinutes)
        }
    }
    $jobs | ForEach-Object { Receive-Job $_ | Out-File -Append -Encoding utf8 "$out\ab$Tag-all.log" }
    $jobs | Remove-Job
    $sw.Stop()

    Write-Host ""
    foreach ($m in $Models) {
        $dest = "$out\ab$Tag-$m.progress.jsonl"
        $merged = @(Get-ChildItem "$($dirs[$m])\progress-*.jsonl" -ErrorAction SilentlyContinue |
            ForEach-Object { Get-Content $_ | Where-Object { $_.Trim() } })
        [System.IO.File]::WriteAllLines($dest, $merged, $Utf8NoBom)
        $ok = @($merged | Where-Object { $_ -match '"correct":true' }).Count
        Write-Host ("  {0,-22} {1}/{2} answered, {3} correct -> {4}" -f `
            $m, $merged.Count, $rows.Count, $ok, $dest) -ForegroundColor Green
    }
    Write-Host ("`nwall clock: {0:N1} min for {1} model(s)" -f $sw.Elapsed.TotalMinutes, $Models.Count) -ForegroundColor Yellow
    Write-Host "Remote runtime is still billing - stop the Colab keep-alive cell." -ForegroundColor Yellow
    Write-Host "`nCompare with:  python scripts/ab_analyze.py <fileA> <fileB> nameA nameB" -ForegroundColor Cyan
    exit 0
}

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
