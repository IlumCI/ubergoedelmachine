<#
.SYNOPSIS
  Fetch a public reasoning set into Samaritan's reasoning-JSONL format.

.DESCRIPTION
  The reasoning surface needs data, and the honest first set is one whose answers
  a machine can verify without a judge — an exact match. This fetches such a set
  from the Hugging Face datasets-server (rows API, so no parquet reader is
  needed) and writes it as the `{question, answer, answer_kind, domain, split}`
  lines `samaritan_corpus::load_reasoning` reads.

  Three sets, all public and all exact-match:

    gsm8k          openai/gsm8k — grade-school math, the standard set.
    gsm-symbolic   apple/GSM-Symbolic — GSM8K's templates regenerated with fresh
                   numbers and names, built to expose memorization: models that
                   scored high on GSM8K dropped here because they had learned the
                   originals, not the reasoning. Use -Variant to pick the tier:
                   main, p1 (one extra clause), or p2 (two — the hardest, and the
                   one that most separates reasoning from recall).
    aime25         math-ai/aime25 — AIME 2025 competition math, a real step up in
                   difficulty. Integer answers (0-999), so the numeric grader
                   scores them exactly — a hard eval with no LLM judge needed.
                   Only 30 problems, so the number is a coarse but honest probe.

  Prefer gsm-symbolic as the *held-out* measure precisely because it cannot have
  leaked into any base model's training the way GSM8K has. GPQA and HLE are not
  fetched here — gated / licensed, so operator-supplied — and HLE must only ever
  be held-out, never trained on.

  Split label matters for the training loop, not for `reason_eval` (which loads
  everything as held-out): keep a set you might train on labelled `train`, and
  anything you only measure against labelled `held_out`.

.EXAMPLE
  scripts\fetch-reasoning-set.ps1 -Count 200
  cargo run -p samaritan-run --example reason_eval    # then, with the server up:
  #   $env:DATASET = "$env:USERPROFILE\models\reasoning\gsm8k-test.jsonl"

.PARAMETER Dataset
  Which known set to fetch: gsm8k or gsm-symbolic (default).

.PARAMETER Variant
  For gsm-symbolic only: main, p1, or p2 (default) — the difficulty tier.

.PARAMETER Split
  Source split on Hugging Face: test (default) or train. gsm-symbolic is
  test-only.

.PARAMETER Count
  How many items to fetch. The rows API pages 100 at a time; this handles that.

.PARAMETER SplitLabel
  The split to stamp on each task: held_out (default, safe — never trained on)
  or train.
#>
param(
    [ValidateSet("gsm8k", "gsm-symbolic", "aime25")][string]$Dataset = "gsm-symbolic",
    [ValidateSet("main", "p1", "p2")][string]$Variant = "p2",
    [ValidateSet("train", "test")][string]$Split = "test",
    [int]$Count = 200,
    [ValidateSet("held_out", "train")][string]$SplitLabel = "held_out",
    [string]$Out
)

$ErrorActionPreference = "Stop"

# Answers in both sets are a worked solution ending in "#### <answer>". The
# answer is not always an integer (GSM-Symbolic p2 has decimals), so it is taken
# verbatim after the marker, not coerced.
$extract = {
    param($row)
    @{ question = $row.question; answer = (($row.answer -split '####')[-1].Trim() -replace ',', '') }
}

# Per-dataset specifics: the HF repo, its config, the domain, and the extractor.
$spec = switch ($Dataset) {
    "gsm8k" {
        @{ Repo = "openai/gsm8k"; Config = "main"; Domain = "math"; Extract = $extract; Tag = "gsm8k-$Split" }
    }
    "gsm-symbolic" {
        # The memorization-robust variant: templates regenerated with fresh
        # numbers/names, so it cannot have leaked into a base model's training.
        @{ Repo = "apple/GSM-Symbolic"; Config = $Variant; Domain = "math"; Extract = $extract; Tag = "gsm-symbolic-$Variant" }
    }
    "aime25" {
        # A real step up from grade-school: AIME 2025 competition math. Answers
        # are integers 0-999, so the numeric grader scores them exactly — a hard
        # eval that still needs no LLM judge. 30 problems (Count caps higher but
        # the fetch stops when the set runs out). Config is `default`, split test.
        $extractAime = {
            param($row)
            @{ question = $row.problem; answer = ("$($row.answer)").Trim() }
        }
        @{ Repo = "math-ai/aime25"; Config = "default"; Domain = "math"; Extract = $extractAime; Tag = "aime25" }
    }
}

if (-not $Out) {
    $Out = Join-Path "$env:USERPROFILE\models\reasoning" "$($spec.Tag).jsonl"
}

$repoEnc = [System.Uri]::EscapeDataString($spec.Repo)
Write-Host ("fetching {0} [{1}] x{2} -> {3}" -f $spec.Repo, $Split, $Count, $Out) -ForegroundColor Cyan

$rows = New-Object System.Collections.Generic.List[object]
$offset = 0
while ($rows.Count -lt $Count) {
    $len = [Math]::Min(100, $Count - $rows.Count)
    $url = "https://datasets-server.huggingface.co/rows?dataset=$repoEnc&config=$($spec.Config)&split=$Split&offset=$offset&length=$len"
    try {
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 60
    } catch {
        Write-Host "rows API request failed: $_" -ForegroundColor Yellow
        Write-Host "The dataset may not be auto-converted yet, or the network is down." -ForegroundColor Yellow
        exit 1
    }
    if (-not $resp.rows -or $resp.rows.Count -eq 0) { break }
    foreach ($r in $resp.rows) { $rows.Add($r.row) }
    $offset += $resp.rows.Count
    Write-Host ("  {0} rows..." -f $rows.Count) -ForegroundColor DarkGray
    if ($resp.rows.Count -lt $len) { break }   # ran out
}

if ($rows.Count -eq 0) { Write-Host "no rows fetched." -ForegroundColor Red; exit 1 }

$lines = New-Object System.Collections.Generic.List[string]
foreach ($row in $rows) {
    $x = & $spec.Extract $row
    if (-not $x.question -or -not $x.answer) { continue }
    $obj = [ordered]@{
        question    = $x.question
        answer      = $x.answer
        answer_kind = "exactMatch"
        domain      = $spec.Domain
        split       = $SplitLabel
    }
    $lines.Add(($obj | ConvertTo-Json -Compress))
}

$dir = Split-Path $Out -Parent
New-Item -ItemType Directory -Force -Path $dir | Out-Null
# UTF-8 *without* a BOM: a BOM on the first line would break the JSONL reader.
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllLines($Out, $lines, $utf8NoBom)

Write-Host ("wrote {0} items to {1}" -f $lines.Count, $Out) -ForegroundColor Green
Write-Host "next: serve.ps1 -Role solver, then" -ForegroundColor Cyan
Write-Host ("  `$env:DATASET = `"{0}`"; cargo run -p samaritan-run --example reason_eval" -f $Out) -ForegroundColor DarkGray
