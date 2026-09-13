<#
.SYNOPSIS
  Fetch Humanity's Last Exam (text-only) into the HleQuestion JSONL the
  `samaritan-eval` HLE runner reads.

.DESCRIPTION
  HLE (cais/hle) is GATED and licensed, so — unlike the sets in
  fetch-reasoning-set.ps1 — the public datasets-server won't serve it. You must
  first accept the terms at https://huggingface.co/datasets/cais/hle with your
  Hugging Face account, then pass a read-scoped HF token here; this authenticates
  the datasets-server rows API with it.

  HLE is multi-modal, but Samaritan's solver path is text-only, so this keeps only
  the text questions (rows with no image) — an image question sent as text would
  just be graded as a miss. The output is `{id, question, answer, answer_type}`,
  exactly what `samaritan_eval::load` reads.

  Do NOT commit or redistribute the output — HLE asks that the benchmark not be
  re-shared. It is written under your models directory, held-out, for your own
  scoring only.

.EXAMPLE
  scripts\fetch-hle.ps1 -HfToken hf_xxx -Count 500
  $env:HLE_DATASET = "$env:USERPROFILE\models\hle\hle-text.jsonl"
  $env:JUDGE = "1"; $env:SAMARITAN_MODEL = "samaritan-playout:latest"
  cargo run -p samaritan-eval --example hle    # with SAMARITAN_URL pointed at the A100

.PARAMETER HfToken
  A Hugging Face read token (hf_…) for an account that has accepted the cais/hle
  terms. Not logged. If omitted, the script reads $env:HF_TOKEN.

.PARAMETER Count
  How many text questions to fetch. The rows API pages 100 at a time.

.PARAMETER Split
  Source split on Hugging Face (default test — HLE's benchmark split).

.PARAMETER Out
  Output path. Defaults to $env:USERPROFILE\models\hle\hle-text.jsonl.
#>
param(
    [string]$HfToken = $env:HF_TOKEN,
    [int]$Count = 500,
    [string]$Split = "test",
    [string]$Out
)

$ErrorActionPreference = "Stop"

if (-not $HfToken) {
    Write-Host "No HF token. Accept the terms at https://huggingface.co/datasets/cais/hle," -ForegroundColor Yellow
    Write-Host "create a read token at https://huggingface.co/settings/tokens, then:" -ForegroundColor Yellow
    Write-Host "  scripts\fetch-hle.ps1 -HfToken hf_xxx   (or set `$env:HF_TOKEN)" -ForegroundColor Yellow
    exit 2
}

if (-not $Out) {
    $Out = Join-Path "$env:USERPROFILE\models\hle" "hle-text.jsonl"
}

$repo = "cais/hle"
$repoEnc = [System.Uri]::EscapeDataString($repo)
$headers = @{ Authorization = "Bearer $HfToken" }
Write-Host ("fetching {0} [{1}] text-only x{2} -> {3}" -f $repo, $Split, $Count, $Out) -ForegroundColor Cyan

# Skip image questions: the row's `image` is empty/null for text-only items.
function Test-HasImage($row) {
    $img = $row.image
    if ($null -eq $img) { return $false }
    return (("$img").Trim().Length -gt 0)
}

$lines = New-Object System.Collections.Generic.List[string]
$offset = 0
$kept = 0
$scanned = 0
while ($kept -lt $Count) {
    $len = 100
    $url = "https://datasets-server.huggingface.co/rows?dataset=$repoEnc&config=default&split=$Split&offset=$offset&length=$len"
    try {
        $resp = Invoke-RestMethod -Uri $url -Headers $headers -TimeoutSec 60
    } catch {
        Write-Host "rows API request failed: $_" -ForegroundColor Yellow
        Write-Host "Check the token has read scope and you accepted the cais/hle terms." -ForegroundColor Yellow
        exit 1
    }
    if (-not $resp.rows -or $resp.rows.Count -eq 0) { break }
    foreach ($entry in $resp.rows) {
        $r = $entry.row
        $scanned++
        if (Test-HasImage $r) { continue }
        if (-not $r.question -or -not $r.answer) { continue }
        $atype = if ($r.answer_type) { $r.answer_type } else { "exactMatch" }
        $obj = [ordered]@{
            id          = "$($r.id)"
            question    = $r.question
            answer      = "$($r.answer)"
            answer_type = $atype
        }
        $lines.Add(($obj | ConvertTo-Json -Compress))
        $kept++
        if ($kept -ge $Count) { break }
    }
    $offset += $resp.rows.Count
    Write-Host ("  scanned {0}, kept {1} text-only..." -f $scanned, $kept) -ForegroundColor DarkGray
    if ($resp.rows.Count -lt $len) { break }   # ran out
}

if ($lines.Count -eq 0) { Write-Host "no text-only rows fetched." -ForegroundColor Red; exit 1 }

$dir = Split-Path $Out -Parent
New-Item -ItemType Directory -Force -Path $dir | Out-Null
# UTF-8 without a BOM: a BOM on the first line would break the JSONL reader.
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllLines($Out, $lines, $utf8NoBom)

Write-Host ("wrote {0} text-only questions to {1}" -f $lines.Count, $Out) -ForegroundColor Green
Write-Host "next (with the A100 served and SAMARITAN_URL set):" -ForegroundColor Cyan
Write-Host ("  `$env:HLE_DATASET=`"{0}`"; `$env:JUDGE=`"1`"; `$env:SAMARITAN_MODEL=`"samaritan-playout:latest`"" -f $Out) -ForegroundColor DarkGray
Write-Host "  cargo run -p samaritan-eval --example hle" -ForegroundColor DarkGray
Write-Host "Do not commit or redistribute this file (HLE license)." -ForegroundColor Yellow
