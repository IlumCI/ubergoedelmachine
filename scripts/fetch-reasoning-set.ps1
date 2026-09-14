<#
.SYNOPSIS
  Fetch a public reasoning set into Samaritan's reasoning-JSONL format.

.DESCRIPTION
  The reasoning surface needs data, and the honest first sets are ones whose
  answers a machine can verify without a judge — exact-match or multiple-choice.
  This fetches from the Hugging Face datasets-server (rows API, so no parquet
  reader is needed) and writes the `{question, answer, answer_kind, domain,
  split}` lines `samaritan_corpus::load_reasoning` reads.

  Eval-oriented sets (the originals):

    gsm8k          openai/gsm8k — grade-school math, the standard set.
    gsm-symbolic   apple/GSM-Symbolic — GSM8K's templates regenerated with fresh
                   numbers and names, built to expose memorization. -Variant
                   picks the tier: main, p1, or p2 (hardest).
    aime25         math-ai/aime25 — AIME 2025 competition math. Integer answers,
                   graded exactly. 30 problems.
    aime26         MathArena/aime_2026 — same format, one year newer. NOT
                   contamination-cleaner for a model released after Feb 2026.

  Trainable multi-domain sets (the 2025 tier — newest verifiable data that
  exists; chosen so the teacher is as unlikely as static data allows to have
  memorized them, per the distillation-corpus review of 2026-09-13):

    deepmath       zwhe99/DeepMath-103K (Apr 2025) — hard, decontaminated math
                   with an explicit final_answer. Free-response, exact-match.
    supergpqa      m-a-p/SuperGPQA (Feb 2025) — 26.5K graduate-level MC across
                   285 disciplines; fetched *balanced by discipline* so no field
                   dominates. Multiple-choice (answer_letter).
    medxpertqa     TsinghuaC3I/MedXpertQA Text (Jan 2025) — expert-level medical
                   MC (options A.., label). Multiple-choice.
    frames         google/frames-benchmark (Sep 2024) — 824 hard multi-hop QA,
                   short answers graded by the normalized matcher (fuzziest of
                   the four; a few misses will be grading, not reasoning).

    blend          all four at 400/600/300/200 into trainable-blend.jsonl,
                   stamped split=train — the multi-domain distillation corpus.
                   Per-set files are also written, so a re-fetch is incremental.

  Sampling note: the big sets are fetched with *strided* pages spread across the
  whole file, because HF datasets are often sorted (by topic or difficulty) and
  a head-slice would silently skew the blend.

  Prefer gsm-symbolic as a *held-out* measure. GPQA and HLE are not fetched here
  — gated/licensed, operator-supplied — and HLE must only ever be held-out.

.EXAMPLE
  scripts\fetch-reasoning-set.ps1 -Dataset blend
  # -> %USERPROFILE%\models\reasoning\trainable-blend.jsonl (split=train)
  # then: $env:DATASET points selftrain_export (or reason_eval) at it.

.PARAMETER Dataset
  Which set to fetch (or `blend` for the trainable multi-domain corpus).

.PARAMETER Variant
  For gsm-symbolic only: main, p1, or p2 (default) — the difficulty tier.

.PARAMETER Split
  Source split on Hugging Face where a set does not pin its own.

.PARAMETER Count
  How many items to fetch. Ignored by `blend`, which uses its fixed mix.

.PARAMETER SplitLabel
  The split stamped on each task: held_out (default) or train. `blend` defaults
  to train (it exists to be trained on) unless you pass this explicitly.
#>
param(
    [ValidateSet("gsm8k", "gsm-symbolic", "aime25", "aime26",
                 "deepmath", "supergpqa", "medxpertqa", "frames", "blend")]
    [string]$Dataset = "gsm-symbolic",
    [ValidateSet("main", "p1", "p2")][string]$Variant = "p2",
    [ValidateSet("train", "test")][string]$Split = "test",
    [int]$Count = 200,
    [ValidateSet("held_out", "train")][string]$SplitLabel = "held_out",
    [string]$Out
)

$ErrorActionPreference = "Stop"
$reasoningDir = Join-Path $env:USERPROFILE "models\reasoning"

# ---------------------------------------------------------------- extractors --
# Each takes a row and returns @{question; answer; domain?} or $null to skip.

# GSM-style: a worked solution ending in "#### <answer>".
$extractGsm = {
    param($row)
    @{ question = $row.question; answer = (($row.answer -split '####')[-1].Trim() -replace ',', '') }
}

# AIME: the statement is the question, the gold answer an integer.
$extractAime = {
    param($row)
    @{ question = $row.problem; answer = ("$($row.answer)").Trim() }
}

# DeepMath: explicit final_answer; drop the pathological-difficulty tail so the
# teacher's verified-trace yield stays worth the compute (difficulty runs ~3-9).
$extractDeepMath = {
    param($row)
    if ($null -ne $row.difficulty -and [double]$row.difficulty -gt 8) { return $null }
    @{ question = $row.question; answer = ("$($row.final_answer)").Trim() }
}

# SuperGPQA: options is a list; letters are positional (A..). answer_letter is
# the gold. The discipline becomes the task's domain, which is also what the
# balanced selection groups by.
$extractSuperGpqa = {
    param($row)
    $opts = @($row.options)
    if ($opts.Count -lt 2) { return $null }
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine(("$($row.question)").TrimEnd())
    [void]$sb.AppendLine()
    [void]$sb.AppendLine("Options:")
    for ($i = 0; $i -lt $opts.Count; $i++) {
        [void]$sb.AppendLine(("{0}) {1}" -f [char](65 + $i), $opts[$i]))
    }
    @{ question = $sb.ToString().TrimEnd()
       answer   = ("$($row.answer_letter)").Trim()
       domain   = ("$($row.discipline)").Trim().ToLower() }
}

# MedXpertQA: options is an OBJECT keyed by letter (A..E or A..J), label is the
# gold letter. Letters come from the keys themselves, so they can never drift
# out of step with the gold.
$extractMedX = {
    param($row)
    $props = @($row.options.PSObject.Properties | Sort-Object Name)
    if ($props.Count -lt 2) { return $null }
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine(("$($row.question)").TrimEnd())
    [void]$sb.AppendLine()
    [void]$sb.AppendLine("Options:")
    foreach ($p in $props) { [void]$sb.AppendLine(("{0}) {1}" -f $p.Name, $p.Value)) }
    @{ question = $sb.ToString().TrimEnd(); answer = ("$($row.label)").Trim() }
}

# FRAMES: Prompt/Answer columns; short factual answers.
$extractFrames = {
    param($row)
    @{ question = $row.Prompt; answer = ("$($row.Answer)").Trim() }
}

# ------------------------------------------------------------------- specs ----
# TotalRows (verified against the HF dataset viewer, 2026-09-13) enables strided
# fetching; AnswerKind is what load_reasoning's serde expects.
function Get-Spec([string]$name) {
    switch ($name) {
        "gsm8k" {
            @{ Repo = "openai/gsm8k"; Config = "main"; Domain = "math"; Extract = $extractGsm
               AnswerKind = "exactMatch"; Tag = "gsm8k-$Split" }
        }
        "gsm-symbolic" {
            @{ Repo = "apple/GSM-Symbolic"; Config = $Variant; Domain = "math"; Extract = $extractGsm
               AnswerKind = "exactMatch"; Tag = "gsm-symbolic-$Variant" }
        }
        "aime25" {
            @{ Repo = "math-ai/aime25"; Config = "default"; Split = "test"; Domain = "math"
               Extract = $extractAime; AnswerKind = "exactMatch"; Tag = "aime25" }
        }
        "aime26" {
            @{ Repo = "MathArena/aime_2026"; Config = "default"; Split = "train"; Domain = "math"
               Extract = $extractAime; AnswerKind = "exactMatch"; Tag = "aime26" }
        }
        "deepmath" {
            @{ Repo = "zwhe99/DeepMath-103K"; Config = "default"; Split = "train"; Domain = "math"
               Extract = $extractDeepMath; AnswerKind = "exactMatch"; Tag = "deepmath"
               TotalRows = 103000 }
        }
        "supergpqa" {
            @{ Repo = "m-a-p/SuperGPQA"; Config = "default"; Split = "train"; Domain = "science"
               Extract = $extractSuperGpqa; AnswerKind = "multipleChoice"; Tag = "supergpqa"
               TotalRows = 26500; Balanced = $true }
        }
        "medxpertqa" {
            @{ Repo = "TsinghuaC3I/MedXpertQA"; Config = "Text"; Split = "test"; Domain = "medicine"
               Extract = $extractMedX; AnswerKind = "multipleChoice"; Tag = "medxpertqa"
               TotalRows = 2450 }
        }
        "frames" {
            @{ Repo = "google/frames-benchmark"; Config = "default"; Split = "test"; Domain = "qa"
               Extract = $extractFrames; AnswerKind = "exactMatch"; Tag = "frames"
               TotalRows = 824 }
        }
        default { throw "unknown dataset spec: $name" }
    }
}

# ------------------------------------------------------------------ fetching --
function Get-Page([string]$repoEnc, [string]$config, [string]$split, [int]$offset, [int]$len) {
    $url = "https://datasets-server.huggingface.co/rows?dataset=$repoEnc&config=$config&split=$split&offset=$offset&length=$len"
    try {
        Invoke-RestMethod -Uri $url -TimeoutSec 90
    } catch {
        Write-Host "rows API request failed at offset ${offset}: $_" -ForegroundColor Yellow
        Write-Host "The dataset may not be auto-converted yet, or the network is down." -ForegroundColor Yellow
        exit 1
    }
}

# Fetch raw rows. Sequential for small sets; strided pages spread across the
# whole file for big ones, because HF sets are often sorted (topic/difficulty)
# and a head-slice would skew the sample. Rows the API truncated are skipped —
# a cut-off question or option list must never be graded.
function Get-Rows($spec, [int]$want) {
    $repoEnc = [System.Uri]::EscapeDataString($spec.Repo)
    $srcSplit = if ($spec.Contains("Split")) { $spec.Split } else { $Split }
    $rows = New-Object System.Collections.Generic.List[object]
    # Over-fetch a little so per-row filters and truncation-skips still leave
    # enough to trim down to `want`.
    $target = [int][Math]::Ceiling($want * 1.4)
    $total = 0
    if ($spec.Contains("TotalRows")) { $total = [int]$spec.TotalRows }

    if ($total -gt ($target * 2)) {
        $pages = [int][Math]::Ceiling($target / 100.0)
        $step = [int][Math]::Floor(($total - 100) / [Math]::Max(1, $pages - 1))
        for ($i = 0; $i -lt $pages; $i++) {
            $off = [Math]::Min($i * $step, [Math]::Max(0, $total - 100))
            $resp = Get-Page $repoEnc $spec.Config $srcSplit $off 100
            if (-not $resp.rows) { continue }
            foreach ($entry in $resp.rows) {
                if ($entry.truncated_cells -and @($entry.truncated_cells).Count -gt 0) { continue }
                $rows.Add($entry.row)
            }
            Write-Host ("  {0} rows (offset {1})..." -f $rows.Count, $off) -ForegroundColor DarkGray
        }
    } else {
        $offset = 0
        while ($rows.Count -lt $target) {
            $len = [Math]::Min(100, $target - $rows.Count)
            $resp = Get-Page $repoEnc $spec.Config $srcSplit $offset $len
            if (-not $resp.rows -or @($resp.rows).Count -eq 0) { break }
            foreach ($entry in $resp.rows) {
                if ($entry.truncated_cells -and @($entry.truncated_cells).Count -gt 0) { continue }
                $rows.Add($entry.row)
            }
            $offset += @($resp.rows).Count
            Write-Host ("  {0} rows..." -f $rows.Count) -ForegroundColor DarkGray
            if (@($resp.rows).Count -lt $len) { break }   # ran out
        }
    }
    return ,$rows
}

# Round-robin across domains so one discipline cannot dominate a slice (only
# meaningful for specs whose extractor emits a per-row domain, e.g. SuperGPQA).
function Select-Balanced($items, [int]$want) {
    $groups = @($items | Group-Object { $_.domain })
    if ($groups.Count -le 1) {
        if ($items.Count -le $want) { return ,$items }
        return ,@($items[0..($want - 1)])
    }
    $picked = New-Object System.Collections.Generic.List[object]
    $index = @{}
    foreach ($g in $groups) { $index[$g.Name] = 0 }
    while ($picked.Count -lt $want) {
        $advanced = $false
        foreach ($g in $groups) {
            if ($picked.Count -ge $want) { break }
            $i = $index[$g.Name]
            if ($i -lt $g.Group.Count) {
                $picked.Add($g.Group[$i])
                $index[$g.Name] = $i + 1
                $advanced = $true
            }
        }
        if (-not $advanced) { break }   # every group exhausted
    }
    return ,$picked
}

# Fetch one spec, transform, and write its JSONL. Returns the written lines so
# `blend` can concatenate without re-reading files.
function Invoke-Fetch($spec, [int]$want, [string]$label, [string]$outPath) {
    Write-Host ("fetching {0} [{1}] x{2} -> {3}" -f $spec.Repo, $spec.Config, $want, $outPath) -ForegroundColor Cyan
    $raw = Get-Rows $spec $want

    $items = New-Object System.Collections.Generic.List[object]
    foreach ($row in $raw) {
        $x = & $spec.Extract $row
        if ($null -eq $x) { continue }
        if (-not $x.question -or -not $x.answer) { continue }
        if (-not $x.ContainsKey("domain") -or -not $x.domain) { $x.domain = $spec.Domain }
        $items.Add($x)
    }
    if ($items.Count -eq 0) { Write-Host "no usable rows from $($spec.Repo)." -ForegroundColor Red; exit 1 }

    $chosen = if ($spec.Contains("Balanced") -and $spec.Balanced) {
        Select-Balanced $items $want
    } elseif ($items.Count -gt $want) {
        ,@($items[0..($want - 1)])
    } else {
        ,$items
    }

    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($x in $chosen) {
        $obj = [ordered]@{
            question    = $x.question
            answer      = $x.answer
            answer_kind = $spec.AnswerKind
            domain      = $x.domain
            split       = $label
        }
        $lines.Add(($obj | ConvertTo-Json -Compress))
    }

    $dir = Split-Path $outPath -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    # UTF-8 *without* a BOM: a BOM on the first line would break the JSONL reader.
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($outPath, $lines, $utf8NoBom)
    Write-Host ("wrote {0} items to {1}" -f $lines.Count, $outPath) -ForegroundColor Green
    return ,$lines
}

# --------------------------------------------------------------------- main ---
if ($Dataset -eq "blend") {
    # The trainable multi-domain corpus: balanced across base domains and answer
    # formats so the student generalizes instead of overfitting one field. It
    # exists to be trained on, so split defaults to `train` here.
    $label = if ($PSBoundParameters.ContainsKey("SplitLabel")) { $SplitLabel } else { "train" }
    $mix = @(
        @{ Name = "deepmath";   Count = 400 },
        @{ Name = "supergpqa";  Count = 600 },
        @{ Name = "medxpertqa"; Count = 300 },
        @{ Name = "frames";     Count = 200 }
    )
    $buckets = New-Object System.Collections.Generic.List[object]
    foreach ($m in $mix) {
        $spec = Get-Spec $m.Name
        $path = Join-Path $reasoningDir "$($spec.Tag).jsonl"
        $lines = Invoke-Fetch $spec $m.Count $label $path
        $buckets.Add(@{ Name = $m.Name; Lines = @($lines) })
    }
    # INTERLEAVE, do not concatenate. A run over this corpus can be cut short at
    # any point (a Colab drop, a token budget, a LIMIT), and a dataset-ordered
    # file would hand that run a single domain -- silently reproducing the
    # math-only bias this blend exists to avoid. Each set is spread evenly across
    # the final order by fractional position, so ANY prefix is domain-balanced.
    $keyed = New-Object System.Collections.Generic.List[object]
    foreach ($b in $buckets) {
        $n = [Math]::Max(1, $b.Lines.Count)
        for ($i = 0; $i -lt $b.Lines.Count; $i++) {
            $keyed.Add([pscustomobject]@{
                Key = ($i + 0.5) / $n; Set = $b.Name; Line = $b.Lines[$i]
            })
        }
    }
    $all = @($keyed | Sort-Object Key, Set | ForEach-Object { $_.Line })
    $blendOut = if ($Out) { $Out } else { Join-Path $reasoningDir "trainable-blend.jsonl" }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($blendOut, $all, $utf8NoBom)
    Write-Host ("blend: {0} items -> {1} (split={2})" -f $all.Count, $blendOut, $label) -ForegroundColor Green
    Write-Host "next (with the teacher served and SAMARITAN_URL set):" -ForegroundColor Cyan
    Write-Host ("  `$env:DATASET = `"{0}`"; cargo run -p samaritan-run --example selftrain_export" -f $blendOut) -ForegroundColor DarkGray
    exit 0
}

$spec = Get-Spec $Dataset
if (-not $Out) { $Out = Join-Path $reasoningDir "$($spec.Tag).jsonl" }
$null = Invoke-Fetch $spec $Count $SplitLabel $Out
Write-Host "next: serve.ps1 -Role solver (or the A100 tunnel), then" -ForegroundColor Cyan
Write-Host ("  `$env:DATASET = `"{0}`"; cargo run -p samaritan-run --example reason_eval" -f $Out) -ForegroundColor DarkGray
