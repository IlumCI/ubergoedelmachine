<#
.SYNOPSIS
  Refine -ngl around the coarse optimum, and test the AVX-VNNI CPU backend.

.DESCRIPTION
  The coarse sweep steps by 4, so the true peak can sit between samples. This
  checks each layer around it.

  It also tests one thing worth knowing: llama.cpp selected
  `ggml-cpu-haswell.dll` on this machine even though `ggml-cpu-alderlake.dll`
  ships alongside it. Haswell is plain AVX2; the alderlake variant adds
  AVX-VNNI, which accelerates the int8 dot products that k-quants like Q6_K
  lean on. Since most of this model runs on CPU, that is worth a measurement
  rather than an assumption — the backend loader may be declining it for a
  good reason.
#>
param(
    [string]$ModelDir = "$env:USERPROFILE\models",
    [string]$ToolsDir = "$env:USERPROFILE\tools\llama.cpp",
    [int[]]$Ngl = @(16, 17, 18, 19, 20, 21),
    [int]$Tokens = 64
)

$ErrorActionPreference = "Stop"
$model = Join-Path $ModelDir "Huihui-Ministral-3-8B-Reasoning-2512-abliterated.Q6_K.gguf"
$env:PATH = "$ToolsDir;$env:PATH"

function Measure-Tps([int]$n, [string]$backendDir) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if ($backendDir) { $env:GGML_BACKEND_PATH = $backendDir } else { Remove-Item Env:\GGML_BACKEND_PATH -ErrorAction SilentlyContinue }
    $raw = & llama-bench -m $model -ngl $n -p 0 -n $Tokens -t 6 -o json
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { return $null }
    try { return [double](([string]::Join("", $raw) | ConvertFrom-Json)[0].avg_ts) } catch { return $null }
}

Write-Host "Refining -ngl (default backend selection)" -ForegroundColor Cyan
$best = [pscustomobject]@{ Ngl = 0; Tps = 0.0 }
foreach ($n in $Ngl) {
    Write-Host ("  ngl={0,-3} " -f $n) -NoNewline
    $t = Measure-Tps $n $null
    if ($null -eq $t) { Write-Host "failed (out of VRAM)" -ForegroundColor DarkYellow; continue }
    Write-Host ("{0,6:N2} tok/s" -f $t)
    if ($t -gt $best.Tps) { $best = [pscustomobject]@{ Ngl = $n; Tps = $t } }
}

Write-Host ""
Write-Host ("Best: -ngl {0} at {1:N2} tok/s" -f $best.Ngl, $best.Tps) -ForegroundColor Green

# Only the alderlake CPU variant plus the pieces a run cannot do without, so
# the loader has no haswell build to prefer.
Write-Host ""
Write-Host "Testing the AVX-VNNI (alderlake) CPU backend at the best -ngl" -ForegroundColor Cyan
$alt = Join-Path $env:TEMP "ggml-vnni"
New-Item -ItemType Directory -Force -Path $alt | Out-Null
foreach ($f in @("ggml-cpu-alderlake.dll", "ggml-cuda.dll")) {
    $src = Join-Path $ToolsDir $f
    if (Test-Path $src) { Copy-Item $src $alt -Force }
}

$vnni = Measure-Tps $best.Ngl $alt
Remove-Item Env:\GGML_BACKEND_PATH -ErrorAction SilentlyContinue

if ($null -eq $vnni) {
    Write-Host "  alderlake backend did not load - the default choice stands" -ForegroundColor DarkYellow
} else {
    $delta = 100 * ($vnni - $best.Tps) / $best.Tps
    Write-Host ("  alderlake: {0,6:N2} tok/s  ({1:+0.0;-0.0;0}% vs haswell)" -f $vnni, $delta)
    if ($delta -gt 3) {
        Write-Host "  Worth pinning: set GGML_BACKEND_PATH to a dir holding only the alderlake CPU dll." -ForegroundColor Green
    } else {
        Write-Host "  No meaningful gain; llama.cpp's default selection was right." -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host ("  .\scripts\serve.ps1 -Ngl {0} -Parallel 4" -f $best.Ngl) -ForegroundColor Cyan
