<#
.SYNOPSIS
  Download the Q6_K weights Samaritan runs on.

.DESCRIPTION
  Huihui-Ministral-3-8B-Reasoning-2512-abliterated, Q6_K, 6.97 GB.

  Q6_K is the floor you asked for and it is well above the point where
  quantisation damage matters for this size of model. It also does not fit in
  4 GB of VRAM, which is why serve.ps1 does a split load. See the note at the
  bottom of README.md for the arithmetic.

  The mmproj files are deliberately not fetched. This is a multimodal
  checkpoint, but nothing in the harness sends it an image, and the projector
  would cost VRAM that is better spent on another two transformer layers.
#>
param(
    [string]$ModelDir = "$env:USERPROFILE\models",
    [ValidateSet("Q6_K", "Q8_0")]
    [string]$Quant = "Q6_K"
)

$ErrorActionPreference = "Stop"

$repo = "mradermacher/Huihui-Ministral-3-8B-Reasoning-2512-abliterated-GGUF"
$file = "Huihui-Ministral-3-8B-Reasoning-2512-abliterated.$Quant.gguf"
$url  = "https://huggingface.co/$repo/resolve/main/$file"
$dest = Join-Path $ModelDir $file

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if (Test-Path $dest) {
    $gb = (Get-Item $dest).Length / 1GB
    Write-Host ("Already present: {0} ({1:N2} GB)" -f $dest, $gb) -ForegroundColor Green
    exit 0
}

$expected = @{ "Q6_K" = 6.49; "Q8_0" = 8.41 }[$Quant]
Write-Host ("Fetching {0} (~{1:N2} GB) -> {2}" -f $file, $expected, $dest) -ForegroundColor Cyan
Write-Host "This is a large download; it resumes if interrupted." -ForegroundColor DarkGray

# curl handles resume and shows progress; Invoke-WebRequest buffers the whole
# body in memory, which is a poor idea at seven gigabytes.
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if ($curl) {
    & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "download failed with exit code $LASTEXITCODE" }
} else {
    Invoke-WebRequest -Uri $url -OutFile $dest
}

$gb = (Get-Item $dest).Length / 1GB
Write-Host ("Done: {0:N2} GB" -f $gb) -ForegroundColor Green
Write-Host "Next: scripts\tune-ngl.ps1 to find the best split, then scripts\serve.ps1" -ForegroundColor Cyan
