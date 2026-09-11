<#
.SYNOPSIS
  Install llama.cpp (CUDA build) so serve.ps1 and tune-ngl.ps1 can run.

.DESCRIPTION
  Fetches llama-server, llama-bench and friends from the upstream release and
  unpacks them into a local tools directory, then tells you what to add to
  PATH. Nothing is installed system-wide and nothing touches the registry.

  CUDA 13.3 is selected to match the toolkit already on this machine
  (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3). A CUDA build is
  required even though most of the model runs on CPU: without it `-ngl` does
  nothing and the split load is not available at all.

  The cudart archive carries the CUDA runtime DLLs. It is fetched by default
  because a matching toolkit on PATH is not the same as the exact runtime the
  binaries were linked against, and the failure mode when they disagree is an
  unhelpful "the application was unable to start correctly".
#>
param(
    [string]$Build = "b10907",
    [string]$Cuda = "13.3",
    [string]$ToolsDir = "$env:USERPROFILE\tools\llama.cpp",
    [switch]$SkipCudart
)

$ErrorActionPreference = "Stop"

$base = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
$tmp = Join-Path $env:TEMP "llamacpp-$Build"
New-Item -ItemType Directory -Force -Path $tmp, $ToolsDir | Out-Null

function Get-Archive([string]$name) {
    $dest = Join-Path $tmp $name
    if (Test-Path $dest) {
        Write-Host "  cached: $name" -ForegroundColor DarkGray
        return $dest
    }
    Write-Host "  fetching $name" -ForegroundColor Cyan
    & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o $dest "$base/$name"
    if ($LASTEXITCODE -ne 0) { throw "download of $name failed ($LASTEXITCODE)" }
    return $dest
}

$archives = @("llama-$Build-bin-win-cuda-$Cuda-x64.zip")
if (-not $SkipCudart) { $archives += "cudart-llama-bin-win-cuda-$Cuda-x64.zip" }

foreach ($a in $archives) {
    $zip = Get-Archive $a
    Write-Host "  unpacking $a" -ForegroundColor DarkGray
    Expand-Archive -Path $zip -DestinationPath $ToolsDir -Force
}

$server = Get-ChildItem -Path $ToolsDir -Recurse -Filter "llama-server.exe" |
          Select-Object -First 1
if (-not $server) { throw "llama-server.exe not found under $ToolsDir after unpacking" }
$binDir = $server.Directory.FullName

Write-Host ""
Write-Host "Installed to $binDir" -ForegroundColor Green
# No `2>&1` here. llama-server writes its banner to stderr, and in Windows
# PowerShell redirecting a native command's stderr wraps each line in a
# NativeCommandError -- which, under $ErrorActionPreference = "Stop", kills
# the script after a completely successful install.
& $server.FullName --version
if ($LASTEXITCODE -ne 0) { throw "llama-server would not run (exit $LASTEXITCODE)" }

if ($env:PATH -notlike "*$binDir*") {
    Write-Host ""
    Write-Host "Add it to PATH for this session:" -ForegroundColor Cyan
    Write-Host "  `$env:PATH = `"$binDir;`$env:PATH`""
    Write-Host ""
    Write-Host "Or permanently, for your user:" -ForegroundColor Cyan
    Write-Host "  [Environment]::SetEnvironmentVariable('PATH', `"$binDir;`" + [Environment]::GetEnvironmentVariable('PATH','User'), 'User')"
}

Write-Host ""
Write-Host "Then: .\scripts\tune-ngl.ps1" -ForegroundColor Cyan
