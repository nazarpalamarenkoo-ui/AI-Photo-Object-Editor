$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$WeightsDir = Join-Path $ProjectRoot "backend\weights"

Write-Host "========================================"
Write-Host "   AI Image Editor - Model Downloader"
Write-Host "========================================"
Write-Host ""
Write-Host "Weights directory:"
Write-Host "  $WeightsDir"
Write-Host ""

New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null
New-Item -ItemType Directory -Force -Path "$WeightsDir\lama_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "$WeightsDir\rembg" | Out-Null


function Download-Model {
    param (
        [string]$Name,
        [string]$Url,
        [string]$Output
    )

    if (Test-Path $Output) {
        Write-Host "[SKIP] $Name already exists"
        Write-Host "       $Output"
        Write-Host ""
        return
    }

    Write-Host "[DOWNLOAD] $Name"
    Write-Host "           $Url"
    Write-Host "           -> $Output"

    # curl.exe is available on modern Windows.
    # --continue-at - allows resuming interrupted downloads.
    & curl.exe `
        --location `
        --fail `
        --retry 5 `
        --retry-delay 3 `
        --retry-all-errors `
        --continue-at - `
        --output "$Output" `
        "$Url"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to download $Name"
    }

    if (!(Test-Path $Output)) {
        Write-Error "Downloaded file does not exist: $Output"
    }

    $FileSize = (Get-Item $Output).Length

    if ($FileSize -eq 0) {
        Remove-Item $Output -Force
        Write-Error "Downloaded file is empty: $Output"
    }

    Write-Host "[OK] $Name"
    Write-Host ""
}


# ---------------------------------------------------------
# YOLOv10m
# ---------------------------------------------------------

Download-Model `
    "YOLOv10m" `
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10m.pt" `
    "$WeightsDir\yolov10m.pt"


# ---------------------------------------------------------
# MobileSAM
# ---------------------------------------------------------

Download-Model `
    "MobileSAM" `
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt" `
    "$WeightsDir\mobile_sam.pt"


# ---------------------------------------------------------
# LaMa / Big-LaMa
# ---------------------------------------------------------

Download-Model `
    "Big-LaMa" `
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" `
    "$WeightsDir\lama_cache\big-lama.pt"


# ---------------------------------------------------------
# U2Net / rembg
# ---------------------------------------------------------

Download-Model `
    "U2Net" `
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx" `
    "$WeightsDir\rembg\u2net.onnx"


Write-Host "========================================"
Write-Host "   All model weights are ready"
Write-Host "========================================"
Write-Host ""

Get-ChildItem $WeightsDir -Recurse -File |
    Where-Object {
        $_.Extension -in ".pt", ".onnx"
    } |
    ForEach-Object {
        Write-Host $_.FullName
    }