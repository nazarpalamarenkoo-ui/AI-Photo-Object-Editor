#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/backend/weights"

echo "========================================"
echo "   AI Image Editor - Model Downloader"
echo "========================================"
echo
echo "Weights directory:"
echo "  $WEIGHTS_DIR"
echo

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$WEIGHTS_DIR/lama_cache"
mkdir -p "$WEIGHTS_DIR/rembg"

download_model() {
    local name="$1"
    local url="$2"
    local output="$3"

    if [ -f "$output" ]; then
        echo "[SKIP] $name already exists"
        echo "       $output"
        echo
        return 0
    fi

    echo "[DOWNLOAD] $name"
    echo "           $url"
    echo "           -> $output"

    curl \
        --location \
        --fail \
        --retry 5 \
        --retry-delay 3 \
        --retry-all-errors \
        --continue-at - \
        --output "$output" \
        "$url"

    if [ ! -s "$output" ]; then
        echo "[ERROR] Downloaded file is empty: $output"
        rm -f "$output"
        return 1
    fi

    echo "[OK] $name"
    echo
}

# ---------------------------------------------------------
# YOLOv10m
# ---------------------------------------------------------

download_model \
    "YOLOv10m" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10m.pt" \
    "$WEIGHTS_DIR/yolov10m.pt"

# ---------------------------------------------------------
# MobileSAM
# ---------------------------------------------------------

download_model \
    "MobileSAM" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt" \
    "$WEIGHTS_DIR/mobile_sam.pt"

# ---------------------------------------------------------
# LaMa / Big-LaMa
# ---------------------------------------------------------

download_model \
    "Big-LaMa" \
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" \
    "$WEIGHTS_DIR/lama_cache/big-lama.pt"

# ---------------------------------------------------------
# U2Net / rembg
# ---------------------------------------------------------

download_model \
    "U2Net" \
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx" \
    "$WEIGHTS_DIR/rembg/u2net.onnx"

echo "========================================"
echo "   All model weights are ready"
echo "========================================"
echo

find "$WEIGHTS_DIR" -type f \
    \( -name "*.pt" -o -name "*.onnx" \) \
    -printf "%p\n" 2>/dev/null || true