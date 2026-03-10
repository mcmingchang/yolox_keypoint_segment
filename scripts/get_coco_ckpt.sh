#!/usr/bin/env bash
# Download COCO pretrained checkpoint for YOLOX-seg_s export.
# There is no dedicated YOLOX-seg COCO release; we use YOLOX-s (same backbone: depth=0.33, width=0.50).
# Source: https://github.com/Megvii-BaseDetection/YOLOX/releases (README benchmark table)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/weights"
OUTPUT_FILE="$WEIGHTS_DIR/yolox_seg_s_coco.pth"

# Official YOLOX-s COCO checkpoint (0.1.1rc0)
URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
MIN_SIZE_MB=10

mkdir -p "$WEIGHTS_DIR"

get_size_bytes() {
  [ -f "$1" ] || return
  # Portable: wc -c is POSIX; fallback to stat (macOS: -f%z, Linux: -c%s)
  wc -c < "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null
}

if [ -f "$OUTPUT_FILE" ]; then
  SIZE_BYTES=$(get_size_bytes "$OUTPUT_FILE")
  [ -n "$SIZE_BYTES" ] && SIZE_MB=$((SIZE_BYTES / 1024 / 1024))
  if [ -n "$SIZE_MB" ] && [ "$SIZE_MB" -ge "$MIN_SIZE_MB" ]; then
    echo "Checkpoint already present: $OUTPUT_FILE ($SIZE_MB MB)"
    exit 0
  fi
  [ -n "$SIZE_MB" ] && echo "Existing file too small ($SIZE_MB MB), re-downloading."
fi

echo "Downloading YOLOX-s COCO pretrained weights to $OUTPUT_FILE ..."
if command -v curl &>/dev/null; then
  curl -sSL -o "$OUTPUT_FILE" "$URL"
elif command -v wget &>/dev/null; then
  wget -q -O "$OUTPUT_FILE" "$URL"
else
  echo "Error: need curl or wget" >&2
  exit 1
fi

SIZE_BYTES=$(get_size_bytes "$OUTPUT_FILE")
SIZE_MB=$((SIZE_BYTES / 1024 / 1024))
if [ "$SIZE_MB" -lt "$MIN_SIZE_MB" ]; then
  echo "Error: download too small ($SIZE_MB MB), expected >= $MIN_SIZE_MB MB" >&2
  rm -f "$OUTPUT_FILE"
  exit 1
fi
echo "Downloaded $OUTPUT_FILE ($SIZE_MB MB). OK."
