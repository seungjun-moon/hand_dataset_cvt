#!/usr/bin/env bash
# Download EgoDex dataset from https://github.com/apple/ml-egodex
# Files are placed under RAW/egodex/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/../RAW/egodex"
BASE_URL="https://ml-site.cdn-apple.com/datasets/egodex"

PARTS=(test part1 part2 part3 part4 part5)

usage() {
    echo "Usage: $0 [PART ...]"
    echo ""
    echo "Download EgoDex dataset parts into RAW/egodex/."
    echo ""
    echo "Available parts: ${PARTS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0              # Download all parts"
    echo "  $0 test         # Download test set only (16 GB)"
    echo "  $0 part1 part2  # Download specific parts"
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# Use specified parts or default to all
if [[ $# -gt 0 ]]; then
    SELECTED=("$@")
else
    SELECTED=("${PARTS[@]}")
fi

mkdir -p "$RAW_DIR"

for part in "${SELECTED[@]}"; do
    zip_file="${RAW_DIR}/${part}.zip"
    url="${BASE_URL}/${part}.zip"

    if [[ -d "${RAW_DIR}/${part}" ]]; then
        echo "[skip] ${part} already exists at ${RAW_DIR}/${part}"
        continue
    fi

    echo "[download] ${part}.zip ..."
    curl -L -C - -o "$zip_file" "$url"

    echo "[unzip] ${part}.zip ..."
    unzip -q -o "$zip_file" -d "$RAW_DIR"

    rm "$zip_file"
    echo "[done] ${part}"
done

echo "All requested parts downloaded to ${RAW_DIR}"
