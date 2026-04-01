#!/usr/bin/env bash
# Setup uv virtual environment for hand_dataset_cvt
# Pins versions to match the ego_pipeline conda environment (Python 3.10, CUDA 12.8)
#
# Usage:
#   bash setup_env.sh          # full setup
#   bash setup_env.sh --no-torch  # skip PyTorch (if already installed or CPU-only)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_VERSION="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

NO_TORCH=false
for arg in "$@"; do
    case "$arg" in
        --no-torch) NO_TORCH=true ;;
    esac
done

# ------------------------------------------------------------------
# 1. Check uv is available
# ------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "Using uv $(uv --version)"

# ------------------------------------------------------------------
# 2. Create venv with Python 3.10
# ------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at ${VENV_DIR} ..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

# ------------------------------------------------------------------
# 3. Install PyTorch + CUDA 12.8 (from PyTorch wheel index)
# ------------------------------------------------------------------
if [ "$NO_TORCH" = false ]; then
    echo "Installing PyTorch (CUDA 12.8)..."
    uv pip install --python "$VENV_DIR/bin/python" \
        "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" \
        --index-url "$TORCH_INDEX"
else
    echo "Skipping PyTorch installation (--no-torch)"
fi

# ------------------------------------------------------------------
# 4. Install remaining dependencies
# ------------------------------------------------------------------
echo "Installing project dependencies..."
uv pip install --python "$VENV_DIR/bin/python" \
    "numpy==1.26.4" \
    "h5py==3.15.1" \
    "opencv-python==4.11.0.86" \
    "matplotlib==3.10.8" \
    "scikit-image==0.25.2" \
    "tqdm==4.67.2" \
    "pyyaml==6.0.3" \
    "pillow==12.1.0" \
    "webdataset==1.0.2" \
    "smplx==0.1.28" \
    "manopth @ git+https://github.com/hassony2/manopth.git"

# ------------------------------------------------------------------
# 5. Done
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Environment ready!"
echo "=========================================="
echo ""
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Or run scripts directly:"
echo "  uv run --python ${VENV_DIR}/bin/python python scripts/visualize.py --help"
