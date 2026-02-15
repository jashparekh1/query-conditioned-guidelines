#!/usr/bin/env bash
# Create a fresh venv and install dependencies. Run from repo root.
# Requires Python >= 3.10. Load your desired Python first, e.g.:
#   module load python/3.11.9
#   ./scripts/setup_venv.sh
# Optional: ./scripts/setup_venv.sh cuda   to also install requirements-cuda.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python
fi

echo "Using: $($PYTHON --version)"
echo "Creating venv at $REPO_ROOT/venv ..."

if [[ -d venv ]]; then
  echo "Removing existing venv ..."
  rm -rf venv
fi

"$PYTHON" -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ "${1:-}" == "cuda" ]]; then
  echo "Installing CUDA extras ..."
  pip install -r requirements-cuda.txt
fi

echo "Done. Activate with: source venv/bin/activate"
