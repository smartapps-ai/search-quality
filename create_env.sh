#!/usr/bin/env bash
set -euo pipefail

# Creates a virtual environment at .venv and installs requirements
# Usage: ./create_env.sh  OR  PYTHON=python3.11 ./create_env.sh

PYTHON=${PYTHON:-python3}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON' not found." >&2
  exit 1
fi

# # Require Python >= 3.9 for some build dependencies
# PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
# PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
# PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
# if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
#   echo "Detected Python $PY_VER. This project requires Python >= 3.9 to install build dependencies." >&2
#   echo "Install a newer Python (3.9+) and re-run: PYTHON=python3.11 ./create_env.sh" >&2
#   exit 1
# fi

echo "Creating virtualenv with $PYTHON -> .venv"
$PYTHON -m venv .venv

echo "Activating virtualenv and installing requirements"
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate the environment with: source .venv/bin/activate"
