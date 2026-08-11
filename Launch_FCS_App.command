#!/bin/bash
# Double-click this file to launch the FCS/FCCS Analysis app in your browser.
# First run sets up a local Python environment (.venv) and installs dependencies;
# subsequent runs are fast.
set -e
cd "$(dirname "$0")"
APP_DIR="$(pwd)"
VENV_DIR="$APP_DIR/.venv"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ (e.g. via python.org or Homebrew) and try again."
  read -p "Press Enter to close..."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "First run: setting up the app environment (this can take a minute)..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
fi

echo "Starting FCS/FCCS Analysis app..."
"$VENV_DIR/bin/streamlit" run "$APP_DIR/Home.py"
