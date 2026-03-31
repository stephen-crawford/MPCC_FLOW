#!/bin/bash
# Helper script to run tests using uv
# This ensures the correct Python environment is used

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use uv to run Python with the project environment
export PATH="$HOME/.local/bin:$PATH"
uv run python "$@"

