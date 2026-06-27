#!/usr/bin/env bash
set -euo pipefail

log() { echo "[postCreate] $*"; }

# Ensure user-level scripts are on PATH (pip --user installs here)
export PATH="$HOME/.local/bin:$PATH"

pip install -v -e .

log "Done."