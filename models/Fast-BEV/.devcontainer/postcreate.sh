#!/usr/bin/env bash
set -euo pipefail

log() { echo "[postCreate] $*"; }

# Ensure user-level scripts are on PATH
export PATH="$HOME/.local/bin:$PATH"

# Build the checked-out Fast-BEV repo against the local CUDA toolchain.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6+PTX}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
log "MAX_JOBS=${MAX_JOBS}"
log "FORCE_CUDA=${FORCE_CUDA}"

PIP_FLAGS="-v --no-build-isolation --no-warn-script-location"

cd /workspace

if [[ ! -f setup.py ]]; then
  log "Expected to run from the Fast-BEV repository root at /workspace, but setup.py was not found."
  exit 1
fi

log "Installing Fast-BEV in editable mode..."
python -m pip install --user ${PIP_FLAGS} -e .

# Jupyter kernel (optional but nice)
log "Registering Jupyter kernel..."
python -m ipykernel install --user --name fast-bev --display-name "Python 3 (Fast-BEV)"

log "Done."
