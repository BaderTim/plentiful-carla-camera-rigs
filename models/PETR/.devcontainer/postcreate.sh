#!/usr/bin/env bash
set -euo pipefail

log() { echo "[postCreate] $*"; }

# Ensure user-level scripts are on PATH
export PATH="$HOME/.local/bin:$PATH"

# 1) Set local CUDA archs
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# Helpful defaults for building
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
log "MAX_JOBS=${MAX_JOBS}"

PIP_FLAGS="-v --no-build-isolation --no-warn-script-location"

# Install mmdetection3d (v0.17.1)
if [[ ! -d mmdetection3d ]]; then
  log "Cloning mmdetection3d v0.17.1..."
  git clone https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
  cd mmdetection3d
  git checkout v0.17.1
  # requirements are already in Docker, but we do -e for compilation of CUDA ops
  pip install ${PIP_FLAGS} -e .
  cd ..
else
  log "mmdetection3d already exists. Re-installing in editable mode to ensure CUDA ops are compiled for local GPU."
  cd mmdetection3d
  pip install ${PIP_FLAGS} -e .
  cd ..
fi

# Jupyter kernel (optional but nice)
log "Registering Jupyter kernel..."
python -m ipykernel install --user --name petr --display-name "Python 3 (PETR)"

log "Done."
