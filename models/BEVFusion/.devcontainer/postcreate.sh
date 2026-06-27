#!/usr/bin/env bash
set -euo pipefail

log() { echo "[postCreate] $*"; }

# Ensure user-level scripts are on PATH (pip --user installs here)
export PATH="$HOME/.local/bin:$PATH"

# -----------------------------
# 1) Detect local CUDA archs
# -----------------------------
log "Detecting CUDA archs..."
ARCHS="$(
python - <<'PY'
import torch
try:
    torch.cuda.init()
    caps = {f"{m}.{n}" for i in range(torch.cuda.device_count())
            for (m, n) in [torch.cuda.get_device_capability(i)]}
    print(";".join(sorted(caps)))
except Exception:
    # Fallback if no GPU visible during container create
    print("7.0;7.5;8.0;8.6;8.9;9.0")
PY
)"
export TORCH_CUDA_ARCH_LIST="${ARCHS}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# Helpful defaults for building (set MAX_JOBS to a lower value if RAM is limited)
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
log "MAX_JOBS=${MAX_JOBS}"

PIP_FLAGS="-v --no-build-isolation --no-warn-script-location --user"
if [[ -d "offline_wheels" ]]; then
  log "Found offline_wheels directory. Using offline installation."
  PIP_FLAGS="${PIP_FLAGS} --no-index --find-links=offline_wheels"
  log "Pre-installing build tools..."
  pip install ${PIP_FLAGS} setuptools wheel Cython build
fi

# -----------------------------
# 2) Install editable deps if present
# -----------------------------

# 2a) MMCV (with CUDA ops)
if [[ -d third_party/mmcv ]]; then
  log "Installing MMCV editable with CUDA ops..."
  MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install ${PIP_FLAGS} -e third_party/mmcv
else
  log ">> third_party/mmcv not found; installing as regular package"
  if [[ -d "offline_wheels" ]]; then
    pip install ${PIP_FLAGS} mmcv-full==1.4.0
  else
    pip install ${PIP_FLAGS} mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
  fi
fi

# 2b) mmdetection
if [[ -d third_party/mmdetection ]]; then
  log "Installing mmdetection editable..."
  pip install ${PIP_FLAGS} -e third_party/mmdetection
else
  log ">> third_party/mmdetection not found; installing as regular package"
  pip install ${PIP_FLAGS} mmdet==2.20.0
fi


# -----------------------------
# 3) Install dependenceis that require compilation
# -----------------------------
log "Installing dependencies that require compilation..."
pip install ${PIP_FLAGS} flash-attn==0.2.0


# -----------------------------
# 4) Run setup.py to install codebase
# -----------------------------
log "Installing codebase (running setup.py)..."
python setup.py develop --user

# -----------------------------
# 5) Jupyter kernel (re-initialize ipykernel)
# -----------------------------
log "Registering kernel..."
python -m ipykernel install --user --name bevfusion --display-name "Python (bevfusion)"

log "Done."
