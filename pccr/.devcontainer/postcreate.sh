#!/usr/bin/env bash
set -euo pipefail

log() { echo "[postCreate] $*"; }

# Ensure user-level scripts are on PATH (pip --user installs here)
export PATH="$HOME/.local/bin:$PATH"

# -----------------------------
# 1) Select CUDA arch
# -----------------------------
log "Detecting CUDA arch..."

GPU_ARCH="$(
python - <<'PY'
import torch

try:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}.{minor}")
    else:
        print("")
except Exception:
    print("")
PY
)"

case "${GPU_ARCH}" in
    8.9|9.*|10.*|11.*|12.*)
        # PyTorch 1.10 / CUDA 11.1 do not understand these native targets.
        # Compile Ampere PTX and let the driver JIT it for the newer GPU.
        TORCH_CUDA_ARCH_LIST="8.6+PTX"
        ;;
    8.6)
        TORCH_CUDA_ARCH_LIST="8.6"
        ;;
    8.0)
        TORCH_CUDA_ARCH_LIST="8.0"
        ;;
    7.5)
        TORCH_CUDA_ARCH_LIST="7.5"
        ;;
    7.0)
        TORCH_CUDA_ARCH_LIST="7.0"
        ;;
    *)
        # Safe default for a container created without GPU access.
        TORCH_CUDA_ARCH_LIST="8.6+PTX"
        ;;
esac

export TORCH_CUDA_ARCH_LIST
log "GPU_ARCH=${GPU_ARCH:-not-visible}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# Helpful defaults for building (set MAX_JOBS to a lower value if RAM is limited)
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
log "MAX_JOBS=${MAX_JOBS}"

# We’ll consistently install to user site to avoid permission warnings
PIP_FLAGS="-v --no-build-isolation --no-warn-script-location --user"

# -----------------------------
# 2) Install mmlab stuff
# -----------------------------

# 2a) MMCV (with CUDA ops)
log "Installing mmcv"
# comment out "-f ..." if mmlab server cannot be reached. this leads to local compiling, taking around 10m
pip install ${PIP_FLAGS} mmcv-full==1.6.2 # -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

# 2b) mmdetection
log "Installing mmdet"
pip install ${PIP_FLAGS} "mmdet>=2.24.0,<3.0.0"

# 2c) mmdetection3d
log "Installing mmdet3d"
pip install ${PIP_FLAGS} "git+https://github.com/open-mmlab/mmdetection3d.git@v1.0.0rc6"

# 2d) mmsegmentation
log "Installing mmseg"
pip install ${PIP_FLAGS} "mmsegmentation>=0.20.0,<1.0.0"

# -----------------------------
# 3) Jupyter kernel (re-initialize ipykernel)
# -----------------------------
log "Registering kernel..."
python -m pip install ${PIP_FLAGS} ipykernel
python -m ipykernel install --user --name pccr --display-name "Python (pccr)"

log "Done."
