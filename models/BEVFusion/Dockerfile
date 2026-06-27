# --- Build-time Arguments  ---
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu 20.04 (glibc >= 2.28), CUDA 11.1, cuDNN 8 (devel)
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Re-declare args in this stage
ARG DEBIAN_FRONTEND=noninteractive

# Fail fast in RUN
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# ---- System deps for building CUDA ops and common tooling ----
# Note: Ubuntu 20.04 uses Python 3.8 by default
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential cmake ninja-build pkg-config \
      libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
      openmpi-bin openmpi-common libopenmpi-dev \
      ffmpeg \
      python3 python3-pip python3-venv python3-dev \
      python3-tk \
      vim nano htop less sudo wget git  \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3 and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip

# ---- Install PyTorch 1.10.1 + cu111 (and torchvision/torchaudio) ----
RUN pip install --no-cache-dir \
      torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 \
      -f https://download.pytorch.org/whl/cu111/torch_stable.html

# ---- Install base Python deps for notebooks, typing, linting ----
RUN pip install --no-cache-dir \
      jupyterlab ipykernel \
      typing-extensions pyright \
      black isort ruff \
      mmengine \
      mpi4py==3.0.3 numba==0.48.0 Pillow==8.4.0 numpy==1.23.5 "protobuf<5" \
      scipy matplotlib opencv-python tqdm yacs addict yapf torchpack nuscenes-devkit

# Add non-root dev user 
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ENV PATH="/home/dev/.local/bin:${PATH}"
WORKDIR /workspace

# Keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
