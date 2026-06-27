# --- Build-time Arguments  ---
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu 20.04 (glibc >= 2.28), CUDA 11.1, cuDNN 8 (devel)
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Re-declare args in this stage
ARG DEBIAN_FRONTEND=noninteractive

# ---- System deps for building Fast-BEV and CUDA ops ----
RUN apt-get update && apt-get install -y --no-install-recommends \
            ca-certificates curl ffmpeg git gosu \
      build-essential cmake ninja-build pkg-config \
      libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
      openmpi-bin openmpi-common libopenmpi-dev \
      python3 python3-pip python3-venv python3-dev \
      python3-tk \
            htop less nano sudo vim wget \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3 and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip 'setuptools<60' wheel

COPY requirements.txt /tmp/fast-bev/requirements.txt
COPY requirements /tmp/fast-bev/requirements

# ---- Install PyTorch 1.10.1 + cu111 (and torchvision/torchaudio) ----
RUN pip install --no-cache-dir \
      torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 \
      -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install the pinned OpenMMLab stack from the README.
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1


# Install repository runtime deps and a small dev-tooling layer for the container.
RUN pip install --no-cache-dir -r /tmp/fast-bev/requirements/runtime.txt && \
        pip install --no-cache-dir \
            Cython build pytest \
            black isort ruff pyright \
            jupyterlab ipykernel yapf==0.32.0 && \
        rm -rf /tmp/fast-bev

# Add non-root dev user 
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
WORKDIR /workspace

# Keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
