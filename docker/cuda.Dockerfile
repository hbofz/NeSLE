FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/nesle
COPY . /workspace/nesle

RUN python3 -m pip install --break-system-packages -e '.[dev,rl]'

ENV NESLE_CUDA_ARCH=sm_80
ENV NESLE_PIP_INSTALL_FLAGS=--break-system-packages
ENV PYTHONPATH=src

CMD ["python3", "-m", "pytest", "tests/", "-v"]
