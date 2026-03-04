# Image for query-conditioned-guidelines: CUDA + Python + vLLM + project deps.
# Project code is bind-mounted at run time; this image only provides the env.
# Build from repo root: docker build -f containers/Dockerfile -t qcg:latest .

ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python && pip install --upgrade pip

WORKDIR /workspace

# Copy only requirement files so we can install deps; repo is mounted at run time.
COPY requirements.txt requirements-vllm.txt requirements-cuda.txt ./

# Install project deps + vLLM (order matters: base first, then vllm, then cuda extras if needed)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-vllm.txt \
    && (pip install --no-cache-dir -r requirements-cuda.txt || true)

# Default: assume project is mounted at /workspace/query-conditioned-guidelines
WORKDIR /workspace/query-conditioned-guidelines
ENV PYTHONPATH=/workspace/query-conditioned-guidelines

CMD ["bash"]
