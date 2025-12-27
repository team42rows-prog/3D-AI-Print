# RunPod Serverless GPU Worker for Hunyuan3D
# A100/L4 GPU optimized

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System dependencies (including OpenGL for PyMeshLab)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopengl0 \
    libglx0 \
    libegl1 \
    libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create workspace
WORKDIR /app

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone Hunyuan3D repository
RUN git clone https://github.com/Tencent/Hunyuan3D-2.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Install Hunyuan3D dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers>=4.38.0 \
    diffusers>=0.27.0 \
    accelerate>=0.27.0 \
    safetensors \
    trimesh \
    pygltflib \
    xformers

# Download model weights (will be cached in Docker image)
# Note: This increases image size but reduces cold start time
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('tencent/Hunyuan3D-2', local_dir='./weights')" || true

# Copy handler
COPY handler.py /app/handler.py

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Run handler
CMD ["python", "-u", "handler.py"]
