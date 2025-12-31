# RunPod Serverless GPU Worker for Hunyuan3D-2.1
# H100 GPU optimized - Maximum quality 3D generation with PBR textures
# Build v38 - Repaired mesh keeps original format (STL→STL, GLB→GLB) unless output_format specified

FROM nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
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
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default (required by Hunyuan3D-2.1)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create workspace
WORKDIR /app

# Install PyTorch 2.5.1 with CUDA 12.4 (required by Hunyuan3D-2.1)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone Hunyuan3D-2.1 repository (latest version with PBR textures)
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Install Hunyuan3D-2.1 dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install custom rasterizer for texture generation
WORKDIR /app/hunyuan3d/hy3dpaint/custom_rasterizer
RUN pip install -e .

# Compile mesh painter for texture generation
WORKDIR /app/hunyuan3d/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh || true

WORKDIR /app/hunyuan3d

# Download Real-ESRGAN for texture upscaling
RUN mkdir -p hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt || true

# Install additional dependencies (including boto3 for Cloudflare R2 upload)
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    pymeshfix \
    networkx \
    hf_transfer

# Download model weights (cached in Docker image for fast cold start)
# Hunyuan3D-2.1 with all subfolders
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('tencent/Hunyuan3D-2.1')" || true

# Copy handler to hunyuan3d directory (where hy3dshape module is available)
COPY handler.py /app/hunyuan3d/handler.py

WORKDIR /app/hunyuan3d

# Add hunyuan3d to PYTHONPATH so hy3dshape can be imported
ENV PYTHONPATH="/app/hunyuan3d:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Run handler from hunyuan3d directory
CMD ["python", "-u", "/app/hunyuan3d/handler.py"]
