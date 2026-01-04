# RunPod Serverless GPU Worker for Hunyuan3D-2.1
# H100 GPU optimized - Maximum quality 3D generation with PBR textures
# Build v44 - Unified Dockerfile with pyrender EGL/OSMesa support
#
# Actions supported:
#   - generate: Text/Image to 3D (Hunyuan3D-2.1)
#   - validate: Mesh validation + auto-repair (pymeshfix)
#   - render_preview: Offscreen mesh rendering (pyrender + EGL)
#   - render_painted: PBR texture generation + rendering (Paint 2.1)
#   - generate_multiview: Multi-view 3D generation (Hunyuan3D-2mv)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/.cache/huggingface
# CUDA architectures: 8.0=A100, 8.6=RTX3090, 8.9=RTX4090, 9.0=H100
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"

# Pyrender: Use EGL for GPU-accelerated headless rendering (preferred on NVIDIA)
# Falls back to OSMesa if EGL unavailable
ENV PYOPENGL_PLATFORM=egl

WORKDIR /app

# =============================================================================
# System Dependencies
# =============================================================================
# - OSMesa + EGL for pyrender offscreen rendering
# - Mesa/OpenGL libs for 3D rendering
# - Build tools for CUDA extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6-dev \
    freeglut3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopengl0 \
    libglx0 \
    libegl1 \
    libglvnd0 \
    ninja-build \
    && apt-get remove -y python3-blinker || true \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# PyTorch 2.5.1 (required by Hunyuan3D-2.1)
# =============================================================================
# Base image has 2.4.0, upgrade to 2.5.1 for compatibility
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# =============================================================================
# Hunyuan3D-2.1 Setup
# =============================================================================
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Fix bpy version: 4.0 doesn't exist, use 4.1.0 from Blender's PyPI
# Remove torch requirements (we installed specific versions above)
RUN sed -i 's/bpy==4.0/bpy==4.1.0/g' requirements.txt && \
    sed -i '/^torch/d' requirements.txt && \
    sed -i '/^torchvision/d' requirements.txt && \
    sed -i '/^torchaudio/d' requirements.txt

# Install Hunyuan3D requirements (with Blender's PyPI for bpy)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.blender.org/pypi/

# Build custom rasterizer for texture generation
WORKDIR /app/hunyuan3d/hy3dpaint/custom_rasterizer
RUN pip install -e . --no-build-isolation

# Compile mesh painter for texture generation
WORKDIR /app/hunyuan3d/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh || true

WORKDIR /app/hunyuan3d

# Download Real-ESRGAN upscaler for texture enhancement
RUN mkdir -p hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt || true

# =============================================================================
# Additional Python Dependencies
# =============================================================================
# - runpod: RunPod serverless SDK
# - boto3: Cloudflare R2 / S3 upload
# - pymeshfix: Advanced mesh repair for validate action
# - pyrender + PyOpenGL: Offscreen 3D rendering for render_preview
# - networkx: Graph operations for trimesh
# - hf_transfer: Fast HuggingFace downloads
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    pymeshfix \
    networkx \
    hf_transfer \
    pyrender \
    PyOpenGL \
    PyOpenGL_accelerate

# =============================================================================
# Environment & Handler
# =============================================================================
# PYTHONPATH for Hunyuan3D imports
ENV PYTHONPATH="/app/hunyuan3d:${PYTHONPATH}"

# Copy handler
COPY gpu-worker/handler.py /app/hunyuan3d/handler.py

# Run handler
CMD ["python", "-u", "/app/hunyuan3d/handler.py"]
