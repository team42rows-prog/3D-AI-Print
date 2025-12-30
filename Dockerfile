# RunPod Serverless GPU Worker for Hunyuan3D-2.1
# H100 GPU optimized - Maximum quality 3D generation with PBR textures
# Build v33 - Added Cloudflare R2 upload for large files

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/.cache/huggingface
# CUDA architectures for build (no GPU during Docker build)
# 8.0=A100, 8.6=RTX3090, 8.9=RTX4090, 9.0=H100
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"

WORKDIR /app

# Remove system packages that cause distutils conflicts
RUN apt-get update && apt-get remove -y python3-blinker || true && \
    rm -rf /var/lib/apt/lists/*

# Clone Hunyuan3D-2.1 repository
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Fix bpy version: 4.0 doesn't exist, use 4.1.0 from Blender's PyPI
# Also remove torch/torchvision/torchaudio from requirements (we install specific versions)
RUN sed -i 's/bpy==4.0/bpy==4.1.0/g' requirements.txt && \
    sed -i '/^torch/d' requirements.txt && \
    sed -i '/^torchvision/d' requirements.txt && \
    sed -i '/^torchaudio/d' requirements.txt

# Install PyTorch 2.5.1 with CUDA 12.4 FIRST (required by Hunyuan3D-2.1)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install requirements (with Blender's extra index for bpy)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.blender.org/pypi/

# Build custom rasterizer
WORKDIR /app/hunyuan3d/hy3dpaint/custom_rasterizer
RUN pip install -e . --no-build-isolation

# Compile mesh painter
WORKDIR /app/hunyuan3d/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh || true

WORKDIR /app/hunyuan3d

# Download Real-ESRGAN upscaler
RUN mkdir -p hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt || true

# Install runpod and boto3 (for Cloudflare R2 upload)
RUN pip install --no-cache-dir runpod boto3

# Set PYTHONPATH so hy3dshape can be imported
ENV PYTHONPATH="/app/hunyuan3d:${PYTHONPATH}"

# Copy handler to hunyuan3d directory
COPY gpu-worker/handler.py /app/hunyuan3d/handler.py

# Run handler from hunyuan3d directory
CMD ["python", "-u", "/app/hunyuan3d/handler.py"]
