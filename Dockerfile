# RunPod Serverless GPU Worker for Hunyuan3D-2.1
# H100 GPU optimized - Maximum quality 3D generation with PBR textures
# Build v28 - Follow official Hunyuan3D-2.1 installation

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/.cache/huggingface

WORKDIR /app

# Upgrade to PyTorch 2.5.1 (required by Hunyuan3D-2.1)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone Hunyuan3D-2.1 repository
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Build custom rasterizer
WORKDIR /app/hunyuan3d/hy3dpaint/custom_rasterizer
RUN pip install -e .

# Compile mesh painter
WORKDIR /app/hunyuan3d/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh || true

WORKDIR /app/hunyuan3d

# Download Real-ESRGAN upscaler
RUN mkdir -p hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt || true

# Install runpod
RUN pip install --no-cache-dir runpod

# Set PYTHONPATH so hy3dshape can be imported
ENV PYTHONPATH="/app/hunyuan3d:${PYTHONPATH}"

# Copy handler to hunyuan3d directory
COPY gpu-worker/handler.py /app/hunyuan3d/handler.py

# Run handler from hunyuan3d directory
CMD ["python", "-u", "/app/hunyuan3d/handler.py"]
