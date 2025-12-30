# RunPod Serverless GPU Worker for Hunyuan3D-2.1
# H100 GPU optimized - Maximum quality 3D generation with PBR textures
# Build v27 - Add omegaconf

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/.cache/huggingface

WORKDIR /app

# Install core dependencies first
RUN pip install --no-cache-dir \
    runpod \
    trimesh \
    pygltflib \
    huggingface_hub \
    pillow \
    requests \
    diffusers==0.30.0 \
    transformers==4.46.0 \
    accelerate==1.1.1 \
    safetensors \
    einops \
    omegaconf

# Clone Hunyuan3D-2.1 repository (has hy3dshape module)
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Install Hunyuan3D-2.1 dependencies (some may fail for optional deps)
RUN pip install --no-cache-dir -r requirements.txt || echo "Some optional deps failed, continuing..."

# Set PYTHONPATH so hy3dshape can be imported
ENV PYTHONPATH="/app/hunyuan3d:${PYTHONPATH}"

WORKDIR /app/hunyuan3d

# Copy handler to hunyuan3d directory
COPY gpu-worker/handler.py /app/hunyuan3d/handler.py

# Run handler from hunyuan3d directory
CMD ["python", "-u", "/app/hunyuan3d/handler.py"]
