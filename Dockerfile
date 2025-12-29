# RunPod Serverless GPU Worker for Hunyuan3D-2.1
# H100 GPU optimized - Maximum quality 3D generation with PBR textures
# Build v25 - Use Hunyuan3D-2.1 with hy3dshape - force no cache

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/.cache/huggingface

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    trimesh \
    pygltflib \
    huggingface_hub \
    pillow \
    requests

# Clone Hunyuan3D-2.1 repository (has hy3dshape module)
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Install Hunyuan3D-2.1 dependencies (ignore errors for optional deps)
RUN pip install --no-cache-dir -r requirements.txt || true

# Set PYTHONPATH so hy3dshape can be imported
ENV PYTHONPATH="/app/hunyuan3d:${PYTHONPATH}"

WORKDIR /app/hunyuan3d

# Copy handler to hunyuan3d directory
COPY gpu-worker/handler.py /app/hunyuan3d/handler.py

# Run handler from hunyuan3d directory
CMD ["python", "-u", "/app/hunyuan3d/handler.py"]
