# RunPod Serverless GPU Worker for Hunyuan3D
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

# Clone Hunyuan3D repository
RUN git clone https://github.com/Tencent/Hunyuan3D-2.git /app/hunyuan3d

WORKDIR /app/hunyuan3d

# Install Hunyuan3D dependencies (ignore errors for optional deps)
RUN pip install --no-cache-dir -r requirements.txt || true

WORKDIR /app

# Copy handler from gpu-worker folder
COPY gpu-worker/handler.py /app/handler.py

# Run handler
CMD ["python", "-u", "handler.py"]
