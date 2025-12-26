#!/bin/bash
set -e

echo "=== Hunyuan3D RunPod Worker Startup ==="

# Install dependencies
pip install --no-cache-dir runpod trimesh pygltflib huggingface_hub pillow requests

# Clone Hunyuan3D if not exists
if [ ! -d "/app/hunyuan3d" ]; then
    echo "Cloning Hunyuan3D..."
    git clone https://github.com/Tencent/Hunyuan3D-2.git /app/hunyuan3d
    cd /app/hunyuan3d
    pip install --no-cache-dir -r requirements.txt || true
fi

# Download handler from GitHub
echo "Downloading handler..."
curl -sL https://raw.githubusercontent.com/team42rows-prog/3D-AI-Print/main/gpu-worker/handler.py -o /app/handler.py

# Start handler
echo "Starting handler..."
cd /app
python -u handler.py
