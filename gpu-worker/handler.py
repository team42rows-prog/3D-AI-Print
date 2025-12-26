"""
RunPod Serverless Handler for Hunyuan3D.

This handler runs on RunPod's serverless GPU infrastructure to generate
3D models from text prompts or images using Hunyuan3D.
"""

import base64
import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import runpod
import torch

# Add Hunyuan3D to path
sys.path.insert(0, "/app/hunyuan3d")


# Global model reference (loaded once, reused across requests)
_pipeline = None
_model_loaded = False


def load_model():
    """Load Hunyuan3D model (called once on cold start)."""
    global _pipeline, _model_loaded

    if _model_loaded:
        return _pipeline

    print("Loading Hunyuan3D model...")
    start_time = time.time()

    try:
        # Import Hunyuan3D components
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        # Check for GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Load pipeline
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Model loaded successfully
        _model_loaded = True
        print(f"Model loaded in {time.time() - start_time:.1f}s")

        return _pipeline

    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise


def generate_3d(
    prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
    octree_depth: int = 8,
    output_format: str = "glb",
) -> bytes:
    """
    Generate a 3D model from text or image.

    Args:
        prompt: Text description (for text-to-3D)
        image_url: URL to image (for image-to-3D)
        image_base64: Base64 encoded image (for image-to-3D)
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        octree_depth: Mesh octree depth (higher = more detail)
        output_format: Output format (glb, obj, ply)

    Returns:
        GLB file bytes
    """
    from PIL import Image
    import requests

    pipeline = load_model()

    # Prepare input
    input_image = None

    if image_base64:
        # Decode base64 image
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    elif image_url:
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        input_image = Image.open(io.BytesIO(response.content)).convert("RGB")

    # Generate 3D model
    print(f"Generating 3D model: steps={steps}, guidance={guidance_scale}")
    start_time = time.time()

    if input_image:
        # Image-to-3D
        result = pipeline(
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_depth=octree_depth,
        )
    else:
        # Text-to-3D
        if not prompt:
            raise ValueError("Either prompt, image_url, or image_base64 required")

        result = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_depth=octree_depth,
        )

    print(f"Generation completed in {time.time() - start_time:.1f}s")

    # Export mesh
    mesh = result.mesh

    # Export to buffer
    buffer = io.BytesIO()

    if output_format == "glb":
        mesh.export(buffer, file_type="glb")
    elif output_format == "obj":
        mesh.export(buffer, file_type="obj")
    elif output_format == "ply":
        mesh.export(buffer, file_type="ply")
    else:
        mesh.export(buffer, file_type="glb")

    buffer.seek(0)
    return buffer.read()


def handler(job: dict) -> dict:
    """
    RunPod handler function.

    Input format:
    {
        "input": {
            "prompt": "a detailed dragon figurine",  # OR
            "image_url": "https://...",              # OR
            "image_base64": "data:image/png;base64,...",
            "steps": 30,
            "guidance_scale": 7.5,
            "octree_depth": 8,
            "output_format": "glb"
        }
    }

    Output format:
    {
        "glb_base64": "...",  # Base64 encoded GLB file
        "generation_time": 45.2,  # Seconds
        "vertices": 12345,
        "faces": 24680
    }
    """
    try:
        job_input = job.get("input", {})

        # Validate input
        prompt = job_input.get("prompt")
        image_url = job_input.get("image_url")
        image_base64 = job_input.get("image_base64")

        if not any([prompt, image_url, image_base64]):
            return {"error": "No input provided. Provide prompt, image_url, or image_base64"}

        # Generation parameters
        steps = job_input.get("steps", 30)
        guidance_scale = job_input.get("guidance_scale", 7.5)
        octree_depth = job_input.get("octree_depth", 8)
        output_format = job_input.get("output_format", "glb")

        # Generate model
        start_time = time.time()

        glb_bytes = generate_3d(
            prompt=prompt,
            image_url=image_url,
            image_base64=image_base64,
            steps=steps,
            guidance_scale=guidance_scale,
            octree_depth=octree_depth,
            output_format=output_format,
        )

        generation_time = time.time() - start_time

        # Get mesh info
        import trimesh
        mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")

        vertices = 0
        faces = 0
        if isinstance(mesh, trimesh.Scene):
            for geom in mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    vertices += len(geom.vertices)
                    faces += len(geom.faces)
        elif isinstance(mesh, trimesh.Trimesh):
            vertices = len(mesh.vertices)
            faces = len(mesh.faces)

        # Encode result
        glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")

        return {
            "glb_base64": glb_base64,
            "generation_time": round(generation_time, 2),
            "vertices": vertices,
            "faces": faces,
            "file_size_bytes": len(glb_bytes),
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


# Pre-load model on container start (reduces cold start time)
print("Initializing Hunyuan3D worker...")
try:
    load_model()
    print("Worker ready!")
except Exception as e:
    print(f"Warning: Failed to pre-load model: {e}")
    print("Model will be loaded on first request")

# Start RunPod handler
runpod.serverless.start({"handler": handler})
