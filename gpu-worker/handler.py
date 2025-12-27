"""
RunPod Serverless Handler for Hunyuan3D.

This handler runs on RunPod's serverless GPU infrastructure to generate
3D models from text prompts or images using Hunyuan3D.

Hunyuan3D-2 is an image-to-3D model. For text-to-3D, we first generate
an image using SDXL-Turbo, then convert that image to 3D.
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


# Global model references (loaded/unloaded dynamically to manage VRAM)
_shape_pipeline = None
_text2img_pipeline = None


def load_shape_pipeline():
    """Load Hunyuan3D model for image-to-3D."""
    global _shape_pipeline

    if _shape_pipeline is not None:
        return _shape_pipeline

    print("Loading Hunyuan3D for image-to-3D...")
    start_time = time.time()

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Hunyuan3D loaded in {time.time() - start_time:.1f}s")

    return _shape_pipeline


def unload_shape_pipeline():
    """Unload Hunyuan3D to free VRAM."""
    global _shape_pipeline
    if _shape_pipeline is not None:
        del _shape_pipeline
        _shape_pipeline = None
        torch.cuda.empty_cache()
        print("Hunyuan3D unloaded, VRAM freed")


def load_text2img_pipeline():
    """Load SDXL-Turbo for text-to-image."""
    global _text2img_pipeline

    if _text2img_pipeline is not None:
        return _text2img_pipeline

    print("Loading SDXL-Turbo for text-to-image...")
    start_time = time.time()

    from diffusers import AutoPipelineForText2Image
    _text2img_pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    _text2img_pipeline.to("cuda")
    print(f"SDXL-Turbo loaded in {time.time() - start_time:.1f}s")

    return _text2img_pipeline


def unload_text2img_pipeline():
    """Unload SDXL-Turbo to free VRAM."""
    global _text2img_pipeline
    if _text2img_pipeline is not None:
        del _text2img_pipeline
        _text2img_pipeline = None
        torch.cuda.empty_cache()
        print("SDXL-Turbo unloaded, VRAM freed")


def text_to_image(prompt: str, steps: int = 4) -> "Image":
    """Generate an image from text using SDXL-Turbo."""
    from PIL import Image

    # Unload Hunyuan3D first to free VRAM
    unload_shape_pipeline()

    text2img = load_text2img_pipeline()

    print(f"Generating image from prompt: {prompt[:50]}...")

    # SDXL-Turbo works best with 1-4 steps
    result = text2img(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,  # SDXL-Turbo doesn't need guidance
    )

    image = result.images[0]

    # Unload SDXL-Turbo to make room for Hunyuan3D
    unload_text2img_pipeline()

    return image


def generate_3d(
    prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
    octree_resolution: int = 256,
    face_count: int = 50000,
    output_format: str = "glb",
) -> bytes:
    """
    Generate a 3D model from text or image.

    For text-to-3D: First generates an image with SDXL-Turbo, then converts to 3D.
    For image-to-3D: Directly converts the provided image to 3D.

    Args:
        prompt: Text description (for text-to-3D via SDXL-Turbo)
        image_url: URL to image (for direct image-to-3D)
        image_base64: Base64 encoded image (for direct image-to-3D)
        steps: Number of inference steps for 3D generation
        guidance_scale: Guidance scale for 3D generation
        octree_resolution: Mesh octree resolution (128, 256, or 512)
        face_count: Target number of faces in output mesh
        output_format: Output format (glb, obj, ply)

    Returns:
        GLB file bytes
    """
    from PIL import Image
    import requests

    # Prepare input image
    input_image = None

    if image_base64:
        # Decode base64 image
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("Using provided base64 image")

    elif image_url:
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        input_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        print(f"Downloaded image from URL: {image_url[:50]}...")

    elif prompt:
        # Text-to-3D: First generate image from text
        print("Text-to-3D: Generating image from prompt first...")
        input_image = text_to_image(prompt)
        print("Image generated, now converting to 3D...")

    else:
        raise ValueError("Either prompt, image_url, or image_base64 required")

    # Generate 3D model from image
    print(f"Generating 3D model: steps={steps}, guidance={guidance_scale}, octree_resolution={octree_resolution}, face_count={face_count}")
    start_time = time.time()

    # Load Hunyuan3D (will be unloaded if text2img was used)
    shape_pipeline = load_shape_pipeline()

    result = shape_pipeline(
        image=input_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        octree_resolution=octree_resolution,
        target_face_count=face_count,
    )

    print(f"3D generation completed in {time.time() - start_time:.1f}s")

    # Extract mesh from result
    # Handle different return types from Hunyuan3D
    import trimesh as tm
    if isinstance(result, tm.Trimesh):
        mesh = result
    elif isinstance(result, list):
        # List[List[Trimesh]] or List[Trimesh]
        if len(result) > 0:
            if isinstance(result[0], list):
                mesh = result[0][0]
            else:
                mesh = result[0]
        else:
            raise ValueError("Empty result from pipeline")
    else:
        # Try to access as object attribute
        mesh = getattr(result, 'mesh', result)

    print(f"Mesh type: {type(mesh)}")

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
            "steps": 30,                   # 20-50, more = better quality
            "guidance_scale": 7.5,         # 5-10
            "octree_resolution": 256,      # 128, 256, or 512 (mesh density)
            "face_count": 50000,           # target face count
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
        octree_resolution = job_input.get("octree_resolution", 256)
        face_count = job_input.get("face_count", 50000)
        output_format = job_input.get("output_format", "glb")

        # Generate model
        start_time = time.time()

        glb_bytes = generate_3d(
            prompt=prompt,
            image_url=image_url,
            image_base64=image_base64,
            steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            face_count=face_count,
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


# Pre-load Hunyuan3D on container start (reduces cold start time)
# SDXL-Turbo is loaded on-demand only for text-to-3D requests
print("Initializing Hunyuan3D worker...")
try:
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    load_shape_pipeline()
    print("Worker ready!")
except Exception as e:
    print(f"Warning: Failed to pre-load model: {e}")
    print("Model will be loaded on first request")

# Start RunPod handler
runpod.serverless.start({"handler": handler})
