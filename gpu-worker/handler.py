"""
RunPod Serverless Handler for Hunyuan3D-2.1 + Mesh Validation.

This handler runs on RunPod's serverless GPU infrastructure (H100) to:
1. Generate 3D models from text prompts or images using Hunyuan3D-2.1
2. Validate 3D meshes for printability (overhangs, watertight, etc.)

Hunyuan3D-2.1 is an image-to-3D model with PBR texture support.
For text-to-3D, we first generate an image using SDXL-Turbo, then convert to 3D.

Version: 3.1.0 - Added Cloudflare R2 upload for large files
"""

import base64
import io
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

import boto3
from botocore.config import Config
import numpy as np
import runpod
import torch


# =============================================================================
# Cloudflare R2 Storage Configuration
# =============================================================================

# Public R2 development URL
R2_PUBLIC_URL = "https://pub-db39d0352849406d821a66de8de2433f.r2.dev"


def get_r2_client():
    """Get boto3 client configured for Cloudflare R2."""
    endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL")
    access_key = os.environ.get("BUCKET_ACCESS_KEY_ID")
    secret_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY")

    if not all([endpoint_url, access_key, secret_key]):
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_to_r2(file_bytes: bytes, filename: str, content_type: str = "model/gltf-binary") -> Optional[str]:
    """
    Upload file to Cloudflare R2 and return public URL.

    Args:
        file_bytes: The file content as bytes
        filename: The filename to use in the bucket
        content_type: MIME type of the file

    Returns:
        Public URL to the file, or None if R2 not configured
    """
    client = get_r2_client()
    if not client:
        print("R2 not configured, falling back to base64")
        return None

    bucket_name = os.environ.get("BUCKET_NAME", "42rows-3d-print")

    # Generate unique key with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    object_key = f"models/{timestamp}_{unique_id}_{filename}"

    try:
        # Upload file
        client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=file_bytes,
            ContentType=content_type,
        )
        print(f"Uploaded {len(file_bytes)} bytes to R2: {object_key}")

        # Return public URL
        public_url = f"{R2_PUBLIC_URL}/{object_key}"
        print(f"Public URL: {public_url}")
        return public_url

    except Exception as e:
        print(f"R2 upload failed: {e}")
        return None

# Add Hunyuan3D-2.1 to path
sys.path.insert(0, "/app/hunyuan3d/hy3dshape")
sys.path.insert(0, "/app/hunyuan3d/hy3dpaint")
sys.path.insert(0, "/app/hunyuan3d")


# =============================================================================
# Printer Presets Database
# =============================================================================

PRINTER_PRESETS = {
    # === RESIN MSLA ===
    "anycubic_photon_m3": {
        "type": "resin",
        "build_mm": {"x": 180, "y": 164, "z": 102},
        "xy_um": 50,
        "layer_um": {"min": 10, "max": 150, "optimal": 50},
        "min_wall_mm": 0.3,
        "min_feature_mm": 0.15,
        "max_overhang_deg": 45,
    },
    "anycubic_photon_m5s": {
        "type": "resin",
        "build_mm": {"x": 218, "y": 123, "z": 200},
        "xy_um": 19,
        "layer_um": {"min": 10, "max": 150, "optimal": 50},
        "min_wall_mm": 0.25,
        "min_feature_mm": 0.1,
        "max_overhang_deg": 45,
    },
    "elegoo_mars_3": {
        "type": "resin",
        "build_mm": {"x": 143, "y": 90, "z": 175},
        "xy_um": 35,
        "layer_um": {"min": 10, "max": 200, "optimal": 50},
        "min_wall_mm": 0.25,
        "min_feature_mm": 0.1,
        "max_overhang_deg": 45,
    },
    "elegoo_saturn_3": {
        "type": "resin",
        "build_mm": {"x": 218, "y": 123, "z": 250},
        "xy_um": 19,
        "layer_um": {"min": 10, "max": 150, "optimal": 50},
        "min_wall_mm": 0.3,
        "min_feature_mm": 0.15,
        "max_overhang_deg": 45,
    },
    "formlabs_form_3": {
        "type": "resin",
        "build_mm": {"x": 145, "y": 145, "z": 185},
        "xy_um": 25,
        "layer_um": {"min": 25, "max": 300, "optimal": 100},
        "min_wall_mm": 0.2,
        "min_feature_mm": 0.1,
        "max_overhang_deg": 45,
    },
    "formlabs_form_4": {
        "type": "resin",
        "build_mm": {"x": 200, "y": 125, "z": 210},
        "xy_um": 50,
        "layer_um": {"min": 25, "max": 200, "optimal": 100},
        "min_wall_mm": 0.2,
        "min_feature_mm": 0.1,
        "max_overhang_deg": 45,
    },

    # === FDM ===
    "prusa_mk4": {
        "type": "fdm",
        "build_mm": {"x": 250, "y": 210, "z": 220},
        "nozzle_mm": 0.4,
        "layer_um": {"min": 50, "max": 300, "optimal": 200},
        "min_wall_mm": 0.8,
        "min_feature_mm": 0.4,
        "max_overhang_deg": 50,
    },
    "prusa_mini": {
        "type": "fdm",
        "build_mm": {"x": 180, "y": 180, "z": 180},
        "nozzle_mm": 0.4,
        "layer_um": {"min": 50, "max": 300, "optimal": 150},
        "min_wall_mm": 0.8,
        "min_feature_mm": 0.4,
        "max_overhang_deg": 50,
    },
    "bambu_x1c": {
        "type": "fdm",
        "build_mm": {"x": 256, "y": 256, "z": 256},
        "nozzle_mm": 0.4,
        "layer_um": {"min": 50, "max": 400, "optimal": 200},
        "min_wall_mm": 0.8,
        "min_feature_mm": 0.4,
        "max_overhang_deg": 55,
    },
    "bambu_a1_mini": {
        "type": "fdm",
        "build_mm": {"x": 180, "y": 180, "z": 180},
        "nozzle_mm": 0.4,
        "layer_um": {"min": 50, "max": 400, "optimal": 200},
        "min_wall_mm": 0.8,
        "min_feature_mm": 0.4,
        "max_overhang_deg": 55,
    },
    "creality_ender_3": {
        "type": "fdm",
        "build_mm": {"x": 220, "y": 220, "z": 250},
        "nozzle_mm": 0.4,
        "layer_um": {"min": 100, "max": 400, "optimal": 200},
        "min_wall_mm": 0.8,
        "min_feature_mm": 0.4,
        "max_overhang_deg": 45,
    },

    # === SLS ===
    "generic_sls": {
        "type": "sls",
        "build_mm": {"x": 160, "y": 160, "z": 320},
        "layer_um": {"min": 80, "max": 150, "optimal": 100},
        "min_wall_mm": 0.7,
        "min_feature_mm": 0.5,
        "max_overhang_deg": 90,  # SLS doesn't need supports
    },

    # === Generic Presets ===
    "generic_resin": {
        "type": "resin",
        "build_mm": {"x": 150, "y": 100, "z": 150},
        "xy_um": 50,
        "layer_um": {"min": 25, "max": 100, "optimal": 50},
        "min_wall_mm": 0.3,
        "min_feature_mm": 0.15,
        "max_overhang_deg": 45,
    },
    "generic_fdm": {
        "type": "fdm",
        "build_mm": {"x": 200, "y": 200, "z": 200},
        "nozzle_mm": 0.4,
        "layer_um": {"min": 100, "max": 400, "optimal": 200},
        "min_wall_mm": 0.8,
        "min_feature_mm": 0.4,
        "max_overhang_deg": 45,
    },
}


# =============================================================================
# Global Model References
# =============================================================================

_shape_pipeline = None
_text2img_pipeline = None


def load_shape_pipeline():
    """Load Hunyuan3D-2.1 model for image-to-3D (3.0B parameters, highest quality)."""
    global _shape_pipeline

    if _shape_pipeline is not None:
        return _shape_pipeline

    print("Loading Hunyuan3D-2.1 (3.0B model) for image-to-3D...")
    start_time = time.time()

    # Import from hy3dshape (Hunyuan3D-2.1 library)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    # Load Hunyuan3D-2.1 - the latest 3.0B parameter model with PBR support
    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        device="cuda",
        dtype=torch.float16,
    )
    print(f"Hunyuan3D-2.1 loaded in {time.time() - start_time:.1f}s")

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


# =============================================================================
# 3D Generation Functions
# =============================================================================

def text_to_image(prompt: str, steps: int = 4) -> "Image":
    """Generate an image from text using SDXL-Turbo."""
    from PIL import Image

    # With H100 80GB we don't need to unload models
    text2img = load_text2img_pipeline()

    print(f"Generating image from prompt: {prompt[:50]}...")

    # SDXL-Turbo works best with 1-4 steps
    result = text2img(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,  # SDXL-Turbo doesn't need guidance
    )

    image = result.images[0]
    return image


def generate_3d(
    prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    steps: int = 100,
    guidance_scale: float = 7.5,
    dual_guidance_scale: float = 10.5,
    octree_resolution: int = 512,
    num_chunks: int = 12000,
    output_format: str = "glb",
) -> bytes:
    """
    Generate a 3D model from text or image using Hunyuan3D-2.1.

    For text-to-3D: First generates an image with SDXL-Turbo, then converts to 3D.
    For image-to-3D: Directly converts the provided image to 3D.

    H100 optimized parameters for MAXIMUM quality:
    - steps: 100 (maximum detail, more iterations)
    - guidance_scale: 7.5 (balanced adherence to input)
    - dual_guidance_scale: 10.5 (enhanced dual guidance)
    - octree_resolution: 512 (maximum geometric detail)
    - num_chunks: 12000 (higher processing chunks for detail)
    """
    from PIL import Image
    import requests

    # Prepare input image
    input_image = None

    if image_base64:
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("Using provided base64 image")

    elif image_url:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        input_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        print(f"Downloaded image from URL: {image_url[:50]}...")

    elif prompt:
        print("Text-to-3D: Generating image from prompt first...")
        input_image = text_to_image(prompt)
        print("Image generated, now converting to 3D...")

    else:
        raise ValueError("Either prompt, image_url, or image_base64 required")

    # Generate 3D model from image
    print(f"Generating 3D model: steps={steps}, guidance={guidance_scale}, octree={octree_resolution}")
    start_time = time.time()

    shape_pipeline = load_shape_pipeline()

    # Call pipeline with H100 optimized parameters
    result = shape_pipeline(
        image=input_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        dual_guidance_scale=dual_guidance_scale,
        dual_guidance=True,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type="trimesh",
    )

    print(f"3D generation completed in {time.time() - start_time:.1f}s")

    # Extract mesh from result
    import trimesh as tm
    if isinstance(result, tm.Trimesh):
        mesh = result
    elif isinstance(result, list):
        if len(result) > 0:
            if isinstance(result[0], list):
                mesh = result[0][0]
            else:
                mesh = result[0]
        else:
            raise ValueError("Empty result from pipeline")
    elif hasattr(result, 'mesh'):
        mesh = result.mesh
    else:
        mesh = result

    # Export to buffer
    buffer = io.BytesIO()
    mesh.export(buffer, file_type=output_format)
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# Mesh Validation Functions
# =============================================================================

def detect_mesh_file_type(mesh_bytes: bytes) -> Optional[str]:
    """
    Detect mesh file type from magic bytes.

    Returns file type string for trimesh (stl, glb, obj, ply) or None if unknown.
    """
    if len(mesh_bytes) < 10:
        return None

    # GLB/GLTF binary: starts with "glTF" magic
    if mesh_bytes[:4] == b'glTF':
        return 'glb'

    # STL binary: usually starts with header, but check for "solid" (ASCII STL)
    # ASCII STL starts with "solid "
    if mesh_bytes[:6].lower() == b'solid ':
        return 'stl'

    # PLY: starts with "ply"
    if mesh_bytes[:3].lower() == b'ply':
        return 'ply'

    # OBJ: typically starts with comments (#) or vertex (v )
    first_lines = mesh_bytes[:200].decode('utf-8', errors='ignore').lower()
    if first_lines.startswith('#') or '\nv ' in first_lines or first_lines.startswith('v '):
        return 'obj'

    # Default to GLB since that's what Hunyuan3D outputs
    return 'glb'


def detect_overhangs(mesh, threshold_deg: float = 45.0) -> Dict[str, Any]:
    """
    Detect overhang faces using face normals and ray casting.

    Returns info about overhangs: percentage, max angle, needs supports.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        return {"error": "Not a valid trimesh"}

    # Get face normals
    face_normals = mesh.face_normals

    # Calculate angle from vertical (Z-up)
    z_component = face_normals[:, 2]

    # Angle from horizontal plane
    angles_from_horizontal = np.degrees(np.arcsin(np.clip(-z_component, -1, 1)))

    # Overhangs are faces pointing downward at angle > threshold from vertical
    overhang_mask = angles_from_horizontal > threshold_deg
    overhang_count = np.sum(overhang_mask)
    total_faces = len(face_normals)

    overhang_percentage = (overhang_count / total_faces * 100) if total_faces > 0 else 0
    max_overhang_angle = float(np.max(angles_from_horizontal)) if len(angles_from_horizontal) > 0 else 0

    # Classify severity
    if overhang_percentage > 30:
        severity = "critical"
        needs_supports = True
    elif overhang_percentage > 10:
        severity = "warning"
        needs_supports = True
    elif overhang_percentage > 0:
        severity = "info"
        needs_supports = False
    else:
        severity = "ok"
        needs_supports = False

    return {
        "overhang_percentage": round(overhang_percentage, 2),
        "overhang_face_count": int(overhang_count),
        "total_faces": int(total_faces),
        "max_angle_deg": round(max_overhang_angle, 1),
        "threshold_deg": threshold_deg,
        "needs_supports": needs_supports,
        "severity": severity,
    }


def check_mesh_quality(mesh) -> List[Dict[str, Any]]:
    """
    Run comprehensive mesh quality checks.

    Returns list of issues found.
    """
    import trimesh

    issues = []

    if not isinstance(mesh, trimesh.Trimesh):
        issues.append({
            "check": "valid_mesh",
            "passed": False,
            "severity": "critical",
            "message": "Invalid mesh format",
        })
        return issues

    # 1. Watertight check
    is_watertight = mesh.is_watertight
    issues.append({
        "check": "watertight",
        "passed": is_watertight,
        "severity": "critical" if not is_watertight else "ok",
        "message": "Mesh is watertight" if is_watertight else "Mesh has holes - not suitable for printing",
    })

    # 2. Winding consistency (normals)
    try:
        is_winding_consistent = mesh.is_winding_consistent
    except:
        is_winding_consistent = True

    issues.append({
        "check": "normals_consistent",
        "passed": is_winding_consistent,
        "severity": "critical" if not is_winding_consistent else "ok",
        "message": "Normals are consistent" if is_winding_consistent else "Inconsistent normals - may print inside-out",
    })

    # 3. Valid volume
    try:
        is_volume = mesh.is_volume
    except:
        is_volume = mesh.is_watertight

    issues.append({
        "check": "valid_volume",
        "passed": is_volume,
        "severity": "critical" if not is_volume else "ok",
        "message": "Valid 3D volume" if is_volume else "Not a valid closed volume",
    })

    # 4. Degenerate faces (zero area)
    face_areas = mesh.area_faces
    degenerate_count = np.sum(face_areas < 1e-10)
    has_degenerate = degenerate_count > 0

    issues.append({
        "check": "no_degenerate_faces",
        "passed": not has_degenerate,
        "severity": "warning" if has_degenerate else "ok",
        "message": f"{degenerate_count} degenerate faces found" if has_degenerate else "No degenerate faces",
        "count": int(degenerate_count),
    })

    # 5. Mesh complexity
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)

    if face_count > 500000:
        complexity_severity = "warning"
        complexity_msg = f"High polygon count ({face_count:,}) - may slow slicing"
    elif face_count < 100:
        complexity_severity = "warning"
        complexity_msg = f"Very low polygon count ({face_count}) - may lack detail"
    else:
        complexity_severity = "ok"
        complexity_msg = f"Reasonable polygon count ({face_count:,})"

    issues.append({
        "check": "mesh_complexity",
        "passed": complexity_severity == "ok",
        "severity": complexity_severity,
        "message": complexity_msg,
        "vertices": int(vertex_count),
        "faces": int(face_count),
    })

    return issues


def check_print_dimensions(
    mesh,
    build_volume: Dict[str, float],
    target_size_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Check if mesh fits in build volume.

    Returns dimension info and fit status.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        return {"error": "Invalid mesh"}

    # Get mesh bounds
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]

    mesh_dims = {
        "x": float(dimensions[0]),
        "y": float(dimensions[1]),
        "z": float(dimensions[2]),
    }

    # If target_size specified, calculate what dimensions would be after scaling
    if target_size_mm:
        current_max = max(dimensions)
        scale_factor = target_size_mm / current_max if current_max > 0 else 1
        mesh_dims = {k: v * scale_factor for k, v in mesh_dims.items()}

    # Check fit
    fits_x = mesh_dims["x"] <= build_volume["x"]
    fits_y = mesh_dims["y"] <= build_volume["y"]
    fits_z = mesh_dims["z"] <= build_volume["z"]
    fits = fits_x and fits_y and fits_z

    # Calculate percentage of build volume used
    volume_usage = {
        "x": round(mesh_dims["x"] / build_volume["x"] * 100, 1),
        "y": round(mesh_dims["y"] / build_volume["y"] * 100, 1),
        "z": round(mesh_dims["z"] / build_volume["z"] * 100, 1),
    }

    return {
        "fits_build_volume": fits,
        "mesh_dimensions_mm": {k: round(v, 2) for k, v in mesh_dims.items()},
        "build_volume_mm": build_volume,
        "volume_usage_percent": volume_usage,
        "scaled": target_size_mm is not None,
        "target_size_mm": target_size_mm,
    }


def validate_mesh(
    mesh_base64: str,
    printer: str = "generic_resin",
    target_size_mm: Optional[float] = None,
    check_overhangs: bool = True,
    auto_repair: bool = False,
    custom_build_volume: Optional[Dict[str, float]] = None,
    file_type: Optional[str] = None,
    output_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive mesh validation for 3D printing.

    Args:
        output_format: Format for repaired mesh output. If None, uses input file_type.
    """
    import trimesh

    start_time = time.time()

    # Get printer settings
    if printer in PRINTER_PRESETS:
        printer_settings = PRINTER_PRESETS[printer].copy()
    else:
        printer_settings = PRINTER_PRESETS["generic_resin"].copy()
        printer = "generic_resin"

    # Override with custom build volume if provided
    if custom_build_volume:
        printer_settings["build_mm"] = custom_build_volume

    # Decode mesh
    try:
        if mesh_base64.startswith("data:"):
            data_header = mesh_base64.split(",", 0)[0] if "," in mesh_base64 else ""
            if not file_type and "model/" in data_header:
                mime_part = data_header.split(";")[0].replace("data:", "")
                if "/" in mime_part:
                    file_type = mime_part.split("/")[1]
            mesh_base64 = mesh_base64.split(",", 1)[1]
        mesh_bytes = base64.b64decode(mesh_base64)
    except Exception as e:
        return {"error": f"Failed to decode mesh: {e}"}

    # Try to detect file type from magic bytes if not provided
    if not file_type:
        file_type = detect_mesh_file_type(mesh_bytes)
        print(f"Auto-detected file type: {file_type}")

    # Load mesh with trimesh
    try:
        loaded = trimesh.load(io.BytesIO(mesh_bytes), file_type=file_type)

        # Handle scene (multiple meshes)
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            else:
                return {"error": "No valid geometry in file"}
        else:
            mesh = loaded
    except Exception as e:
        return {"error": f"Failed to load mesh: {e}"}

    # Scale if target size specified
    if target_size_mm and hasattr(mesh, 'bounds'):
        bounds = mesh.bounds
        current_max = max(bounds[1] - bounds[0])
        if current_max > 0:
            scale_factor = target_size_mm / current_max
            mesh.apply_scale(scale_factor)
            print(f"Scaled mesh by {scale_factor:.4f}x to target {target_size_mm}mm")

    # Run validation checks
    issues = check_mesh_quality(mesh)

    # Check dimensions
    dim_check = check_print_dimensions(
        mesh,
        printer_settings["build_mm"],
        target_size_mm=None,
    )

    if not dim_check.get("fits_build_volume", True):
        issues.append({
            "check": "fits_build_volume",
            "passed": False,
            "severity": "critical",
            "message": f"Model too large for {printer} build volume",
            "details": dim_check,
        })
    else:
        issues.append({
            "check": "fits_build_volume",
            "passed": True,
            "severity": "ok",
            "message": "Model fits in build volume",
            "details": dim_check,
        })

    # Check overhangs
    overhang_info = None
    if check_overhangs:
        overhang_info = detect_overhangs(
            mesh,
            threshold_deg=printer_settings.get("max_overhang_deg", 45),
        )

        if overhang_info.get("needs_supports"):
            issues.append({
                "check": "overhangs",
                "passed": False,
                "severity": overhang_info["severity"],
                "message": f"{overhang_info['overhang_percentage']:.1f}% of faces are overhangs (>{overhang_info['threshold_deg']}°)",
                "details": overhang_info,
            })
        else:
            issues.append({
                "check": "overhangs",
                "passed": True,
                "severity": "ok",
                "message": "No significant overhangs detected",
                "details": overhang_info,
            })

    # Auto repair if requested
    repairs_made = []
    repair_stats = {}
    repaired_mesh_base64 = None
    repaired_mesh_url = None
    # Determine output format for repaired mesh: requested > input > default glb
    export_format = output_format or file_type or "glb"
    if export_format == "gltf":
        export_format = "glb"  # trimesh exports gltf as glb

    if auto_repair:
        try:
            initial_watertight = mesh.is_watertight
            initial_vertices = len(mesh.vertices)
            initial_faces = len(mesh.faces)

            # Try PyMeshFix first
            pymeshfix_success = False
            try:
                import pymeshfix

                print("Using PyMeshFix for advanced mesh repair...")

                tin = pymeshfix.PyTMesh(verbose=False)
                tin.load_array(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))

                initial_boundaries = tin.boundaries()
                repair_stats["initial_holes"] = initial_boundaries

                try:
                    tin.join_closest_components()
                    repairs_made.append("Joined nearby components")
                except Exception as e:
                    print(f"join_closest_components skipped: {e}")

                try:
                    holes_filled = tin.fill_small_boundaries(nbe=0, refine=True)
                    if holes_filled > 0:
                        repairs_made.append(f"Filled {holes_filled} holes")
                        repair_stats["holes_filled"] = holes_filled
                except Exception as e:
                    print(f"fill_small_boundaries error: {e}")

                try:
                    tin.clean(max_iters=10, inner_loops=3)
                    repairs_made.append("Removed self-intersections")
                except Exception as e:
                    print(f"clean() error: {e}")

                try:
                    tin.remove_smallest_components()
                    repairs_made.append("Removed debris components")
                except Exception as e:
                    print(f"remove_smallest_components skipped: {e}")

                final_boundaries = tin.boundaries()
                repair_stats["final_holes"] = final_boundaries
                vclean, fclean = tin.return_arrays()

                mesh = trimesh.Trimesh(vertices=vclean, faces=fclean)
                mesh.fix_normals()
                repairs_made.append("Fixed normals")

                pymeshfix_success = True
                print(f"PyMeshFix repair complete: {initial_boundaries} -> {final_boundaries} holes")

            except ImportError:
                print("PyMeshFix not available, falling back to trimesh repair")
            except Exception as e:
                print(f"PyMeshFix repair failed: {e}, falling back to trimesh")

            # Fallback to trimesh basic repair if PyMeshFix failed
            if not pymeshfix_success:
                print("Using trimesh basic repair...")

                if not mesh.is_watertight:
                    trimesh.repair.fill_holes(mesh)
                    if mesh.is_watertight:
                        repairs_made.append("Filled holes (trimesh)")

                if hasattr(mesh, 'fix_normals'):
                    mesh.fix_normals()
                    repairs_made.append("Fixed normals")

                try:
                    mesh.update_faces(mesh.nondegenerate_faces())
                    repairs_made.append("Removed degenerate faces")
                except Exception:
                    pass

            final_watertight = mesh.is_watertight
            final_vertices = len(mesh.vertices)
            final_faces = len(mesh.faces)

            repair_stats["initial_watertight"] = initial_watertight
            repair_stats["final_watertight"] = final_watertight
            repair_stats["vertices_before"] = initial_vertices
            repair_stats["vertices_after"] = final_vertices
            repair_stats["faces_before"] = initial_faces
            repair_stats["faces_after"] = final_faces
            repair_stats["pymeshfix_used"] = pymeshfix_success

            if repairs_made:
                buffer = io.BytesIO()
                mesh.export(buffer, file_type=export_format)
                repaired_bytes = buffer.getvalue()
                print(f"Exported repaired mesh as {export_format} ({len(repaired_bytes)} bytes)")

                # Upload repaired mesh to R2 (same as generation)
                repaired_filename = f"repaired_{time.strftime('%Y%m%d_%H%M%S')}.{export_format}"
                repaired_mesh_url = upload_to_r2(repaired_bytes, repaired_filename)

                if repaired_mesh_url:
                    print(f"Repaired mesh uploaded to R2: {repaired_mesh_url}")
                else:
                    # Fallback to base64 only for small files
                    if len(repaired_bytes) < 10 * 1024 * 1024:  # < 10MB
                        repaired_mesh_base64 = base64.b64encode(repaired_bytes).decode("utf-8")
                        print(f"Repaired mesh returned as base64 ({len(repaired_bytes)} bytes)")
                    else:
                        print(f"WARNING: Repaired mesh too large ({len(repaired_bytes)} bytes) and R2 upload failed")

                print(f"Repair complete: watertight {initial_watertight} -> {final_watertight}")

        except Exception as e:
            repairs_made.append(f"Repair failed: {e}")
            repair_stats["error"] = str(e)

    # Calculate summary
    critical_issues = [i for i in issues if i.get("severity") == "critical" and not i.get("passed")]
    is_printable = len(critical_issues) == 0

    # Generate recommendations
    recommendations = []

    if not is_printable:
        if any(i["check"] == "watertight" for i in critical_issues):
            recommendations.append("Use mesh repair software (e.g., Meshmixer) to fix holes")
        if any(i["check"] == "fits_build_volume" for i in critical_issues):
            max_dim = max(dim_check["mesh_dimensions_mm"].values())
            build_max = min(printer_settings["build_mm"].values())
            recommended_size = build_max * 0.9
            recommendations.append(f"Scale model down to {recommended_size:.0f}mm or smaller")

    if overhang_info and overhang_info.get("needs_supports"):
        if printer_settings["type"] == "resin":
            recommendations.append("Add supports in slicer software (auto-supports usually work well)")
            recommendations.append("Consider tilting model 15-30° to reduce overhangs")
        else:
            recommendations.append("Enable supports in slicer (tree supports work well for organic shapes)")
            recommendations.append("Orient largest flat surface on build plate")

    validation_time = time.time() - start_time

    issue_strings = [i["message"] for i in issues if i.get("severity") == "critical" and not i.get("passed")]
    warning_strings = [i["message"] for i in issues if i.get("severity") == "warning" and not i.get("passed")]

    is_watertight = mesh.is_watertight if hasattr(mesh, 'is_watertight') else False
    has_consistent_normals = mesh.is_winding_consistent if hasattr(mesh, 'is_winding_consistent') else True
    has_positive_volume = mesh.is_volume if hasattr(mesh, 'is_volume') else is_watertight

    volume_cm3 = None
    if is_watertight and hasattr(mesh, 'volume'):
        try:
            volume_cm3 = float(mesh.volume) / 1000
        except:
            pass

    overhang_response = None
    if overhang_info:
        overhang_response = {
            "total_faces": overhang_info.get("total_faces", 0),
            "overhang_faces": overhang_info.get("overhang_face_count", 0),
            "overhang_percentage": overhang_info.get("overhang_percentage", 0),
            "max_overhang_angle": overhang_info.get("max_angle_deg", 0),
            "needs_supports": overhang_info.get("needs_supports", False),
        }

    return {
        "success": True,
        "valid": is_printable,
        "printer": printer,
        "printer_type": printer_settings["type"],
        "vertices": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
        "faces": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        "dimensions_mm": dim_check.get("mesh_dimensions_mm"),
        "volume_cm3": round(volume_cm3, 3) if volume_cm3 else None,
        "is_watertight": is_watertight,
        "has_consistent_normals": has_consistent_normals,
        "has_positive_volume": has_positive_volume,
        "fits_build_volume": dim_check.get("fits_build_volume", False),
        "issues": issue_strings,
        "warnings": warning_strings,
        "recommendations": recommendations,
        "overhangs": overhang_response,
        "repaired": len(repairs_made) > 0 and (repaired_mesh_url is not None or repaired_mesh_base64 is not None),
        "repairs_made": repairs_made,
        "repair_stats": repair_stats if repair_stats else None,
        "repaired_mesh_url": repaired_mesh_url,  # R2 URL (preferred for large files)
        "repaired_mesh_base64": repaired_mesh_base64,  # Fallback for small files
        "repaired_mesh_format": export_format if (repaired_mesh_url or repaired_mesh_base64) else None,
        "target_size_mm": target_size_mm,
        "validation_time_seconds": round(validation_time, 2),
    }


# =============================================================================
# Handler Functions
# =============================================================================

def handle_generate(job_input: dict, job_id: str = None) -> dict:
    """Handle 3D generation request."""
    prompt = job_input.get("prompt")
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")

    if not any([prompt, image_url, image_base64]):
        return {"error": "No input provided. Provide prompt, image_url, or image_base64"}

    # H100 optimized defaults - MAXIMUM QUALITY
    steps = job_input.get("steps", 100)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    dual_guidance_scale = job_input.get("dual_guidance_scale", 10.5)
    octree_resolution = job_input.get("octree_resolution", 512)
    num_chunks = job_input.get("num_chunks", 12000)
    output_format = job_input.get("output_format", "glb")

    start_time = time.time()

    glb_bytes = generate_3d(
        prompt=prompt,
        image_url=image_url,
        image_base64=image_base64,
        steps=steps,
        guidance_scale=guidance_scale,
        dual_guidance_scale=dual_guidance_scale,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
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

    # Try to upload to R2 first (for large files)
    file_size = len(glb_bytes)
    filename = f"{job_id or 'model'}.glb"

    glb_url = upload_to_r2(glb_bytes, filename)

    result = {
        "generation_time": round(generation_time, 2),
        "vertices": vertices,
        "faces": faces,
        "file_size_bytes": file_size,
    }

    if glb_url:
        # R2 upload succeeded - return URL
        result["glb_url"] = glb_url
        result["storage"] = "r2"
        print(f"Returning R2 URL for {file_size} byte file")
    else:
        # Fallback to base64 (for small files or if R2 not configured)
        result["glb_base64"] = base64.b64encode(glb_bytes).decode("utf-8")
        result["storage"] = "base64"
        print(f"Returning base64 for {file_size} byte file")

    return result


def handle_validate(job_input: dict) -> dict:
    """Handle mesh validation request."""
    import requests
    from urllib.parse import urlparse

    mesh_url = job_input.get("mesh_url")
    mesh_base64 = job_input.get("mesh_base64")
    file_type = job_input.get("file_type")

    if not mesh_url and not mesh_base64:
        return {"error": "Either mesh_url or mesh_base64 must be provided"}

    if mesh_url:
        try:
            if not file_type:
                parsed = urlparse(mesh_url)
                path = parsed.path.lower()
                path = path.split('?')[0]
                if path.endswith('.glb'):
                    file_type = 'glb'
                elif path.endswith('.stl'):
                    file_type = 'stl'
                elif path.endswith('.obj'):
                    file_type = 'obj'
                elif path.endswith('.ply'):
                    file_type = 'ply'
                elif path.endswith('.gltf'):
                    file_type = 'gltf'
                print(f"Detected file_type from URL: {file_type}")

            print(f"Downloading mesh from URL: {mesh_url[:100]}...")
            # Increased timeout: 300s for large files (132MB+ needs more time)
            # Using stream=True for memory efficiency with large files
            response = requests.get(mesh_url, timeout=300, stream=True)
            response.raise_for_status()

            # Read in chunks to handle large files efficiently
            chunks = []
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    total_size += len(chunk)
                    # Log progress for large files
                    if total_size % (10 * 1024 * 1024) == 0:  # Every 10MB
                        print(f"Downloaded {total_size / 1024 / 1024:.1f}MB...")

            mesh_bytes = b''.join(chunks)
            mesh_base64 = base64.b64encode(mesh_bytes).decode('utf-8')
            print(f"Downloaded {len(mesh_bytes)} bytes ({len(mesh_bytes) / 1024 / 1024:.1f}MB) from URL")
        except requests.Timeout as e:
            return {
                "error": f"Timeout downloading mesh (300s exceeded for large file): {str(e)}",
                "success": False,
                "valid": False,
                "vertices": 0,
                "faces": 0,
            }
        except requests.RequestException as e:
            return {
                "error": f"Failed to download mesh from URL: {str(e)}",
                "success": False,
                "valid": False,
                "vertices": 0,
                "faces": 0,
            }

    return validate_mesh(
        mesh_base64=mesh_base64,
        printer=job_input.get("printer", "generic_resin"),
        target_size_mm=job_input.get("target_size_mm"),
        check_overhangs=job_input.get("check_overhangs", True),
        auto_repair=job_input.get("auto_repair", False),
        custom_build_volume=job_input.get("build_volume_mm"),
        file_type=file_type,
        output_format=job_input.get("output_format"),  # If specified, override output format
    )


# =============================================================================
# Render Preview Functions
# =============================================================================

def render_mesh_to_images(
    mesh,
    views: List[str] = ["front", "left", "back", "right"],
    resolution: int = 512,
    background_color: List[float] = [1.0, 1.0, 1.0, 1.0],
) -> Dict[str, bytes]:
    """
    Render a mesh to PNG images from multiple views using pyrender.

    Args:
        mesh: trimesh.Trimesh object
        views: List of view names (front, left, back, right, top, isometric)
        resolution: Image resolution (square)
        background_color: RGBA background color

    Returns:
        Dict mapping view name to PNG bytes
    """
    import pyrender
    from PIL import Image

    # Create pyrender mesh from trimesh
    # Apply a nice grey material for preview
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 1.0],
        metallicFactor=0.2,
        roughnessFactor=0.6,
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    # Create scene
    scene = pyrender.Scene(bg_color=background_color, ambient_light=[0.3, 0.3, 0.3])
    scene.add(pr_mesh)

    # Calculate camera distance based on mesh bounds
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.max(bounds[1] - bounds[0])
    camera_distance = size * 2.0

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)

    # Add lights
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)

    # View camera positions (looking at center)
    view_configs = {
        "front": {"angle": 0, "elevation": 0},
        "left": {"angle": 90, "elevation": 0},
        "back": {"angle": 180, "elevation": 0},
        "right": {"angle": 270, "elevation": 0},
        "top": {"angle": 0, "elevation": 90},
        "isometric": {"angle": 45, "elevation": 35},
    }

    # Create offscreen renderer
    renderer = pyrender.OffscreenRenderer(resolution, resolution)

    rendered_images = {}

    try:
        for view_name in views:
            if view_name not in view_configs:
                continue

            config = view_configs[view_name]
            angle_rad = np.radians(config["angle"])
            elev_rad = np.radians(config["elevation"])

            # Calculate camera position
            cam_x = camera_distance * np.cos(elev_rad) * np.sin(angle_rad) + center[0]
            cam_y = camera_distance * np.cos(elev_rad) * np.cos(angle_rad) + center[1]
            cam_z = camera_distance * np.sin(elev_rad) + center[2]

            # Camera looks at center
            camera_pos = np.array([cam_x, cam_y, cam_z])

            # Build camera transform matrix (look at center)
            forward = center - camera_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, [0, 0, 1])
            if np.linalg.norm(right) < 0.001:
                right = np.cross(forward, [0, 1, 0])
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            camera_pose = np.eye(4)
            camera_pose[:3, 0] = right
            camera_pose[:3, 1] = up
            camera_pose[:3, 2] = -forward
            camera_pose[:3, 3] = camera_pos

            # Add camera and light to scene
            cam_node = scene.add(camera, pose=camera_pose)
            light_node = scene.add(light, pose=camera_pose)

            # Render
            color, _ = renderer.render(scene)

            # Remove nodes for next view
            scene.remove_node(cam_node)
            scene.remove_node(light_node)

            # Convert to PNG bytes
            img = Image.fromarray(color)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            rendered_images[view_name] = buffer.getvalue()

            print(f"Rendered {view_name} view: {len(rendered_images[view_name])} bytes")

    finally:
        renderer.delete()

    return rendered_images


def handle_render_preview(job_input: dict) -> dict:
    """
    Handle mesh rendering preview request.

    Renders a 3D mesh to PNG images from multiple views (front, left, back, right).
    Uses a neutral grey material with studio lighting - no textures.

    Input:
    {
        "action": "render_preview",
        "mesh_url": "https://...",           # OR
        "mesh_base64": "...",                # Base64 encoded mesh
        "views": ["front", "left", "back", "right"],  # Optional, default all 4
        "resolution": 512,                   # Optional, default 512
        "file_type": "glb"                   # Optional, auto-detect
    }

    Returns:
    {
        "success": True,
        "images": {
            "front": "https://r2.../front.png",  # or base64
            "left": "https://r2.../left.png",
            ...
        },
        "render_time_seconds": 1.5
    }
    """
    import trimesh
    import requests
    from urllib.parse import urlparse

    start_time = time.time()

    mesh_url = job_input.get("mesh_url")
    mesh_base64 = job_input.get("mesh_base64")
    views = job_input.get("views", ["front", "left", "back", "right"])
    resolution = job_input.get("resolution", 512)
    file_type = job_input.get("file_type")

    if not mesh_url and not mesh_base64:
        return {"error": "Either mesh_url or mesh_base64 must be provided", "success": False}

    # Load mesh
    try:
        if mesh_url:
            # Detect file type from URL
            if not file_type:
                parsed = urlparse(mesh_url)
                path = parsed.path.lower().split('?')[0]
                if path.endswith('.glb'):
                    file_type = 'glb'
                elif path.endswith('.stl'):
                    file_type = 'stl'
                elif path.endswith('.obj'):
                    file_type = 'obj'
                elif path.endswith('.ply'):
                    file_type = 'ply'
                else:
                    file_type = 'glb'  # Default

            print(f"Downloading mesh from URL: {mesh_url[:100]}...")
            response = requests.get(mesh_url, timeout=120)
            response.raise_for_status()
            mesh_bytes = response.content
            print(f"Downloaded {len(mesh_bytes)} bytes")
        else:
            if mesh_base64.startswith("data:"):
                mesh_base64 = mesh_base64.split(",", 1)[1]
            mesh_bytes = base64.b64decode(mesh_base64)
            if not file_type:
                file_type = detect_mesh_file_type(mesh_bytes) or 'glb'

        # Load with trimesh
        loaded = trimesh.load(io.BytesIO(mesh_bytes), file_type=file_type)

        # Handle scene (multiple meshes)
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            else:
                return {"error": "No valid geometry in file", "success": False}
        else:
            mesh = loaded

        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    except Exception as e:
        return {"error": f"Failed to load mesh: {e}", "success": False}

    # Render views
    try:
        rendered_images = render_mesh_to_images(
            mesh,
            views=views,
            resolution=resolution,
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Failed to render: {e}", "success": False}

    render_time = time.time() - start_time

    # Upload images to R2 or return as base64
    result_images = {}
    for view_name, png_bytes in rendered_images.items():
        filename = f"preview_{view_name}_{uuid.uuid4().hex[:8]}.png"
        url = upload_to_r2(png_bytes, filename, content_type="image/png")

        if url:
            result_images[view_name] = {"url": url}
        else:
            # Fallback to base64
            result_images[view_name] = {
                "base64": base64.b64encode(png_bytes).decode("utf-8")
            }

    return {
        "success": True,
        "images": result_images,
        "views_rendered": list(rendered_images.keys()),
        "resolution": resolution,
        "mesh_vertices": len(mesh.vertices),
        "mesh_faces": len(mesh.faces),
        "render_time_seconds": round(render_time, 2),
    }


# =============================================================================
# Paint 2.1 Functions (PBR Texture Generation)
# =============================================================================

_paint_pipeline = None


def load_paint_pipeline(max_num_view: int = 6, resolution: int = 512):
    """Load Hunyuan3D Paint 2.1 for PBR texture generation."""
    global _paint_pipeline

    # Check if already loaded with same config
    if _paint_pipeline is not None:
        return _paint_pipeline

    print(f"Loading Hunyuan3D Paint 2.1 (max_view={max_num_view}, res={resolution})...")
    start_time = time.time()

    # Add hy3dpaint to path
    sys.path.insert(0, "/app/hunyuan3d/hy3dpaint")

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    config = Hunyuan3DPaintConfig(
        max_num_view=max_num_view,
        resolution=resolution
    )
    _paint_pipeline = Hunyuan3DPaintPipeline(config)

    print(f"Hunyuan3D Paint 2.1 loaded in {time.time() - start_time:.1f}s")
    return _paint_pipeline


def handle_render_painted(job_input: dict, job_id: str = "unknown") -> dict:
    """
    Apply PBR textures to a mesh and render preview images.

    This generates a textured mesh using Hunyuan3D Paint 2.1, then renders
    it to PNG images to show how the model would look when painted.

    Input:
    {
        "action": "render_painted",
        "mesh_url": "https://...",           # OR
        "mesh_base64": "...",                # Base64 encoded mesh (GLB)
        "image_url": "https://...",          # Reference image for texturing
        "image_base64": "...",               # OR base64 reference image
        "views": ["front", "left", "back", "right"],  # Views to render
        "resolution": 512,                   # Render resolution
        "paint_resolution": 512,             # Paint texture resolution (512 or 768)
        "max_views": 6                       # Paint generation views (6-9)
    }

    Returns:
    {
        "success": True,
        "textured_mesh_url": "https://r2.../textured.glb",
        "images": {
            "front": {"url": "https://r2.../front.png"},
            ...
        },
        "paint_time_seconds": 45.0,
        "render_time_seconds": 1.5
    }
    """
    import trimesh
    import requests
    import tempfile
    from PIL import Image

    mesh_url = job_input.get("mesh_url")
    mesh_base64 = job_input.get("mesh_base64")
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")
    views = job_input.get("views", ["front", "left", "back", "right"])
    render_resolution = job_input.get("resolution", 512)
    paint_resolution = job_input.get("paint_resolution", 512)
    max_views = job_input.get("max_views", 6)

    if not mesh_url and not mesh_base64:
        return {"error": "Either mesh_url or mesh_base64 must be provided", "success": False}

    if not image_url and not image_base64:
        return {"error": "Either image_url or image_base64 must be provided for texturing", "success": False}

    paint_start = time.time()

    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()

    try:
        # Download/decode mesh
        if mesh_url:
            print(f"Downloading mesh from URL: {mesh_url[:80]}...")
            response = requests.get(mesh_url, timeout=120)
            response.raise_for_status()
            mesh_bytes = response.content
        else:
            if mesh_base64.startswith("data:"):
                mesh_base64 = mesh_base64.split(",", 1)[1]
            mesh_bytes = base64.b64decode(mesh_base64)

        # Save mesh to temp file
        mesh_path = os.path.join(temp_dir, "input_mesh.glb")
        with open(mesh_path, "wb") as f:
            f.write(mesh_bytes)
        print(f"Saved mesh to {mesh_path} ({len(mesh_bytes)} bytes)")

        # Download/decode reference image
        if image_url:
            print(f"Downloading reference image from URL: {image_url[:80]}...")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            image_bytes = response.content
        else:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)

        # Save image to temp file
        image_path = os.path.join(temp_dir, "reference_image.png")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(image_path)
        print(f"Saved reference image to {image_path}")

        # Load Paint pipeline
        paint_pipeline = load_paint_pipeline(
            max_num_view=max_views,
            resolution=paint_resolution
        )

        # Generate textured mesh
        print(f"Generating PBR textures with Paint 2.1...")
        textured_mesh_path = paint_pipeline(
            mesh_path=mesh_path,
            image_path=image_path
        )
        print(f"Textured mesh saved to: {textured_mesh_path}")

        paint_time = time.time() - paint_start

        # Load textured mesh and render
        render_start = time.time()

        loaded = trimesh.load(textured_mesh_path)
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                textured_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            else:
                return {"error": "No valid geometry in textured mesh", "success": False}
        else:
            textured_mesh = loaded

        # Render textured mesh to images
        rendered_images = render_mesh_to_images(
            textured_mesh,
            views=views,
            resolution=render_resolution,
        )

        render_time = time.time() - render_start

        # Upload textured mesh to R2
        with open(textured_mesh_path, "rb") as f:
            textured_bytes = f.read()

        textured_filename = f"textured_{job_id}_{uuid.uuid4().hex[:8]}.glb"
        textured_url = upload_to_r2(textured_bytes, textured_filename, content_type="model/gltf-binary")

        # Upload rendered images to R2
        result_images = {}
        for view_name, png_bytes in rendered_images.items():
            filename = f"painted_{view_name}_{uuid.uuid4().hex[:8]}.png"
            url = upload_to_r2(png_bytes, filename, content_type="image/png")

            if url:
                result_images[view_name] = {"url": url}
            else:
                result_images[view_name] = {
                    "base64": base64.b64encode(png_bytes).decode("utf-8")
                }

        return {
            "success": True,
            "textured_mesh_url": textured_url,
            "textured_mesh_base64": base64.b64encode(textured_bytes).decode("utf-8") if not textured_url else None,
            "images": result_images,
            "views_rendered": list(rendered_images.keys()),
            "mesh_vertices": len(textured_mesh.vertices),
            "mesh_faces": len(textured_mesh.faces),
            "paint_time_seconds": round(paint_time, 2),
            "render_time_seconds": round(render_time, 2),
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Paint/render failed: {e}", "success": False}

    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# =============================================================================
# Multi-View Generation Functions (Hunyuan3D-2mv)
# =============================================================================

_shape_pipeline_mv = None


def load_multiview_pipeline():
    """Load Hunyuan3D-2mv for multi-view shape generation."""
    global _shape_pipeline_mv

    if _shape_pipeline_mv is not None:
        return _shape_pipeline_mv

    print("Loading Hunyuan3D-2mv for multi-view generation...")

    # Add hunyuan3d-2 to path for hy3dgen imports
    sys.path.insert(0, "/app/hunyuan3d-2")

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    _shape_pipeline_mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv',
        use_safetensors=True,
        device='cuda',
    )

    print("Hunyuan3D-2mv loaded successfully")
    return _shape_pipeline_mv


def handle_generate_multiview(job_input: dict, job_id: str = "unknown") -> dict:
    """
    Generate 3D model from multiple view images using Hunyuan3D-2mv.

    Input:
    {
        "action": "generate_multiview",
        "front_image_url": "https://...",    # OR front_image_base64 (REQUIRED)
        "left_image_url": "https://...",     # Optional
        "back_image_url": "https://...",     # Optional
        "right_image_url": "https://...",    # Optional
        "steps": 30,                         # Optional (default 30)
        "octree_resolution": 380,            # Optional (default 380)
        "num_chunks": 20000,                 # Optional (default 20000)
        "seed": 42,                          # Optional
        "output_format": "glb"               # Optional (default glb)
    }

    Returns:
    {
        "success": True,
        "glb_url": "https://r2.../model.glb",
        "generation_time_seconds": 45.0
    }
    """
    from PIL import Image
    import requests

    start_time = time.time()

    # Get parameters
    steps = job_input.get("steps", 30)
    octree_resolution = job_input.get("octree_resolution", 380)
    num_chunks = job_input.get("num_chunks", 20000)
    seed = job_input.get("seed")
    output_format = job_input.get("output_format", "glb")

    # Load images for each view
    views_data = {}

    for view in ["front", "left", "back", "right"]:
        url_key = f"{view}_image_url"
        b64_key = f"{view}_image_base64"

        if job_input.get(url_key):
            try:
                print(f"Downloading {view} image from URL...")
                response = requests.get(job_input[url_key], timeout=60)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                views_data[view] = img
                print(f"Loaded {view} image: {img.size}")
            except Exception as e:
                if view == "front":
                    return {"error": f"Failed to load front image: {e}", "success": False}
                print(f"Warning: Failed to load {view} image: {e}")

        elif job_input.get(b64_key):
            try:
                b64 = job_input[b64_key]
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                views_data[view] = img
                print(f"Loaded {view} image from base64: {img.size}")
            except Exception as e:
                if view == "front":
                    return {"error": f"Failed to decode front image: {e}", "success": False}
                print(f"Warning: Failed to decode {view} image: {e}")

    if "front" not in views_data:
        return {"error": "Front view image is required", "success": False}

    print(f"Loaded {len(views_data)} views: {list(views_data.keys())}")

    # Load pipeline
    try:
        pipeline = load_multiview_pipeline()
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Failed to load Hunyuan3D-2mv pipeline: {e}", "success": False}

    # Generate mesh
    try:
        print(f"Generating 3D with Hunyuan3D-2mv: steps={steps}, octree={octree_resolution}")

        generator = torch.manual_seed(seed) if seed else None

        result = pipeline(
            image=views_data,
            num_inference_steps=steps,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            generator=generator,
            output_type='trimesh',
        )

        mesh = result[0] if isinstance(result, list) else result

        print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Generation failed: {e}", "success": False}

    generation_time = time.time() - start_time

    # Export mesh
    try:
        buffer = io.BytesIO()
        mesh.export(buffer, file_type=output_format)
        mesh_bytes = buffer.getvalue()
        print(f"Exported {output_format}: {len(mesh_bytes)} bytes")
    except Exception as e:
        return {"error": f"Export failed: {e}", "success": False}

    # Upload to R2
    filename = f"multiview_{job_id}_{uuid.uuid4().hex[:8]}.{output_format}"
    glb_url = upload_to_r2(mesh_bytes, filename)

    if glb_url:
        return {
            "success": True,
            "glb_url": glb_url,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "views_used": list(views_data.keys()),
            "generation_time_seconds": round(generation_time, 2),
        }
    else:
        # Fallback to base64
        return {
            "success": True,
            "glb_base64": base64.b64encode(mesh_bytes).decode("utf-8"),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "views_used": list(views_data.keys()),
            "generation_time_seconds": round(generation_time, 2),
        }


def handler(job: dict) -> dict:
    """
    RunPod handler function.

    Supports actions:
    - "generate" (default): Generate 3D model from text/image using Hunyuan3D-2.1
    - "generate_multiview": Generate 3D model from 4 views using Hunyuan3D-2mv
    - "validate": Validate mesh for printability
    - "render_preview": Render untextured mesh to PNG images (grey material)
    - "render_painted": Apply PBR textures with Paint 2.1 and render to PNG

    Generate Input (Hunyuan3D-2.1 - H100 MAX QUALITY):
    {
        "input": {
            "action": "generate",
            "prompt": "a detailed dragon figurine",  # OR
            "image_url": "https://...",              # OR
            "image_base64": "data:image/png;base64,...",
            "steps": 100,              # Max detail (default 100)
            "guidance_scale": 7.5,     # Input adherence (default 7.5)
            "dual_guidance_scale": 10.5,  # Enhanced guidance (default 10.5)
            "octree_resolution": 512,  # Max geometry detail (default 512)
            "num_chunks": 12000,       # Processing chunks (default 12000)
            "output_format": "glb"
        }
    }

    Validate Input:
    {
        "input": {
            "action": "validate",
            "mesh_url": "https://...",
            "mesh_base64": "...",
            "printer": "anycubic_photon_m3",
            "target_size_mm": 50.0,
            "check_overhangs": true,
            "auto_repair": false,
            "build_volume_mm": {"x": 180, "y": 164, "z": 102}
        }
    }
    """
    try:
        job_input = job.get("input", {})
        job_id = job.get("id", "unknown")
        action = job_input.get("action", "generate")

        if action == "generate":
            return handle_generate(job_input, job_id=job_id)
        elif action == "generate_multiview":
            return handle_generate_multiview(job_input, job_id=job_id)
        elif action == "validate":
            return handle_validate(job_input)
        elif action == "render_preview":
            return handle_render_preview(job_input)
        elif action == "render_painted":
            return handle_render_painted(job_input, job_id=job_id)
        else:
            return {"error": f"Unknown action: {action}. Use 'generate', 'generate_multiview', 'validate', 'render_preview', or 'render_painted'"}

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


# =============================================================================
# Initialization
# =============================================================================

print("=" * 60)
print("Initializing Hunyuan3D-2.1 + Validation worker (H100 optimized)")
print("=" * 60)
try:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    load_shape_pipeline()
    print("Worker ready!")
    print("=" * 60)
except Exception as e:
    print(f"Warning: Failed to pre-load model: {e}")
    print("Model will be loaded on first request")

# Start RunPod handler
runpod.serverless.start({"handler": handler})
# Build trigger v44
