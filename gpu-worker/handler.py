"""
RunPod Serverless Handler for Hunyuan3D + Mesh Validation.

This handler runs on RunPod's serverless GPU infrastructure to:
1. Generate 3D models from text prompts or images using Hunyuan3D
2. Validate 3D meshes for printability (overhangs, watertight, etc.)

Hunyuan3D-2.1 is an image-to-3D model. For text-to-3D, we first generate
an image using SDXL-Turbo, then convert that image to 3D.

Version: 2.4.0 - Upgraded to Hunyuan3D-2.1 (3.0B model) with maximum quality parameters
"""

import base64
import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import runpod
import torch

# Add Hunyuan3D to path
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

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    # Load Hunyuan3D-2.1 - the latest 3.0B parameter model
    # Note: 2.1 is a separate repo, not a subfolder of 2.0
    # See: https://huggingface.co/tencent/Hunyuan3D-2.1
    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        torch_dtype=torch.float16,
        device_map="auto",
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
    steps: int = 50,
    guidance_scale: float = 7.5,
    octree_resolution: int = 384,
    face_count: int = 90000,
    output_format: str = "glb",
) -> bytes:
    """
    Generate a 3D model from text or image using Hunyuan3D-2.1.

    For text-to-3D: First generates an image with SDXL-Turbo, then converts to 3D.
    For image-to-3D: Directly converts the provided image to 3D.

    Default parameters optimized for maximum quality with Hunyuan3D-2.1:
    - steps: 50 (default for v2.1, higher = more detail)
    - guidance_scale: 7.5 (balanced adherence to input)
    - octree_resolution: 384 (default for v2.1, higher = finer geometric detail)
    - face_count: 90000 (high detail mesh output)
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
    print(f"Generating 3D model: steps={steps}, guidance={guidance_scale}")
    start_time = time.time()

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
    else:
        mesh = getattr(result, 'mesh', result)

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

    # Binary STL: 80 byte header + 4 byte triangle count
    # We can't easily distinguish, but if it's not ASCII and not other formats, assume STL

    # PLY: starts with "ply"
    if mesh_bytes[:3].lower() == b'ply':
        return 'ply'

    # OBJ: typically starts with comments (#) or vertex (v )
    first_lines = mesh_bytes[:200].decode('utf-8', errors='ignore').lower()
    if first_lines.startswith('#') or '\nv ' in first_lines or first_lines.startswith('v '):
        return 'obj'

    # Default to GLB since that's what Hunyuan3D outputs
    # and it's the most common format we'll receive
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
    # Downward-facing normals have negative Z component
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
        needs_supports = False  # Minor overhangs might print OK
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
        is_winding_consistent = True  # Assume OK if can't check

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
) -> Dict[str, Any]:
    """
    Comprehensive mesh validation for 3D printing.

    Args:
        mesh_base64: Base64 encoded mesh file (STL, GLB, OBJ, PLY)
        printer: Printer preset name
        target_size_mm: Scale mesh to this size (largest dimension)
        check_overhangs: Whether to analyze overhangs
        auto_repair: Attempt automatic repairs
        custom_build_volume: Override build volume from preset
        file_type: File type hint (stl, glb, obj, ply) - required for BytesIO loading

    Returns:
        Validation results with issues and recommendations
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
            # Extract file type from data URI if present
            data_header = mesh_base64.split(",", 0)[0] if "," in mesh_base64 else ""
            if not file_type and "model/" in data_header:
                # Try to extract from MIME type like "data:model/stl;base64"
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
    original_dimensions = None
    if target_size_mm and hasattr(mesh, 'bounds'):
        bounds = mesh.bounds
        original_dimensions = (bounds[1] - bounds[0]).tolist()
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
        target_size_mm=None,  # Already scaled
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

    if auto_repair:
        try:
            # Track initial state
            initial_watertight = mesh.is_watertight
            initial_vertices = len(mesh.vertices)
            initial_faces = len(mesh.faces)

            # Try PyMeshFix first (much more powerful than trimesh)
            pymeshfix_success = False
            try:
                import pymeshfix

                print("Using PyMeshFix for advanced mesh repair...")

                # Create PyTMesh object for low-level control
                tin = pymeshfix.PyTMesh(verbose=False)
                tin.load_array(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))

                # Track boundaries (holes) before repair
                initial_boundaries = tin.boundaries()
                repair_stats["initial_holes"] = initial_boundaries

                # Step 1: Join nearby disconnected components
                try:
                    tin.join_closest_components()
                    repairs_made.append("Joined nearby components")
                except Exception as e:
                    print(f"join_closest_components skipped: {e}")

                # Step 2: Fill ALL holes (nbe=0 means all holes)
                # refine=True adds vertices to match surrounding density
                try:
                    holes_filled = tin.fill_small_boundaries(nbe=0, refine=True)
                    if holes_filled > 0:
                        repairs_made.append(f"Filled {holes_filled} holes")
                        repair_stats["holes_filled"] = holes_filled
                except Exception as e:
                    print(f"fill_small_boundaries error: {e}")
                    # Try without refine
                    try:
                        holes_filled = tin.fill_small_boundaries(nbe=0, refine=False)
                        if holes_filled > 0:
                            repairs_made.append(f"Filled {holes_filled} holes (simple)")
                            repair_stats["holes_filled"] = holes_filled
                    except:
                        pass

                # Step 3: Remove self-intersections (iterative algorithm)
                try:
                    tin.clean(max_iters=10, inner_loops=3)
                    repairs_made.append("Removed self-intersections")
                except Exception as e:
                    print(f"clean() error: {e}")

                # Step 4: Remove small disconnected components (debris)
                try:
                    tin.remove_smallest_components()
                    repairs_made.append("Removed debris components")
                except Exception as e:
                    print(f"remove_smallest_components skipped: {e}")

                # Get repaired mesh
                final_boundaries = tin.boundaries()
                repair_stats["final_holes"] = final_boundaries
                vclean, fclean = tin.return_arrays()

                # Create new trimesh from repaired arrays
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

                # Fill holes
                if not mesh.is_watertight:
                    trimesh.repair.fill_holes(mesh)
                    if mesh.is_watertight:
                        repairs_made.append("Filled holes (trimesh)")

                # Fix normals
                if hasattr(mesh, 'fix_normals'):
                    mesh.fix_normals()
                    repairs_made.append("Fixed normals")

                # Remove degenerate faces
                try:
                    mesh.update_faces(mesh.nondegenerate_faces())
                    repairs_made.append("Removed degenerate faces")
                except Exception:
                    pass  # Skip if not supported

            # Track final state
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

            # Export repaired mesh
            if repairs_made:
                buffer = io.BytesIO()
                mesh.export(buffer, file_type="glb")
                repaired_mesh_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                print(f"Repair complete: watertight {initial_watertight} -> {final_watertight}")

        except Exception as e:
            repairs_made.append(f"Repair failed: {e}")
            repair_stats["error"] = str(e)

    # Calculate summary
    critical_issues = [i for i in issues if i.get("severity") == "critical" and not i.get("passed")]
    warning_issues = [i for i in issues if i.get("severity") == "warning" and not i.get("passed")]

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
        else:  # FDM
            recommendations.append("Enable supports in slicer (tree supports work well for organic shapes)")
            recommendations.append("Orient largest flat surface on build plate")

    validation_time = time.time() - start_time

    # Extract issue/warning strings for the simplified API response
    issue_strings = [i["message"] for i in issues if i.get("severity") == "critical" and not i.get("passed")]
    warning_strings = [i["message"] for i in issues if i.get("severity") == "warning" and not i.get("passed")]

    # Get mesh properties for response
    is_watertight = mesh.is_watertight if hasattr(mesh, 'is_watertight') else False
    has_consistent_normals = mesh.is_winding_consistent if hasattr(mesh, 'is_winding_consistent') else True
    has_positive_volume = mesh.is_volume if hasattr(mesh, 'is_volume') else is_watertight

    # Calculate volume in cm³
    volume_cm3 = None
    if is_watertight and hasattr(mesh, 'volume'):
        try:
            volume_cm3 = float(mesh.volume) / 1000  # mm³ to cm³
        except:
            pass

    # Format overhangs for API
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

        # Mesh info
        "vertices": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
        "faces": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        "dimensions_mm": dim_check.get("mesh_dimensions_mm"),
        "volume_cm3": round(volume_cm3, 3) if volume_cm3 else None,

        # Validation results
        "is_watertight": is_watertight,
        "has_consistent_normals": has_consistent_normals,
        "has_positive_volume": has_positive_volume,
        "fits_build_volume": dim_check.get("fits_build_volume", False),

        # Issues and warnings
        "issues": issue_strings,
        "warnings": warning_strings,
        "recommendations": recommendations,

        # Overhangs
        "overhangs": overhang_response,

        # Repair info
        "repaired": len(repairs_made) > 0 and repaired_mesh_base64 is not None,
        "repairs_made": repairs_made,
        "repair_stats": repair_stats if repair_stats else None,
        "repaired_mesh_base64": repaired_mesh_base64,

        # Metadata
        "target_size_mm": target_size_mm,
        "validation_time_seconds": round(validation_time, 2),
    }


# =============================================================================
# Handler Functions
# =============================================================================

def handle_generate(job_input: dict) -> dict:
    """Handle 3D generation request."""
    prompt = job_input.get("prompt")
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")

    if not any([prompt, image_url, image_base64]):
        return {"error": "No input provided. Provide prompt, image_url, or image_base64"}

    # Hunyuan3D-2.1 optimal defaults
    steps = job_input.get("steps", 50)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    octree_resolution = job_input.get("octree_resolution", 384)
    face_count = job_input.get("face_count", 90000)
    output_format = job_input.get("output_format", "glb")

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

    glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")

    return {
        "glb_base64": glb_base64,
        "generation_time": round(generation_time, 2),
        "vertices": vertices,
        "faces": faces,
        "file_size_bytes": len(glb_bytes),
    }


def handle_validate(job_input: dict) -> dict:
    """Handle mesh validation request.

    Supports two input methods:
    1. mesh_url: URL to download the mesh from (preferred, more efficient)
    2. mesh_base64: Base64 encoded mesh data (fallback)
    """
    import requests
    from urllib.parse import urlparse

    mesh_url = job_input.get("mesh_url")
    mesh_base64 = job_input.get("mesh_base64")
    file_type = job_input.get("file_type")  # Optional hint

    if not mesh_url and not mesh_base64:
        return {"error": "Either mesh_url or mesh_base64 must be provided"}

    # If URL provided, download the mesh directly (more efficient than base64 in JSON)
    if mesh_url:
        try:
            # Extract file extension from URL if file_type not provided
            if not file_type:
                parsed = urlparse(mesh_url)
                path = parsed.path.lower()
                # Remove query string artifacts
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
            response = requests.get(mesh_url, timeout=60)
            response.raise_for_status()
            mesh_bytes = response.content
            mesh_base64 = base64.b64encode(mesh_bytes).decode('utf-8')
            print(f"Downloaded {len(mesh_bytes)} bytes from URL")
        except requests.RequestException as e:
            return {"error": f"Failed to download mesh from URL: {str(e)}"}

    return validate_mesh(
        mesh_base64=mesh_base64,
        printer=job_input.get("printer", "generic_resin"),
        target_size_mm=job_input.get("target_size_mm"),
        check_overhangs=job_input.get("check_overhangs", True),
        auto_repair=job_input.get("auto_repair", False),
        custom_build_volume=job_input.get("build_volume_mm"),
        file_type=file_type,
    )


def handler(job: dict) -> dict:
    """
    RunPod handler function.

    Supports two actions:
    - "generate" (default): Generate 3D model from text/image
    - "validate": Validate mesh for printability

    Generate Input (Hunyuan3D-2.1 - 3.0B model):
    {
        "input": {
            "action": "generate",  # optional, default
            "prompt": "a detailed dragon figurine",  # OR
            "image_url": "https://...",              # OR
            "image_base64": "data:image/png;base64,...",
            "steps": 50,           # Higher = more detail (default 50)
            "guidance_scale": 7.5, # How closely to follow input (default 7.5)
            "octree_resolution": 384,  # Geometry detail (default 384)
            "face_count": 90000,   # Mesh complexity (default 90000)
            "output_format": "glb"
        }
    }

    Validate Input:
    {
        "input": {
            "action": "validate",
            "mesh_url": "https://...",  # URL to download mesh (preferred, efficient)
            "mesh_base64": "...",  # OR Base64 encoded STL/GLB/OBJ (fallback)
            "printer": "anycubic_photon_m3",  # or "generic_resin", "generic_fdm"
            "target_size_mm": 50.0,  # optional scaling
            "check_overhangs": true,
            "auto_repair": false,
            "build_volume_mm": {"x": 180, "y": 164, "z": 102}  # optional override
        }
    }
    """
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "generate")

        if action == "generate":
            return handle_generate(job_input)
        elif action == "validate":
            return handle_validate(job_input)
        else:
            return {"error": f"Unknown action: {action}. Use 'generate' or 'validate'"}

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


# =============================================================================
# Initialization
# =============================================================================

print("Initializing Hunyuan3D + Validation worker...")
try:
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
