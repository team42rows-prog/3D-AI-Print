"""
MCP Server for 42ROWS 3D Print Generator.

This module exposes the 3D generation capabilities as MCP tools
that can be used by Claude, Cursor, and other MCP-compatible clients.
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from src.core.constants import Quality, InputType, OutputFormat
from src.core.config import get_settings
from src.generators.runpod_client import RunPodClient
from src.generators.imagen_client import ImagenClient
from src.validators.mesh_validator import MeshValidator
from src.utils.image_preprocessing import preprocess_for_3d
from src.orientation import OrientationOptimizer, PrinterType, CrossSectionAnalyzer
from src.sla_supports import generate_supports_for_mesh, SupportTreeConfig

logger = logging.getLogger(__name__)


def _get_server_config() -> tuple[str, int]:
    """Get host and port for MCP server based on environment."""
    from apify import Actor

    host = "0.0.0.0"

    # Apify Standby mode uses ACTOR_STANDBY_PORT
    if Actor.is_at_home():
        port = int(os.environ.get("ACTOR_STANDBY_PORT", "3000"))
    else:
        port = int(os.environ.get("PORT", "3000"))

    return host, port


# Get server configuration
_host, _port = _get_server_config()

# Initialize MCP Server with host/port configured
mcp = FastMCP(
    name="42rows-3d-print-generator",
    host=_host,
    port=_port,
)


# ============================================================================
# Output Models
# ============================================================================

class ModelDimensions(BaseModel):
    """3D model dimensions in millimeters."""
    x: float = Field(description="Width in mm")
    y: float = Field(description="Depth in mm")
    z: float = Field(description="Height in mm")


class GenerationResult(BaseModel):
    """Result of 3D model generation."""
    success: bool
    model_url: Optional[str] = Field(None, description="Direct download URL for the model")
    model_base64: Optional[str] = Field(None, description="Base64 encoded model data")
    format: str = Field(description="Output format (stl, glb, obj)")
    vertices: int = Field(0, description="Number of vertices")
    faces: int = Field(0, description="Number of faces")
    dimensions_mm: Optional[ModelDimensions] = None
    file_size_bytes: int = Field(0, description="File size in bytes")
    generation_time_seconds: float = Field(0, description="Time taken to generate")
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Detailed information about a 3D model."""
    vertices: int
    faces: int
    is_manifold: bool = Field(description="Whether the mesh is watertight (printable)")
    is_printable: bool = Field(description="Whether the model can be 3D printed")
    dimensions_mm: ModelDimensions
    volume_cm3: Optional[float] = Field(None, description="Volume in cubic centimeters")
    surface_area_cm2: Optional[float] = Field(None, description="Surface area in square centimeters")
    bounding_box_mm: ModelDimensions
    issues: list[str] = Field(default_factory=list, description="List of detected issues")
    estimated_print_weight_grams: Optional[float] = Field(None, description="Estimated weight if printed")


class ConversionResult(BaseModel):
    """Result of format conversion."""
    success: bool
    model_url: Optional[str] = None
    model_base64: Optional[str] = None
    source_format: str
    target_format: str
    file_size_bytes: int = 0
    error: Optional[str] = None


class OverhangInfo(BaseModel):
    """Overhang detection results."""
    total_faces: int = Field(description="Total number of faces in the mesh")
    overhang_faces: int = Field(description="Number of faces with overhangs")
    overhang_percentage: float = Field(description="Percentage of faces with overhangs")
    max_overhang_angle: float = Field(description="Maximum overhang angle in degrees")
    needs_supports: bool = Field(description="Whether the model needs support structures")


class ValidationResult(BaseModel):
    """Result of mesh validation for 3D printing."""
    valid: bool = Field(description="Whether the mesh is valid for printing")
    printer: str = Field(description="Printer preset used for validation")
    printer_type: str = Field(description="Type of printer (resin, fdm, sls)")

    # Mesh info
    vertices: int = Field(0, description="Number of vertices")
    faces: int = Field(0, description="Number of faces")
    dimensions_mm: Optional[ModelDimensions] = None
    volume_cm3: Optional[float] = Field(None, description="Volume in cubic centimeters")

    # Validation results
    is_watertight: bool = Field(False, description="Mesh is watertight (manifold)")
    has_consistent_normals: bool = Field(False, description="Face normals are consistent")
    has_positive_volume: bool = Field(False, description="Mesh has positive volume")
    fits_build_volume: bool = Field(False, description="Model fits in printer build volume")

    # Issues and warnings
    issues: list[str] = Field(default_factory=list, description="Critical issues that prevent printing")
    warnings: list[str] = Field(default_factory=list, description="Warnings that may affect print quality")

    # Overhangs (if checked)
    overhangs: Optional[OverhangInfo] = None

    # Repaired mesh (if auto_repair=True)
    repaired: bool = Field(False, description="Whether mesh was repaired")
    repaired_model_url: Optional[str] = Field(None, description="URL to download the repaired model")
    repaired_model_base64: Optional[str] = Field(None, description="Base64 encoded repaired model")
    repair_stats: Optional[dict] = Field(None, description="Detailed repair statistics")
    repairs_made: list[str] = Field(default_factory=list, description="List of repairs performed")

    error: Optional[str] = None


class RotationInfo(BaseModel):
    """Rotation parameters for orientation optimization."""
    axis: list[float] = Field(description="Rotation axis [x, y, z]")
    angle_rad: float = Field(description="Rotation angle in radians")
    angle_deg: float = Field(description="Rotation angle in degrees")
    euler_angles_deg: list[float] = Field(description="Euler angles [X, Y, Z] in degrees")
    matrix: list[list[float]] = Field(description="3x3 rotation matrix")


class CrossSectionInfo(BaseModel):
    """Cross-section analysis results (SLA only)."""
    layer_count: int = Field(description="Number of layers analyzed")
    max_area_mm2: float = Field(description="Maximum cross-sectional area in mm²")
    avg_area_mm2: float = Field(description="Average cross-sectional area in mm²")
    max_peel_force_n: float = Field(description="Estimated maximum peel force in Newtons")
    max_peel_force_z_mm: float = Field(description="Z height of maximum peel force layer")
    critical_layer_count: int = Field(description="Number of layers with significant area changes")


class OrientationScores(BaseModel):
    """Orientation quality scores."""
    unprintability: float = Field(description="Overall unprintability score (lower is better)")
    bottom_area_mm2: float = Field(description="Bottom contact area in mm²")
    overhang_area_mm2: float = Field(description="Overhang area requiring supports in mm²")
    contour_length_mm: float = Field(description="Bottom contour length in mm")


class OrientationResultModel(BaseModel):
    """Result of orientation optimization."""
    success: bool = Field(description="Whether optimization succeeded")
    printer_type: str = Field(description="Printer type used (fdm or sla)")

    # Rotation parameters
    rotation: Optional[RotationInfo] = None

    # Quality scores
    scores: Optional[OrientationScores] = None

    # Cross-section analysis (SLA only)
    cross_section: Optional[CrossSectionInfo] = None

    # Rotated mesh output
    rotated_model_url: Optional[str] = Field(None, description="URL of the rotated model")
    rotated_model_base64: Optional[str] = Field(None, description="Base64 of the rotated model")

    # Metadata
    processing_time_seconds: float = Field(0, description="Processing time")
    critical_angle_deg: float = Field(description="Critical overhang angle used")

    error: Optional[str] = None


class SupportGenerationResult(BaseModel):
    """Result of SLA support generation."""
    success: bool = Field(description="Whether generation succeeded")

    # Statistics
    num_supports: int = Field(0, description="Number of support points generated")
    num_heads: int = Field(0, description="Number of support heads")
    num_pillars: int = Field(0, description="Number of support pillars")
    num_bridges: int = Field(0, description="Number of bridges between supports")

    # Output mesh
    supports_model_url: Optional[str] = Field(None, description="URL of the supports mesh")
    supports_model_base64: Optional[str] = Field(None, description="Base64 of the supports mesh")
    combined_model_url: Optional[str] = Field(None, description="URL of model + supports combined")
    combined_model_base64: Optional[str] = Field(None, description="Base64 of model + supports combined")

    # Mesh info
    support_vertices: int = Field(0, description="Number of vertices in support mesh")
    support_faces: int = Field(0, description="Number of faces in support mesh")

    # Metadata
    processing_time_seconds: float = Field(0, description="Processing time")
    overhang_angle_deg: float = Field(45.0, description="Overhang angle used for detection")
    elevation_mm: float = Field(10.0, description="Object elevation from build plate")

    error: Optional[str] = None


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def generate_3d_from_text(
    prompt: str,
    style: str = "realistic",
    output_format: str = "stl",
    quality: str = "standard",
) -> GenerationResult:
    """
    Generate a 3D printable model from a text description.

    Uses AI to first create an image from the text, then converts
    that image into a 3D mesh suitable for 3D printing.

    Args:
        prompt: Text description of the 3D model to generate.
                Example: "A small dragon figurine with detailed scales"
        style: Visual style - one of: realistic, cartoon, miniature, artistic, mechanical
        output_format: Output file format - one of: stl, glb, obj
        quality: Generation quality - one of: draft (fast), standard (balanced), high (detailed)

    Returns:
        GenerationResult with the generated model URL or base64 data
    """
    import time
    start_time = time.time()

    try:
        # Map quality string to enum
        quality_map = {
            "draft": Quality.LITE,
            "standard": Quality.STANDARD,
            "high": Quality.HIGH,
        }
        quality_enum = quality_map.get(quality, Quality.STANDARD)

        # Enhance prompt based on style
        style_prompts = {
            "realistic": f"{prompt}, highly detailed, photorealistic",
            "cartoon": f"{prompt}, cartoon style, stylized, smooth",
            "miniature": f"{prompt}, tabletop miniature, detailed base, gaming figure",
            "artistic": f"{prompt}, artistic sculpture, elegant design",
            "mechanical": f"{prompt}, mechanical parts, industrial design, precise edges",
        }
        enhanced_prompt = style_prompts.get(style, prompt)

        # Generate 3D model via RunPod
        async with RunPodClient() as client:
            glb_bytes = await client.run_sync(
                input_type=InputType.TEXT,
                input_data=enhanced_prompt,
                quality=quality_enum,
            )

        generation_time = time.time() - start_time

        # Convert to requested format
        output_bytes, mesh_info = await _convert_and_analyze(
            glb_bytes,
            output_format
        )

        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"model_{unique_id}.{output_format}"

        # Always upload to Key-Value Store for direct download URL
        model_url = await _upload_to_storage(output_bytes, filename)

        # Only include base64 for small files (< 10MB) to avoid response size issues
        file_size = len(output_bytes)
        if file_size < 10 * 1024 * 1024:  # 10MB threshold
            model_base64 = base64.b64encode(output_bytes).decode('utf-8')
        else:
            model_base64 = None
            logger.info(f"Skipping base64 encoding for large file ({file_size / 1024 / 1024:.1f}MB)")

        return GenerationResult(
            success=True,
            model_url=model_url,
            model_base64=model_base64,
            format=output_format,
            vertices=mesh_info.get("vertices", 0),
            faces=mesh_info.get("faces", 0),
            dimensions_mm=ModelDimensions(
                x=mesh_info.get("dimensions", [0, 0, 0])[0],
                y=mesh_info.get("dimensions", [0, 0, 0])[1],
                z=mesh_info.get("dimensions", [0, 0, 0])[2],
            ) if mesh_info.get("dimensions") else None,
            file_size_bytes=file_size,
            generation_time_seconds=round(generation_time, 2),
        )

    except Exception as e:
        logger.exception("Error generating 3D model from text")
        return GenerationResult(
            success=False,
            format=output_format,
            error=str(e),
        )


@mcp.tool()
async def generate_3d_from_image(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    output_format: str = "stl",
    quality: str = "standard",
) -> GenerationResult:
    """
    Convert a 2D image into a 3D printable model.

    Works best with images that have:
    - Clear subject on plain/simple background
    - Good lighting and contrast
    - Single object (not complex scenes)

    Args:
        image_url: URL of the image to convert to 3D
        image_base64: Base64 encoded image (alternative to URL)
        output_format: Output file format - one of: stl, glb, obj
        quality: Generation quality - one of: draft (fast), standard (balanced), high (detailed)

    Returns:
        GenerationResult with the generated model URL or base64 data
    """
    import time
    start_time = time.time()

    if not image_url and not image_base64:
        return GenerationResult(
            success=False,
            format=output_format,
            error="Either image_url or image_base64 must be provided",
        )

    try:
        # Map quality
        quality_map = {
            "draft": Quality.LITE,
            "standard": Quality.STANDARD,
            "high": Quality.HIGH,
        }
        quality_enum = quality_map.get(quality, Quality.STANDARD)

        # Determine input type
        if image_base64:
            input_type = InputType.IMAGE_BASE64
            input_data = image_base64
        else:
            input_type = InputType.IMAGE_URL
            input_data = image_url

        # Generate 3D model via RunPod
        async with RunPodClient() as client:
            glb_bytes = await client.run_sync(
                input_type=input_type,
                input_data=input_data,
                quality=quality_enum,
            )

        generation_time = time.time() - start_time

        # Convert to requested format
        output_bytes, mesh_info = await _convert_and_analyze(
            glb_bytes,
            output_format
        )

        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"model_{unique_id}.{output_format}"

        # Always upload to Key-Value Store for direct download URL
        model_url = await _upload_to_storage(output_bytes, filename)

        # Only include base64 for small files (< 10MB) to avoid response size issues
        file_size = len(output_bytes)
        if file_size < 10 * 1024 * 1024:  # 10MB threshold
            model_base64 = base64.b64encode(output_bytes).decode('utf-8')
        else:
            model_base64 = None
            logger.info(f"Skipping base64 encoding for large file ({file_size / 1024 / 1024:.1f}MB)")

        return GenerationResult(
            success=True,
            model_url=model_url,
            model_base64=model_base64,
            format=output_format,
            vertices=mesh_info.get("vertices", 0),
            faces=mesh_info.get("faces", 0),
            dimensions_mm=ModelDimensions(
                x=mesh_info.get("dimensions", [0, 0, 0])[0],
                y=mesh_info.get("dimensions", [0, 0, 0])[1],
                z=mesh_info.get("dimensions", [0, 0, 0])[2],
            ) if mesh_info.get("dimensions") else None,
            file_size_bytes=file_size,
            generation_time_seconds=round(generation_time, 2),
        )

    except Exception as e:
        logger.exception("Error generating 3D model from image")
        return GenerationResult(
            success=False,
            format=output_format,
            error=str(e),
        )


@mcp.tool()
async def generate_3d_with_imagen(
    prompt: str,
    style: str = "realistic",
    output_format: str = "stl",
    quality: str = "standard",
    target_size_mm: float = 50.0,
    google_api_key: Optional[str] = None,
) -> GenerationResult:
    """
    Generate a high-quality 3D printable model using Google Imagen 3.

    This tool uses Google's Imagen 3 API ($0.03/image) to generate a
    high-quality image from your text prompt, then converts it to a
    3D model using Hunyuan3D. This produces significantly better results
    than the standard text-to-3D pipeline.

    The image is automatically preprocessed for optimal 3D conversion:
    - Background removal
    - Object centering
    - Resize to 1024x1024
    - White background

    The mesh is post-processed with trimesh:
    - Scaled to target_size_mm (largest dimension)
    - Centered at origin
    - Placed on build plate (Z=0)

    Args:
        prompt: Text description of the 3D model to generate.
                Example: "A detailed dragon figurine with scales"
        style: Visual style - one of: realistic, cartoon, miniature, artistic, mechanical
        output_format: Output file format - one of: stl, glb, obj
        quality: 3D generation quality - one of: draft (fast), standard (balanced), high (detailed)
        target_size_mm: Target size in mm for the largest dimension (default: 50mm).
                        Example: 28 for tabletop miniatures, 100 for larger figurines
        google_api_key: Optional Google API key (uses GOOGLE_API_KEY env var if not provided)

    Returns:
        GenerationResult with the generated model URL or base64 data
    """
    import time
    start_time = time.time()

    try:
        # Get API key
        settings = get_settings()
        api_key = google_api_key or settings.google_api_key

        if not api_key:
            return GenerationResult(
                success=False,
                format=output_format,
                error="Google API key not provided. Set GOOGLE_API_KEY environment variable or pass google_api_key parameter.",
            )

        # Map quality string to enum
        quality_map = {
            "draft": Quality.LITE,
            "standard": Quality.STANDARD,
            "high": Quality.HIGH,
        }
        quality_enum = quality_map.get(quality, Quality.STANDARD)

        # Step 1: Generate image with Imagen 3
        logger.info(f"Generating image with Imagen 3: {prompt[:50]}...")
        imagen_client = ImagenClient(api_key)
        image_bytes = await imagen_client.generate_image_for_3d(
            prompt=prompt,
            style=style,
        )

        # Step 2: Generate 3D model from preprocessed image
        logger.info("Converting image to 3D with Hunyuan3D...")
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        async with RunPodClient() as client:
            glb_bytes = await client.run_sync(
                input_type=InputType.IMAGE_BASE64,
                input_data=image_base64,
                quality=quality_enum,
            )

        generation_time = time.time() - start_time

        # Convert to requested format and scale to target size
        output_bytes, mesh_info = await _convert_and_analyze(
            glb_bytes,
            output_format,
            target_size_mm=target_size_mm
        )

        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"model_{unique_id}.{output_format}"

        # Upload to Key-Value Store
        model_url = await _upload_to_storage(output_bytes, filename)

        # Only include base64 for small files (< 10MB) to avoid response size issues
        file_size = len(output_bytes)
        if file_size < 10 * 1024 * 1024:  # 10MB threshold
            model_base64 = base64.b64encode(output_bytes).decode('utf-8')
        else:
            model_base64 = None
            logger.info(f"Skipping base64 encoding for large file ({file_size / 1024 / 1024:.1f}MB)")

        return GenerationResult(
            success=True,
            model_url=model_url,
            model_base64=model_base64,
            format=output_format,
            vertices=mesh_info.get("vertices", 0),
            faces=mesh_info.get("faces", 0),
            dimensions_mm=ModelDimensions(
                x=mesh_info.get("dimensions", [0, 0, 0])[0],
                y=mesh_info.get("dimensions", [0, 0, 0])[1],
                z=mesh_info.get("dimensions", [0, 0, 0])[2],
            ) if mesh_info.get("dimensions") else None,
            file_size_bytes=len(output_bytes),
            generation_time_seconds=round(generation_time, 2),
        )

    except Exception as e:
        logger.exception("Error generating 3D model with Imagen")
        return GenerationResult(
            success=False,
            format=output_format,
            error=str(e),
        )


@mcp.tool()
async def get_model_info(
    model_url: Optional[str] = None,
    model_base64: Optional[str] = None,
) -> ModelInfo:
    """
    Analyze a 3D model and return detailed information about its geometry,
    dimensions, and printability.

    Use this to:
    - Check if a model is printable (watertight/manifold)
    - Get exact dimensions before printing
    - Estimate material usage
    - Identify potential issues

    Args:
        model_url: URL of the 3D model to analyze (STL, OBJ, GLB, PLY)
        model_base64: Base64 encoded model file (alternative to URL)

    Returns:
        ModelInfo with detailed analysis of the model
    """
    import trimesh
    import aiohttp

    if not model_url and not model_base64:
        raise ValueError("Either model_url or model_base64 must be provided")

    try:
        # Load the model
        if model_base64:
            if model_base64.startswith("data:"):
                model_base64 = model_base64.split(",", 1)[1]
            model_bytes = base64.b64decode(model_base64)
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as response:
                    response.raise_for_status()
                    model_bytes = await response.read()

        # Save to temp file for trimesh
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            f.write(model_bytes)
            temp_path = Path(f.name)

        try:
            # Load with trimesh
            loaded = trimesh.load(str(temp_path))

            # Handle scene vs single mesh
            if isinstance(loaded, trimesh.Scene):
                meshes = [
                    g for g in loaded.geometry.values()
                    if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                else:
                    raise ValueError("No valid meshes found in file")
            else:
                mesh = loaded

            # Get properties
            bounds = mesh.bounds
            dimensions = bounds[1] - bounds[0]

            is_watertight = mesh.is_watertight

            volume = None
            if is_watertight:
                try:
                    volume = float(mesh.volume) / 1000  # mm³ to cm³
                except:
                    pass

            surface_area = None
            try:
                surface_area = float(mesh.area) / 100  # mm² to cm²
            except:
                pass

            # Detect issues
            issues = []
            if not is_watertight:
                issues.append("Mesh is not watertight - may have holes or gaps")

            if mesh.is_empty:
                issues.append("Mesh appears to be empty")

            # Check for very thin walls (< 0.4mm typical minimum)
            # This is a simplified check
            if volume and surface_area:
                wall_estimate = volume * 1000 / surface_area / 100 if surface_area > 0 else 0
                if wall_estimate < 0.3:
                    issues.append("Model may have very thin walls that could fail during printing")

            # Estimate weight (assuming PLA density ~1.25 g/cm³)
            estimated_weight = volume * 1.25 if volume else None

            return ModelInfo(
                vertices=len(mesh.vertices),
                faces=len(mesh.faces),
                is_manifold=is_watertight,
                is_printable=is_watertight and len(issues) == 0,
                dimensions_mm=ModelDimensions(
                    x=round(float(dimensions[0]), 2),
                    y=round(float(dimensions[1]), 2),
                    z=round(float(dimensions[2]), 2),
                ),
                volume_cm3=round(volume, 2) if volume else None,
                surface_area_cm2=round(surface_area, 2) if surface_area else None,
                bounding_box_mm=ModelDimensions(
                    x=round(float(dimensions[0]), 2),
                    y=round(float(dimensions[1]), 2),
                    z=round(float(dimensions[2]), 2),
                ),
                issues=issues,
                estimated_print_weight_grams=round(estimated_weight, 1) if estimated_weight else None,
            )

        finally:
            temp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Error analyzing model")
        raise ValueError(f"Failed to analyze model: {str(e)}")


@mcp.tool()
async def convert_3d_format(
    model_url: Optional[str] = None,
    model_base64: Optional[str] = None,
    source_format: Optional[str] = None,
    target_format: str = "stl",
) -> ConversionResult:
    """
    Convert a 3D model between different file formats.

    Supported formats:
    - STL: Standard Triangle Language (most common for 3D printing)
    - OBJ: Wavefront Object (with materials)
    - GLB: GL Binary (compact, web-friendly)
    - PLY: Polygon File Format

    Args:
        model_url: URL of the 3D model to convert
        model_base64: Base64 encoded model (alternative to URL)
        source_format: Source format (auto-detected if not provided)
        target_format: Target format - one of: stl, obj, glb, ply

    Returns:
        ConversionResult with the converted model
    """
    import trimesh
    import aiohttp

    if not model_url and not model_base64:
        return ConversionResult(
            success=False,
            source_format=source_format or "unknown",
            target_format=target_format,
            error="Either model_url or model_base64 must be provided",
        )

    try:
        # Load the model
        if model_base64:
            if model_base64.startswith("data:"):
                model_base64 = model_base64.split(",", 1)[1]
            model_bytes = base64.b64decode(model_base64)
            detected_format = source_format or "glb"
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as response:
                    response.raise_for_status()
                    model_bytes = await response.read()

            # Detect format from URL
            if source_format:
                detected_format = source_format
            else:
                url_lower = model_url.lower()
                if ".stl" in url_lower:
                    detected_format = "stl"
                elif ".obj" in url_lower:
                    detected_format = "obj"
                elif ".glb" in url_lower or ".gltf" in url_lower:
                    detected_format = "glb"
                elif ".ply" in url_lower:
                    detected_format = "ply"
                else:
                    detected_format = "glb"  # Default assumption

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=f".{detected_format}", delete=False) as f:
            f.write(model_bytes)
            temp_path = Path(f.name)

        try:
            # Load with trimesh
            loaded = trimesh.load(str(temp_path))

            # Handle scene vs single mesh
            if isinstance(loaded, trimesh.Scene):
                meshes = [
                    g for g in loaded.geometry.values()
                    if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                else:
                    raise ValueError("No valid meshes found in file")
            else:
                mesh = loaded

            # Export to target format
            output_buffer = io.BytesIO()
            mesh.export(output_buffer, file_type=target_format)
            output_bytes = output_buffer.getvalue()

            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            filename = f"converted_{unique_id}.{target_format}"

            # Always upload to Key-Value Store for direct download URL
            url = await _upload_to_storage(output_bytes, filename)

            # Also include base64 for programmatic access
            model_b64 = base64.b64encode(output_bytes).decode('utf-8')

            return ConversionResult(
                success=True,
                model_url=url,
                model_base64=model_b64,
                source_format=detected_format,
                target_format=target_format,
                file_size_bytes=len(output_bytes),
            )

        finally:
            temp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Error converting model")
        return ConversionResult(
            success=False,
            source_format=source_format or "unknown",
            target_format=target_format,
            error=str(e),
        )


@mcp.tool()
async def optimize_orientation(
    model_url: Optional[str] = None,
    model_base64: Optional[str] = None,
    printer_type: str = "fdm",
    output_format: str = "stl",
    extended_mode: bool = True,
    return_rotated_mesh: bool = True,
) -> OrientationResultModel:
    """
    Find the optimal printing orientation for a 3D model.

    This tool analyzes a 3D mesh and determines the best rotation to minimize
    support material, improve print quality, and reduce print failures.

    The optimization considers different factors based on printer type:

    FDM (Fused Deposition Modeling):
    - Critical overhang angle: 45° from horizontal
    - Minimizes support surface area
    - Optimizes for layer adhesion strength

    SLA (Stereolithography/Resin):
    - Critical overhang angle: 19° from horizontal (more restrictive)
    - Minimizes support volume
    - Analyzes cross-sectional areas to estimate peel forces
    - Identifies critical layers with high stress

    Args:
        model_url: URL of the 3D model to optimize (STL, OBJ, GLB, PLY)
        model_base64: Base64 encoded model file (alternative to URL)
        printer_type: Type of printer - "fdm" or "sla" (default: "fdm")
        output_format: Format for rotated mesh output - stl, glb, obj (default: "stl")
        extended_mode: Use extended analysis for more accurate results (default: True)
        return_rotated_mesh: Whether to return the rotated mesh (default: True)

    Returns:
        OrientationResultModel with:
        - Optimal rotation parameters (axis, angle, matrix, euler angles)
        - Quality scores (unprintability, overhang area, bottom area)
        - Cross-section analysis (SLA only: peel forces, critical layers)
        - Rotated mesh URL/base64 (if return_rotated_mesh=True)

    Example usage:
        # For FDM printing
        result = optimize_orientation(model_url="...", printer_type="fdm")

        # For resin/SLA printing (more restrictive)
        result = optimize_orientation(model_url="...", printer_type="sla")

        # Apply rotation in your slicer using euler_angles_deg or matrix
    """
    import trimesh
    import aiohttp
    import time

    start_time = time.time()

    if not model_url and not model_base64:
        return OrientationResultModel(
            success=False,
            printer_type=printer_type,
            critical_angle_deg=45.0 if printer_type == "fdm" else 19.0,
            error="Either model_url or model_base64 must be provided",
        )

    # Validate printer_type
    try:
        ptype = PrinterType(printer_type.lower())
    except ValueError:
        return OrientationResultModel(
            success=False,
            printer_type=printer_type,
            critical_angle_deg=45.0,
            error=f"Invalid printer_type: {printer_type}. Must be 'fdm' or 'sla'",
        )

    try:
        # Load the model
        if model_base64:
            if model_base64.startswith("data:"):
                model_base64 = model_base64.split(",", 1)[1]
            model_bytes = base64.b64decode(model_base64)
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as response:
                    response.raise_for_status()
                    model_bytes = await response.read()

        # Save to temp file for trimesh
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            f.write(model_bytes)
            temp_path = Path(f.name)

        try:
            # Load with trimesh
            loaded = trimesh.load(str(temp_path))

            # Handle scene vs single mesh
            if isinstance(loaded, trimesh.Scene):
                meshes = [
                    g for g in loaded.geometry.values()
                    if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                else:
                    raise ValueError("No valid meshes found in file")
            else:
                mesh = loaded

            # Run orientation optimization
            optimizer = OrientationOptimizer(
                printer_type=ptype,
                extended_mode=extended_mode,
            )

            result = optimizer.optimize(mesh)

            processing_time = time.time() - start_time

            # Build rotation info
            rotation_info = RotationInfo(
                axis=result.rotation_axis,
                angle_rad=round(result.rotation_angle, 6),
                angle_deg=round(result.rotation_angle * 180 / 3.14159265, 2),
                euler_angles_deg=[round(a, 2) for a in result.euler_angles_deg],
                matrix=[[round(x, 6) for x in row] for row in result.rotation_matrix.tolist()],
            )

            # Build scores
            scores = OrientationScores(
                unprintability=round(result.unprintability, 4),
                bottom_area_mm2=round(result.bottom_area, 2),
                overhang_area_mm2=round(result.overhang_area, 2),
                contour_length_mm=round(result.contour_length, 2),
            )

            # Build cross-section info (SLA only)
            cross_section_info = None
            if result.cross_section is not None:
                cs = result.cross_section
                cross_section_info = CrossSectionInfo(
                    layer_count=len(cs.z_levels),
                    max_area_mm2=round(cs.max_area, 2),
                    avg_area_mm2=round(cs.avg_area, 2),
                    max_peel_force_n=round(cs.max_peel_force, 3),
                    max_peel_force_z_mm=round(cs.max_peel_force_z, 2),
                    critical_layer_count=len(cs.critical_layers),
                )

            # Generate rotated mesh if requested
            rotated_url = None
            rotated_b64 = None

            if return_rotated_mesh:
                # Apply rotation
                rotated_mesh = mesh.copy()
                rotation_matrix_4x4 = trimesh.transformations.rotation_matrix(
                    result.rotation_angle,
                    result.rotation_axis
                )
                rotated_mesh.apply_transform(rotation_matrix_4x4)

                # Center and place on build plate
                bounds = rotated_mesh.bounds
                center = (bounds[0] + bounds[1]) / 2
                z_offset = -bounds[0][2]
                rotated_mesh.apply_translation([-center[0], -center[1], z_offset])

                # Export
                output_buffer = io.BytesIO()
                rotated_mesh.export(output_buffer, file_type=output_format)
                rotated_bytes = output_buffer.getvalue()

                # Upload to storage
                unique_id = str(uuid.uuid4())[:8]
                filename = f"oriented_{unique_id}.{output_format}"
                rotated_url = await _upload_to_storage(rotated_bytes, filename)
                rotated_b64 = base64.b64encode(rotated_bytes).decode('utf-8')

            return OrientationResultModel(
                success=True,
                printer_type=ptype.value,
                rotation=rotation_info,
                scores=scores,
                cross_section=cross_section_info,
                rotated_model_url=rotated_url,
                rotated_model_base64=rotated_b64,
                processing_time_seconds=round(processing_time, 2),
                critical_angle_deg=optimizer.critical_angle,
            )

        finally:
            temp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Error optimizing orientation")
        return OrientationResultModel(
            success=False,
            printer_type=printer_type,
            critical_angle_deg=45.0 if printer_type == "fdm" else 19.0,
            processing_time_seconds=round(time.time() - start_time, 2),
            error=str(e),
        )


# Printer presets list for documentation
AVAILABLE_PRINTERS = [
    "anycubic_photon_m3",
    "anycubic_photon_m5s",
    "elegoo_mars_3",
    "elegoo_saturn_3",
    "prusa_mk4",
    "prusa_mini",
    "bambu_x1c",
    "bambu_a1_mini",
    "creality_ender_3",
    "formlabs_form_3",
    "formlabs_form_4",
    "generic_resin",
    "generic_fdm",
    "generic_sls",
]


@mcp.tool()
async def validate_for_printer(
    model_url: Optional[str] = None,
    model_base64: Optional[str] = None,
    printer: str = "generic_resin",
    target_size_mm: Optional[float] = None,
    check_overhangs: bool = True,
    auto_repair: bool = False,
    custom_build_volume_x: Optional[float] = None,
    custom_build_volume_y: Optional[float] = None,
    custom_build_volume_z: Optional[float] = None,
) -> ValidationResult:
    """
    Validate a 3D model for printability on a specific printer.

    This tool checks if a 3D mesh is suitable for printing on the specified
    printer, detecting issues like non-manifold geometry, overhangs that need
    supports, size constraints, and more.

    Available printer presets:
    - RESIN (MSLA): anycubic_photon_m3, anycubic_photon_m5s, elegoo_mars_3,
                    elegoo_saturn_3, formlabs_form_3, formlabs_form_4, generic_resin
    - FDM: prusa_mk4, prusa_mini, bambu_x1c, bambu_a1_mini, creality_ender_3, generic_fdm
    - SLS: generic_sls

    Validation checks include:
    - Watertight (manifold) mesh - required for all printers
    - Consistent face normals
    - Positive volume
    - Fits within build volume
    - Overhang detection (faces that need supports)
    - Degenerate faces (zero-area triangles)

    Args:
        model_url: URL of the 3D model to validate (STL, OBJ, GLB, PLY)
        model_base64: Base64 encoded model file (alternative to URL)
        printer: Printer preset name (default: "generic_resin")
                 See available presets above
        target_size_mm: Scale model to this size before validation (largest dimension)
        check_overhangs: Whether to detect overhangs (default: True)
        auto_repair: Attempt to repair non-manifold meshes (default: False)
        custom_build_volume_x: Custom build volume X in mm (overrides preset)
        custom_build_volume_y: Custom build volume Y in mm (overrides preset)
        custom_build_volume_z: Custom build volume Z in mm (overrides preset)

    Returns:
        ValidationResult with validation status, issues, warnings, and optionally repaired mesh
    """
    import aiohttp

    if not model_url and not model_base64:
        return ValidationResult(
            valid=False,
            printer=printer,
            printer_type="unknown",
            error="Either model_url or model_base64 must be provided",
        )

    try:
        # Build custom build volume if provided
        custom_build_volume = None
        if custom_build_volume_x or custom_build_volume_y or custom_build_volume_z:
            custom_build_volume = {
                "x": custom_build_volume_x or 200,
                "y": custom_build_volume_y or 200,
                "z": custom_build_volume_z or 200,
            }

        # Prepare mesh input - prefer URL for efficiency (RunPod downloads directly)
        mesh_url_to_send = None
        mesh_b64_to_send = None

        if model_url:
            # Pass URL directly to RunPod - most efficient, avoids large JSON payloads
            mesh_url_to_send = model_url
            logger.info(f"Sending mesh_url to RunPod for validation (efficient)")
        elif model_base64:
            # Fallback to base64 if URL not available
            if model_base64.startswith("data:"):
                model_base64 = model_base64.split(",", 1)[1]
            mesh_b64_to_send = model_base64
            logger.info(f"Sending mesh_base64 to RunPod for validation (fallback)")

        # Call RunPod for GPU-accelerated validation
        async with RunPodClient() as client:
            result = await client.validate_mesh(
                mesh_url=mesh_url_to_send,
                mesh_base64=mesh_b64_to_send,
                printer=printer,
                target_size_mm=target_size_mm,
                check_overhangs=check_overhangs,
                auto_repair=auto_repair,
                custom_build_volume=custom_build_volume,
            )

        # Parse the result from RunPod
        if not result.get("success", False):
            return ValidationResult(
                valid=False,
                printer=printer,
                printer_type=result.get("printer_type", "unknown"),
                error=result.get("error", "Validation failed"),
            )

        # Build overhang info if available
        overhang_info = None
        if result.get("overhangs"):
            oh = result["overhangs"]
            overhang_info = OverhangInfo(
                total_faces=oh.get("total_faces", 0),
                overhang_faces=oh.get("overhang_faces", 0),
                overhang_percentage=oh.get("overhang_percentage", 0),
                max_overhang_angle=oh.get("max_overhang_angle", 0),
                needs_supports=oh.get("needs_supports", False),
            )

        # Build dimensions
        dims = result.get("dimensions_mm", {})
        dimensions = ModelDimensions(
            x=dims.get("x", 0),
            y=dims.get("y", 0),
            z=dims.get("z", 0),
        ) if dims else None

        # Handle repaired mesh
        repaired_url = None
        repaired_b64 = None
        logger.info(f"Repair status: repaired={result.get('repaired')}, has_url={bool(result.get('repaired_mesh_url'))}, has_base64={bool(result.get('repaired_mesh_base64'))}")
        logger.info(f"Repair stats: {result.get('repair_stats')}")
        logger.info(f"Repairs made: {result.get('repairs_made')}")

        if result.get("repaired"):
            # Prefer URL from R2 (RunPod uploads large files to R2)
            if result.get("repaired_mesh_url"):
                repaired_url = result["repaired_mesh_url"]
                logger.info(f"Using repaired mesh URL from R2: {repaired_url}")
            elif result.get("repaired_mesh_base64"):
                # Fallback: base64 for small files, upload to our storage
                logger.info("Uploading repaired mesh to storage...")
                repaired_bytes = base64.b64decode(result["repaired_mesh_base64"])
                unique_id = str(uuid.uuid4())[:8]
                filename = f"repaired_{unique_id}.glb"
                repaired_url = await _upload_to_storage(repaired_bytes, filename)
                repaired_b64 = result["repaired_mesh_base64"]
                logger.info(f"Repaired mesh uploaded: {repaired_url}")

        return ValidationResult(
            valid=result.get("valid", False),
            printer=printer,
            printer_type=result.get("printer_type", "unknown"),
            vertices=result.get("vertices", 0),
            faces=result.get("faces", 0),
            dimensions_mm=dimensions,
            volume_cm3=result.get("volume_cm3"),
            is_watertight=result.get("is_watertight", False),
            has_consistent_normals=result.get("has_consistent_normals", False),
            has_positive_volume=result.get("has_positive_volume", False),
            fits_build_volume=result.get("fits_build_volume", False),
            issues=result.get("issues", []),
            warnings=result.get("warnings", []),
            overhangs=overhang_info,
            repaired=result.get("repaired", False),
            repaired_model_url=repaired_url,
            repaired_model_base64=repaired_b64,
            repair_stats=result.get("repair_stats"),
            repairs_made=result.get("repairs_made", []),
        )

    except Exception as e:
        logger.exception("Error validating mesh")
        return ValidationResult(
            valid=False,
            printer=printer,
            printer_type="unknown",
            error=str(e),
        )


@mcp.tool()
async def list_printer_presets() -> dict:
    """
    List all available printer presets for validation.

    Returns a dictionary with printer names and their specifications
    including build volume, resolution, layer height, and printer type.

    Use this to see what printers are available for validate_for_printer.
    """
    # These match the PRINTER_PRESETS in gpu-worker/handler.py
    presets = {
        "anycubic_photon_m3": {
            "type": "resin",
            "build_mm": {"x": 180, "y": 164, "z": 102},
            "xy_um": 50,
            "layer_um": {"min": 10, "max": 150, "optimal": 50},
            "description": "Anycubic Photon M3 - Great for miniatures and fine details",
        },
        "anycubic_photon_m5s": {
            "type": "resin",
            "build_mm": {"x": 218, "y": 123, "z": 200},
            "xy_um": 19,
            "layer_um": {"min": 10, "max": 150, "optimal": 50},
            "description": "Anycubic Photon M5s - 14K resolution for ultra-fine details",
        },
        "elegoo_mars_3": {
            "type": "resin",
            "build_mm": {"x": 143, "y": 90, "z": 175},
            "xy_um": 35,
            "layer_um": {"min": 10, "max": 200, "optimal": 50},
            "description": "Elegoo Mars 3 - Popular entry-level resin printer",
        },
        "elegoo_saturn_3": {
            "type": "resin",
            "build_mm": {"x": 218, "y": 123, "z": 250},
            "xy_um": 19,
            "layer_um": {"min": 10, "max": 150, "optimal": 50},
            "description": "Elegoo Saturn 3 - Large format 12K resin printer",
        },
        "formlabs_form_3": {
            "type": "resin",
            "build_mm": {"x": 145, "y": 145, "z": 185},
            "xy_um": 25,
            "layer_um": {"min": 25, "max": 300, "optimal": 100},
            "description": "Formlabs Form 3 - Professional SLA printer",
        },
        "formlabs_form_4": {
            "type": "resin",
            "build_mm": {"x": 200, "y": 125, "z": 210},
            "xy_um": 50,
            "layer_um": {"min": 25, "max": 200, "optimal": 100},
            "description": "Formlabs Form 4 - Fast professional MSLA",
        },
        "prusa_mk4": {
            "type": "fdm",
            "build_mm": {"x": 250, "y": 210, "z": 220},
            "xy_um": 100,
            "layer_um": {"min": 50, "max": 300, "optimal": 200},
            "description": "Prusa MK4 - Reliable FDM workhorse",
        },
        "prusa_mini": {
            "type": "fdm",
            "build_mm": {"x": 180, "y": 180, "z": 180},
            "xy_um": 100,
            "layer_um": {"min": 50, "max": 300, "optimal": 150},
            "description": "Prusa Mini+ - Compact FDM printer",
        },
        "bambu_x1c": {
            "type": "fdm",
            "build_mm": {"x": 256, "y": 256, "z": 256},
            "xy_um": 100,
            "layer_um": {"min": 50, "max": 400, "optimal": 200},
            "description": "Bambu Lab X1C - High-speed enclosed FDM",
        },
        "bambu_a1_mini": {
            "type": "fdm",
            "build_mm": {"x": 180, "y": 180, "z": 180},
            "xy_um": 100,
            "layer_um": {"min": 50, "max": 400, "optimal": 200},
            "description": "Bambu Lab A1 Mini - Compact multicolor",
        },
        "creality_ender_3": {
            "type": "fdm",
            "build_mm": {"x": 220, "y": 220, "z": 250},
            "xy_um": 100,
            "layer_um": {"min": 100, "max": 400, "optimal": 200},
            "description": "Creality Ender 3 - Budget FDM classic",
        },
        "generic_resin": {
            "type": "resin",
            "build_mm": {"x": 150, "y": 100, "z": 150},
            "xy_um": 50,
            "layer_um": {"min": 25, "max": 100, "optimal": 50},
            "description": "Generic resin printer - Conservative settings",
        },
        "generic_fdm": {
            "type": "fdm",
            "build_mm": {"x": 200, "y": 200, "z": 200},
            "xy_um": 400,
            "layer_um": {"min": 100, "max": 400, "optimal": 200},
            "description": "Generic FDM printer - Standard settings",
        },
        "generic_sls": {
            "type": "sls",
            "build_mm": {"x": 160, "y": 160, "z": 320},
            "xy_um": 100,
            "layer_um": {"min": 80, "max": 150, "optimal": 100},
            "description": "Generic SLS printer - Powder-based",
        },
    }

    return {
        "printers": presets,
        "total": len(presets),
        "types": {
            "resin": [k for k, v in presets.items() if v["type"] == "resin"],
            "fdm": [k for k, v in presets.items() if v["type"] == "fdm"],
            "sls": [k for k, v in presets.items() if v["type"] == "sls"],
        },
    }


@mcp.tool()
async def generate_sla_supports(
    model_url: Optional[str] = None,
    model_base64: Optional[str] = None,
    overhang_angle: float = 45.0,
    support_density: float = 1.0,
    elevation_mm: float = 10.0,
    head_diameter_mm: float = 0.4,
    pillar_diameter_mm: float = 1.0,
    output_format: str = "stl",
    combine_with_model: bool = True,
) -> SupportGenerationResult:
    """
    Generate SLA supports for a 3D model.

    Automatically detects overhanging areas and generates tree-like
    support structures optimized for resin 3D printing.

    Args:
        model_url: URL of the 3D model file (STL, GLB, OBJ)
        model_base64: Base64 encoded model (alternative to URL)
        overhang_angle: Maximum overhang angle in degrees before supports needed (default: 45)
        support_density: Density multiplier for supports (0.5 = sparse, 1.0 = normal, 2.0 = dense)
        elevation_mm: Height to elevate model above build plate (default: 10mm)
        head_diameter_mm: Diameter of support tips touching the model (default: 0.4mm)
        pillar_diameter_mm: Diameter of support pillars (default: 1.0mm)
        output_format: Output format - stl, glb, or obj (default: stl)
        combine_with_model: If True, return combined model+supports mesh (default: True)

    Returns:
        SupportGenerationResult with support mesh URLs and statistics

    Example:
        >>> result = await generate_sla_supports(
        ...     model_url="https://example.com/figurine.stl",
        ...     overhang_angle=40.0,
        ...     support_density=1.2,
        ... )
        >>> print(f"Generated {result.num_supports} supports")
        >>> # Download combined model from result.combined_model_url
    """
    import time
    import trimesh
    import numpy as np

    start_time = time.time()

    if not model_url and not model_base64:
        return SupportGenerationResult(
            success=False,
            error="Either model_url or model_base64 must be provided",
        )

    try:
        # Load mesh
        temp_path = None
        try:
            if model_url:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(model_url) as resp:
                        if resp.status != 200:
                            return SupportGenerationResult(
                                success=False,
                                error=f"Failed to download model: HTTP {resp.status}",
                            )
                        mesh_bytes = await resp.read()
            else:
                mesh_bytes = base64.b64decode(model_base64)

            # Detect file type and load
            temp_path = Path(tempfile.mktemp(suffix=".glb"))
            temp_path.write_bytes(mesh_bytes)

            # Try loading with different formats
            mesh = None
            for fmt in ["glb", "stl", "obj", "ply"]:
                try:
                    mesh = trimesh.load(str(temp_path), file_type=fmt, force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                        if meshes:
                            mesh = trimesh.util.concatenate(meshes)
                        else:
                            mesh = None
                            continue
                    if mesh is not None and len(mesh.vertices) > 0:
                        break
                except Exception:
                    continue

            if mesh is None or len(mesh.vertices) == 0:
                return SupportGenerationResult(
                    success=False,
                    error="Failed to load mesh from provided data",
                )

        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)

        logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Configure support generation
        config = SupportTreeConfig(
            head_front_radius_mm=head_diameter_mm / 2,
            head_back_radius_mm=pillar_diameter_mm / 2,
            object_elevation_mm=elevation_mm,
        )

        # Elevate mesh
        mesh.vertices[:, 2] += elevation_mm

        # Generate supports
        logger.info(f"Generating supports with overhang_angle={overhang_angle}, density={support_density}")

        result = generate_supports_for_mesh(
            mesh_vertices=mesh.vertices,
            mesh_faces=mesh.faces,
            config=config,
            min_z=elevation_mm * 0.1,  # Start supports near the base
            overhang_angle=overhang_angle,
            density=support_density,
        )

        processing_time = time.time() - start_time

        if result.is_empty:
            logger.info("No supports needed for this model")
            return SupportGenerationResult(
                success=True,
                num_supports=0,
                num_heads=0,
                num_pillars=0,
                num_bridges=0,
                processing_time_seconds=round(processing_time, 2),
                overhang_angle_deg=overhang_angle,
                elevation_mm=elevation_mm,
            )

        logger.info(f"Generated {result.num_heads} heads, {result.num_pillars} pillars, {result.num_bridges} bridges")

        # Create support mesh
        support_mesh = result.to_trimesh()

        # Export support mesh
        support_buffer = io.BytesIO()
        support_mesh.export(support_buffer, file_type=output_format)
        support_bytes = support_buffer.getvalue()

        # Upload support mesh
        unique_id = str(uuid.uuid4())[:8]
        support_filename = f"supports_{unique_id}.{output_format}"
        support_url = await _upload_to_storage(support_bytes, support_filename)
        support_b64 = base64.b64encode(support_bytes).decode('utf-8')

        # Combine with model if requested
        combined_url = None
        combined_b64 = None

        if combine_with_model:
            combined_mesh = trimesh.util.concatenate([mesh, support_mesh])

            combined_buffer = io.BytesIO()
            combined_mesh.export(combined_buffer, file_type=output_format)
            combined_bytes = combined_buffer.getvalue()

            combined_filename = f"model_with_supports_{unique_id}.{output_format}"
            combined_url = await _upload_to_storage(combined_bytes, combined_filename)
            combined_b64 = base64.b64encode(combined_bytes).decode('utf-8')

        return SupportGenerationResult(
            success=True,
            num_supports=result.num_heads,
            num_heads=result.num_heads,
            num_pillars=result.num_pillars,
            num_bridges=result.num_bridges,
            supports_model_url=support_url,
            supports_model_base64=support_b64,
            combined_model_url=combined_url,
            combined_model_base64=combined_b64,
            support_vertices=len(support_mesh.vertices),
            support_faces=len(support_mesh.faces),
            processing_time_seconds=round(processing_time, 2),
            overhang_angle_deg=overhang_angle,
            elevation_mm=elevation_mm,
        )

    except Exception as e:
        logger.exception("Error generating supports")
        return SupportGenerationResult(
            success=False,
            processing_time_seconds=round(time.time() - start_time, 2),
            error=str(e),
        )


# ============================================================================
# Helper Functions
# ============================================================================

async def _convert_and_analyze(
    glb_bytes: bytes,
    target_format: str,
    target_size_mm: Optional[float] = None
) -> tuple[bytes, dict]:
    """Convert GLB to target format, optionally scale, and extract mesh info.

    Args:
        glb_bytes: Raw GLB model bytes from Hunyuan3D (or URL:... marker for R2 files)
        target_format: Output format (stl, glb, obj, ply)
        target_size_mm: If provided, scale the mesh so largest dimension equals this value in mm

    Returns:
        Tuple of (output_bytes, mesh_info_dict)
    """
    import trimesh
    import aiohttp

    # Check if glb_bytes is actually a URL marker (from R2 storage)
    if glb_bytes.startswith(b"URL:"):
        glb_url = glb_bytes[4:].decode("utf-8")
        logger.info(f"Downloading GLB from R2: {glb_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(glb_url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to download GLB from R2: {response.status}")
                glb_bytes = await response.read()
        logger.info(f"Downloaded {len(glb_bytes)} bytes from R2")

    # Load GLB
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        f.write(glb_bytes)
        temp_path = Path(f.name)

    try:
        loaded = trimesh.load(str(temp_path))

        # Handle scene
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                g for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh)
            ]
            if meshes:
                mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            else:
                mesh = loaded
        else:
            mesh = loaded

        # Scale to target size if requested
        if target_size_mm and hasattr(mesh, 'bounds') and hasattr(mesh, 'apply_scale'):
            bounds = mesh.bounds
            current_dimensions = bounds[1] - bounds[0]
            current_max = max(current_dimensions)
            if current_max > 0:
                scale_factor = target_size_mm / current_max
                mesh.apply_scale(scale_factor)
                logger.info(f"Scaled mesh by {scale_factor:.4f}x to target {target_size_mm}mm")

        # Center at origin and place on build plate (Z=0)
        if hasattr(mesh, 'bounds') and hasattr(mesh, 'apply_translation'):
            bounds = mesh.bounds
            center = (bounds[0] + bounds[1]) / 2
            # Move center to origin, then raise so bottom is at Z=0
            z_offset = -bounds[0][2]  # Raise by the negative of min Z
            mesh.apply_translation([-center[0], -center[1], z_offset])

        # Get info (after scaling)
        bounds = mesh.bounds if hasattr(mesh, 'bounds') else [[0, 0, 0], [1, 1, 1]]
        dimensions = bounds[1] - bounds[0] if hasattr(mesh, 'bounds') else [0, 0, 0]

        mesh_info = {
            "vertices": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
            "faces": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
            "dimensions": [float(d) for d in dimensions],
            "dimensions_mm": {
                "x": float(dimensions[0]),
                "y": float(dimensions[1]),
                "z": float(dimensions[2]),
            } if target_size_mm else None,
        }

        # Export
        if target_format == "glb":
            # Re-export GLB with scaling applied
            output_buffer = io.BytesIO()
            mesh.export(output_buffer, file_type="glb")
            return output_buffer.getvalue(), mesh_info

        output_buffer = io.BytesIO()
        mesh.export(output_buffer, file_type=target_format)
        return output_buffer.getvalue(), mesh_info

    finally:
        temp_path.unlink(missing_ok=True)


async def _upload_to_storage(data: bytes, filename: str) -> str:
    """Upload file to Apify Key-Value Store and return URL."""
    try:
        from apify import Actor

        # Save to default KV store
        await Actor.set_value(filename, data, content_type="application/octet-stream")

        # Get store ID from environment variable (works in Standby mode)
        store_id = os.environ.get("APIFY_DEFAULT_KEY_VALUE_STORE_ID")
        if not store_id:
            # Fallback: try Actor.config if available
            try:
                store_id = Actor.config.default_key_value_store_id
            except:
                pass

        if store_id:
            return f"https://api.apify.com/v2/key-value-stores/{store_id}/records/{filename}"
        else:
            # Fallback to base64 if we can't get the store ID
            logger.warning("Could not get Key-Value Store ID, returning base64")
            return f"data:application/octet-stream;base64,{base64.b64encode(data).decode()}"

    except Exception as e:
        logger.warning(f"Could not upload to Apify storage: {e}")
        # Return base64 as fallback
        return f"data:application/octet-stream;base64,{base64.b64encode(data).decode()}"


# ============================================================================
# Server Entry Point
# ============================================================================

async def run_mcp_server_async():
    """Run the MCP server for Apify Standby mode (async version).

    Use this when calling from an async context (e.g., asyncio.run(main())).
    Uses Streamable HTTP transport which exposes /mcp endpoint.
    """
    logger.info(f"Starting MCP Server on {_host}:{_port}")
    logger.info(f"ACTOR_STANDBY_PORT: {os.environ.get('ACTOR_STANDBY_PORT', 'not set')}")
    logger.info(f"APIFY_META_ORIGIN: {os.environ.get('APIFY_META_ORIGIN', 'not set')}")

    # Use run_streamable_http_async() for Apify Standby mode
    # This exposes the /mcp endpoint that Apify MCP clients expect
    await mcp.run_streamable_http_async()


def run_mcp_server():
    """Run the MCP server for Apify Standby mode (sync version).

    Use this when calling from a synchronous context.
    """
    logger.info(f"Starting MCP Server on {_host}:{_port}")
    logger.info(f"ACTOR_STANDBY_PORT: {os.environ.get('ACTOR_STANDBY_PORT', 'not set')}")
    logger.info(f"APIFY_META_ORIGIN: {os.environ.get('APIFY_META_ORIGIN', 'not set')}")

    # Run with streamable-http transport for /mcp endpoint
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_mcp_server()
