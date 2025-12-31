"""RunPod Serverless client for Hunyuan3D generation."""

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp

from src.core.config import get_settings
from src.core.constants import Quality, InputType, QUALITY_SETTINGS
from src.core.exceptions import (
    RunPodError,
    RunPodTimeoutError,
    InvalidImageError,
)

logger = logging.getLogger(__name__)


class RunPodClient:
    """Async client for RunPod Serverless API with Hunyuan3D."""

    BASE_URL = "https://api.runpod.ai/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.runpod_api_key
        self.endpoint_id = endpoint_id or settings.runpod_endpoint_id
        self.timeout = timeout or settings.runpod_timeout

        if not self.api_key:
            raise RunPodError("RunPod API key not configured", "MISSING_API_KEY")
        if not self.endpoint_id:
            raise RunPodError("RunPod endpoint ID not configured", "MISSING_ENDPOINT")

        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def headers(self) -> dict:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.headers)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "RunPodClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def run_sync(
        self,
        input_type: InputType,
        input_data: str,
        quality: Quality = Quality.STANDARD,
    ) -> bytes:
        """
        Run Hunyuan3D generation synchronously (wait for result).

        Args:
            input_type: Type of input (text, image_url, image_base64)
            input_data: The prompt text, image URL, or base64 data
            quality: Generation quality level

        Returns:
            GLB file bytes

        Raises:
            RunPodError: If generation fails
            RunPodTimeoutError: If generation times out
        """
        job_id = await self._submit_job(input_type, input_data, quality)
        return await self._wait_for_result(job_id)

    async def run_async(
        self,
        input_type: InputType,
        input_data: str,
        quality: Quality = Quality.STANDARD,
    ) -> str:
        """
        Submit Hunyuan3D generation job and return job ID.

        Args:
            input_type: Type of input (text, image_url, image_base64)
            input_data: The prompt text, image URL, or base64 data
            quality: Generation quality level

        Returns:
            Job ID for polling

        Raises:
            RunPodError: If submission fails
        """
        return await self._submit_job(input_type, input_data, quality)

    async def get_status(self, job_id: str) -> dict:
        """
        Get status of a running job.

        Returns:
            Status dict with 'status' and optionally 'output' or 'error'
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}/{self.endpoint_id}/status/{job_id}"

        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise RunPodError(f"Status check failed: {text}")
            return await response.json()

    async def get_result(self, job_id: str) -> bytes:
        """
        Get result of a completed job.

        Returns:
            GLB file bytes
        """
        status = await self.get_status(job_id)

        if status.get("status") == "FAILED":
            error = status.get("error", "Unknown error")
            raise RunPodError(f"Job failed: {error}", "JOB_FAILED")

        if status.get("status") != "COMPLETED":
            raise RunPodError(
                f"Job not completed. Status: {status.get('status')}",
                "JOB_NOT_COMPLETED"
            )

        output = status.get("output", {})
        return self._extract_glb_from_output(output)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/{self.endpoint_id}/cancel/{job_id}"

        async with session.post(url) as response:
            return response.status == 200

    async def _submit_job(
        self,
        input_type: InputType,
        input_data: str,
        quality: Quality,
    ) -> str:
        """Submit a generation job to RunPod."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/{self.endpoint_id}/run"

        # Build input payload for Hunyuan3D
        payload = self._build_payload(input_type, input_data, quality)

        logger.info(f"Submitting {quality.value} quality job to RunPod")
        logger.debug(f"Payload: {payload}")

        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise RunPodError(f"Job submission failed: {text}")

            data = await response.json()
            job_id = data.get("id")

            if not job_id:
                raise RunPodError("No job ID returned from RunPod")

            logger.info(f"Job submitted: {job_id}")
            return job_id

    async def _wait_for_result(self, job_id: str) -> bytes:
        """Poll for job completion and return result."""
        start_time = time.time()
        poll_interval = 2.0  # Start with 2 second polls

        while True:
            elapsed = time.time() - start_time

            if elapsed > self.timeout:
                await self.cancel(job_id)
                raise RunPodTimeoutError(
                    f"Generation timed out after {self.timeout}s"
                )

            status = await self.get_status(job_id)
            job_status = status.get("status")

            logger.debug(f"Job {job_id} status: {job_status} ({elapsed:.0f}s)")

            if job_status == "COMPLETED":
                output = status.get("output", {})
                return self._extract_glb_from_output(output)

            if job_status == "FAILED":
                error = status.get("error", "Unknown error")
                raise RunPodError(f"Generation failed: {error}", "GENERATION_FAILED")

            if job_status == "CANCELLED":
                raise RunPodError("Job was cancelled", "JOB_CANCELLED")

            # Adaptive polling - slower polls as time goes on
            if elapsed > 60:
                poll_interval = 5.0
            elif elapsed > 30:
                poll_interval = 3.0

            await asyncio.sleep(poll_interval)

    def _build_payload(
        self,
        input_type: InputType,
        input_data: str,
        quality: Quality,
    ) -> dict:
        """Build RunPod job payload for Hunyuan3D."""
        quality_settings = QUALITY_SETTINGS[quality]

        payload = {
            "input": {
                "steps": quality_settings["steps"],
                "guidance_scale": quality_settings["guidance_scale"],
                "dual_guidance_scale": quality_settings.get("dual_guidance_scale", 10.5),
                "octree_resolution": quality_settings["octree_resolution"],
                "num_chunks": quality_settings.get("num_chunks", 10000),
                "output_format": "glb",
            }
        }

        if input_type == InputType.TEXT:
            payload["input"]["prompt"] = input_data
        elif input_type == InputType.IMAGE_URL:
            payload["input"]["image_url"] = input_data
        elif input_type == InputType.IMAGE_BASE64:
            # Ensure proper base64 format
            if not input_data.startswith("data:"):
                input_data = f"data:image/png;base64,{input_data}"
            payload["input"]["image_base64"] = input_data

        return payload

    def _extract_glb_from_output(self, output: dict) -> bytes:
        """Extract GLB bytes from RunPod output."""
        # Output could be base64 encoded GLB or a URL (from R2 storage)
        if "glb_base64" in output:
            return base64.b64decode(output["glb_base64"])

        if "glb_url" in output:
            # GLB stored in Cloudflare R2 - return URL as special marker
            # The caller will handle downloading or returning the URL
            glb_url = output["glb_url"]
            logger.info(f"GLB stored in R2: {glb_url}")
            # Return URL encoded as bytes with marker prefix
            return f"URL:{glb_url}".encode("utf-8")

        if "model" in output:
            # Alternative output format
            return base64.b64decode(output["model"])

        raise RunPodError(
            "No GLB data found in output",
            "NO_GLB_OUTPUT"
        )

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    async def validate_mesh(
        self,
        mesh_url: Optional[str] = None,
        mesh_base64: Optional[str] = None,
        printer: str = "generic_resin",
        target_size_mm: Optional[float] = None,
        check_overhangs: bool = True,
        auto_repair: bool = False,
        custom_build_volume: Optional[dict] = None,
    ) -> dict:
        """
        Validate a 3D mesh for printability on a specific printer.

        Args:
            mesh_url: URL of mesh file - RunPod downloads directly (preferred, more efficient)
            mesh_base64: Base64-encoded mesh file (fallback, less efficient for large files)
            printer: Printer preset name (e.g., "anycubic_photon_m3", "prusa_mk4")
            target_size_mm: Scale model to this size (longest dimension)
            check_overhangs: Whether to detect overhangs
            auto_repair: Whether to attempt automatic repair
            custom_build_volume: Custom build volume {"x": mm, "y": mm, "z": mm}

        Returns:
            Validation result dict with:
            - valid: bool
            - issues: list of issues found
            - warnings: list of warnings
            - mesh_info: dict with vertices, faces, dimensions
            - overhangs: dict with overhang analysis (if check_overhangs=True)
            - repaired_mesh_base64: str (if auto_repair=True and repairs made)

        Raises:
            RunPodError: If validation fails
            RunPodTimeoutError: If validation times out
        """
        if not mesh_url and not mesh_base64:
            raise RunPodError("Either mesh_url or mesh_base64 must be provided")

        job_id = await self._submit_validation_job(
            mesh_url=mesh_url,
            mesh_base64=mesh_base64,
            printer=printer,
            target_size_mm=target_size_mm,
            check_overhangs=check_overhangs,
            auto_repair=auto_repair,
            custom_build_volume=custom_build_volume,
        )
        return await self._wait_for_validation_result(job_id)

    async def _submit_validation_job(
        self,
        mesh_url: Optional[str],
        mesh_base64: Optional[str],
        printer: str,
        target_size_mm: Optional[float],
        check_overhangs: bool,
        auto_repair: bool,
        custom_build_volume: Optional[dict],
    ) -> str:
        """Submit a validation job to RunPod."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/{self.endpoint_id}/run"

        payload = {
            "input": {
                "action": "validate",
                "printer": printer,
                "check_overhangs": check_overhangs,
                "auto_repair": auto_repair,
            }
        }

        # Prefer URL over base64 for efficiency (avoids large JSON payloads)
        if mesh_url:
            payload["input"]["mesh_url"] = mesh_url
            logger.info(f"Using mesh_url for validation (efficient)")
        elif mesh_base64:
            payload["input"]["mesh_base64"] = mesh_base64
            logger.info(f"Using mesh_base64 for validation (fallback)")

        if target_size_mm is not None:
            payload["input"]["target_size_mm"] = target_size_mm

        if custom_build_volume is not None:
            payload["input"]["build_volume_mm"] = custom_build_volume

        logger.info(f"Submitting validation job for printer: {printer}")
        logger.debug(f"Validation payload keys: {list(payload['input'].keys())}")

        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise RunPodError(f"Validation job submission failed: {text}")

            data = await response.json()
            job_id = data.get("id")

            if not job_id:
                raise RunPodError("No job ID returned from RunPod")

            logger.info(f"Validation job submitted: {job_id}")
            return job_id

    async def _wait_for_validation_result(self, job_id: str) -> dict:
        """Poll for validation job completion and return result."""
        start_time = time.time()
        poll_interval = 1.0  # Validation is faster, poll more frequently

        while True:
            elapsed = time.time() - start_time

            # Validation timeout: 5 minutes to allow for large file downloads (132MB+)
            # The GPU worker needs time to: download from R2, load mesh, validate
            validation_timeout = min(self.timeout, 300)  # Max 5 minutes for validation

            if elapsed > validation_timeout:
                await self.cancel(job_id)
                raise RunPodTimeoutError(
                    f"Validation timed out after {validation_timeout}s"
                )

            status = await self.get_status(job_id)
            job_status = status.get("status")

            logger.debug(f"Validation job {job_id} status: {job_status} ({elapsed:.0f}s)")

            if job_status == "COMPLETED":
                output = status.get("output", {})
                return output  # Validation returns dict directly

            if job_status == "FAILED":
                error = status.get("error", "Unknown error")
                raise RunPodError(f"Validation failed: {error}", "VALIDATION_FAILED")

            if job_status == "CANCELLED":
                raise RunPodError("Validation job was cancelled", "JOB_CANCELLED")

            # Adaptive polling
            if elapsed > 30:
                poll_interval = 2.0

            await asyncio.sleep(poll_interval)


async def generate_3d_model(
    prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    quality: Quality = Quality.STANDARD,
    output_path: Optional[Path] = None,
) -> tuple[bytes, Path]:
    """
    High-level function to generate a 3D model.

    Args:
        prompt: Text description (mutually exclusive with image args)
        image_url: URL of image to convert
        image_base64: Base64-encoded image
        quality: Generation quality
        output_path: Optional path to save GLB file

    Returns:
        Tuple of (GLB bytes, path where saved)

    Raises:
        InvalidImageError: If image input is invalid
        RunPodError: If generation fails
    """
    # Determine input type
    if image_base64:
        input_type = InputType.IMAGE_BASE64
        input_data = image_base64
    elif image_url:
        input_type = InputType.IMAGE_URL
        input_data = image_url
    elif prompt:
        input_type = InputType.TEXT
        input_data = prompt
    else:
        raise InvalidImageError("No input provided (prompt, image_url, or image_base64)")

    # Generate model
    async with RunPodClient() as client:
        glb_bytes = await client.run_sync(input_type, input_data, quality)

    # Save to file
    if output_path is None:
        settings = get_settings()
        output_path = Path(settings.temp_dir) / "generated.glb"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)

    logger.info(f"Generated model saved to {output_path} ({len(glb_bytes)} bytes)")
    return glb_bytes, output_path
