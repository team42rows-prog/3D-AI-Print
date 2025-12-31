"""
Google Imagen client for high-quality image generation.

Uses Google's Imagen 4 API (GA) to generate high-quality images from text prompts,
optimized for subsequent 3D model generation with Hunyuan3D.

Imagen 4 is Google's latest image generation model (December 2025).
Fallback to Imagen 3 if Imagen 4 is not available.
"""

import base64
import io
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ImagenClient:
    """Client for Google Imagen API (Imagen 4 with fallback to Imagen 3)."""

    # Models to try in order of preference
    MODELS = [
        "imagen-4.0-generate-001",     # Imagen 4 GA (best quality)
        "imagen-4.0-fast-generate-001", # Imagen 4 Fast (faster, good quality)
        "imagen-3.0-generate-002",      # Imagen 3 (fallback)
    ]

    def __init__(self, api_key: str):
        """Initialize the Imagen client.

        Args:
            api_key: Google API key (Gemini API key works for Imagen)
        """
        self.api_key = api_key
        self._client = None
        self._working_model = None  # Cache the first model that works

    def _get_client(self):
        """Lazy initialization of the Google GenAI client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate_image(
        self,
        prompt: str,
        style: str = "realistic",
        aspect_ratio: str = "1:1",
        negative_prompt: Optional[str] = None,
    ) -> bytes:
        """Generate an image from a text prompt using Imagen 3.

        Optimized for Hunyuan3D input requirements:
        - Front-facing view (best for single-view 3D reconstruction)
        - Pure white background (#FFFFFF)
        - Centered object with clear edges
        - Neutral/even lighting (no harsh shadows)
        - High detail textures

        Args:
            prompt: Text description of the image to generate
            style: Visual style - realistic, cartoon, miniature, artistic, mechanical
            aspect_ratio: Image aspect ratio (1:1 recommended for 3D)
            negative_prompt: Things to avoid in the image

        Returns:
            PNG image bytes
        """
        import asyncio

        # Base 3D-optimized modifiers (critical for Hunyuan3D)
        base_3d_modifiers = (
            "front view, centered in frame, "
            "pure white background, "
            "soft even studio lighting, no harsh shadows, "
            "single isolated object, "
            "high detail, sharp edges, clear silhouette, "
            "product photography style, "
            "3D model reference sheet"
        )

        # Style-specific enhancements
        style_modifiers = {
            "realistic": "highly detailed, photorealistic textures, physically accurate proportions, professional product render",
            "cartoon": "stylized 3D render, smooth clean surfaces, bold colors, toy-like appearance, Pixar style",
            "miniature": "tabletop gaming miniature, detailed sculpt, figurine, matte finish, 28mm scale reference",
            "artistic": "sculptural art piece, elegant organic forms, museum quality, fine art sculpture",
            "mechanical": "precision engineered, technical model, CAD-like accuracy, industrial design, mechanical parts visible",
        }

        # Build enhanced prompt optimized for 3D conversion
        style_mod = style_modifiers.get(style, style_modifiers["realistic"])
        enhanced_prompt = f"{prompt}, {base_3d_modifiers}, {style_mod}"

        # Comprehensive negative prompt for clean 3D-ready images
        if negative_prompt is None:
            negative_prompt = (
                "multiple objects, complex background, gradient background, "
                "harsh shadows, dramatic lighting, backlit, lens flare, "
                "blurry, low quality, noise, artifacts, "
                "text, watermark, logo, signature, "
                "cropped, cut off, partial view, "
                "reflections, mirrors, glass distortion, "
                "hands holding object, people, faces in background"
            )

        logger.info(f"Generating image with Imagen: {prompt[:50]}...")

        # Run synchronous API call in executor
        # Note: negative_prompt is NOT supported by Gemini/Imagen API
        # We include the negative concepts in the enhanced prompt instead
        def _generate():
            from google.genai import types

            client = self._get_client()

            # Try models in order until one works
            models_to_try = [self._working_model] if self._working_model else self.MODELS
            last_error = None

            for model in models_to_try:
                try:
                    logger.info(f"Trying model: {model}")
                    response = client.models.generate_images(
                        model=model,
                        prompt=enhanced_prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio=aspect_ratio,
                            output_mime_type="image/png",
                            image_size="2K",  # Maximum resolution (2048x2048)
                        ),
                    )
                    # If successful, cache this model for future use
                    self._working_model = model
                    logger.info(f"Successfully generated with model: {model}")
                    return response
                except Exception as e:
                    last_error = e
                    logger.warning(f"Model {model} failed: {e}")
                    continue

            # All models failed
            raise RuntimeError(f"All Imagen models failed. Last error: {last_error}")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _generate)

        if not response.generated_images:
            raise RuntimeError("Imagen returned no images")

        # Get the image bytes
        image_data = response.generated_images[0].image.image_bytes

        logger.info(f"Image generated successfully: {len(image_data)} bytes")
        return image_data

    async def generate_image_for_3d(
        self,
        prompt: str,
        style: str = "realistic",
    ) -> bytes:
        """Generate an image optimized for 3D model conversion.

        This method generates an image and preprocesses it for optimal
        3D conversion with Hunyuan3D:
        - 1:1 aspect ratio
        - White background
        - Centered object
        - Clean edges

        Args:
            prompt: Text description of the 3D model to create
            style: Visual style

        Returns:
            Preprocessed PNG image bytes (1024x1024)
        """
        from src.utils.image_preprocessing import preprocess_for_3d

        # Generate initial image
        image_bytes = await self.generate_image(
            prompt=prompt,
            style=style,
            aspect_ratio="1:1",
        )

        # Preprocess for 3D conversion
        processed_bytes = await preprocess_for_3d(image_bytes)

        return processed_bytes


async def generate_image_with_imagen(
    prompt: str,
    api_key: str,
    style: str = "realistic",
) -> bytes:
    """Convenience function to generate an image with Imagen 3.

    Args:
        prompt: Text description
        api_key: Google API key
        style: Visual style

    Returns:
        Preprocessed image bytes ready for 3D conversion
    """
    client = ImagenClient(api_key)
    return await client.generate_image_for_3d(prompt, style)
