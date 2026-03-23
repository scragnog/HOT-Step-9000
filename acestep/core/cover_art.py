"""AI-generated cover art using SDXL Turbo.

Provides a :class:`CoverArtGenerator` that lazy-loads ``stabilityai/sdxl-turbo``
via ``diffusers``, generates a 512×512 image from song metadata (title, style,
lyrics), and saves it as a WebP file.

The model is loaded on first use and unloaded after generation to free VRAM for
the audio pipeline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from loguru import logger

# Default HuggingFace model identifier
_DEFAULT_MODEL_ID = "stabilityai/sdxl-turbo"


def _extract_theme_keywords(lyrics: str, max_keywords: int = 5) -> list[str]:
    """Extract the most common meaningful words from lyrics for prompt enrichment.

    Strips section headers like ``[Verse 1]``, stop-words, and short words,
    then returns the *max_keywords* most frequent tokens.
    """
    if not lyrics or not lyrics.strip():
        return []

    # Strip section headers
    cleaned = re.sub(r"\[.*?\]", "", lyrics)
    # Remove punctuation, lowercase
    cleaned = re.sub(r"[^\w\s]", "", cleaned.lower())

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "its", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "not", "no", "so", "if", "up", "out", "just", "like", "my", "me",
        "we", "you", "your", "they", "them", "he", "she", "her", "his",
        "i", "im", "ive", "dont", "that", "this", "all", "got", "get",
        "when", "what", "where", "how", "why", "oh", "yeah", "ya", "na",
        "la", "da", "uh", "ah", "ooh", "hey", "go", "know", "come", "take",
        "make", "see", "let", "say", "one", "way", "say", "back", "now",
        "more", "than", "into", "over", "down", "been",
    }

    words = cleaned.split()
    # Filter short words and stop words
    words = [w for w in words if len(w) > 3 and w not in stop_words]

    if not words:
        return []

    # Count frequencies
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Sort by frequency, take top N
    sorted_words = sorted(freq, key=freq.get, reverse=True)  # type: ignore[arg-type]
    return sorted_words[:max_keywords]


def _build_prompt(title: str, style: str, lyrics: str) -> str:
    """Build a text-to-image prompt from song metadata.

    Keeps total prompt under ~60 words to stay within CLIP's 77-token limit.
    """
    parts = ["Album cover art"]

    if style:
        # Extract just the genre/mood keywords, not full production descriptions
        style_words = style.strip().split(",")
        # Take first 3 comma-separated phrases, truncate each
        short_style = ", ".join(w.strip() for w in style_words[:3] if w.strip())
        if short_style:
            parts.append(f"for a {short_style} song")

    if title:
        # Strip artist prefix if present (e.g. "Artist - Title" -> "Title")
        clean_title = title.strip()
        if " - " in clean_title:
            clean_title = clean_title.split(" - ", 1)[1].strip()
        parts.append(f'called "{clean_title}"')

    # Extract theme keywords from lyrics
    keywords = _extract_theme_keywords(lyrics, max_keywords=4)
    if keywords:
        parts.append(f"themes of {', '.join(keywords)}")

    # Style guidance
    parts.append("digital art, vibrant, professional album artwork")

    prompt = ", ".join(parts)
    logger.info(f"[CoverArt] Prompt: {prompt}")
    return prompt


class CoverArtGenerator:
    """Generates album cover art using SDXL Turbo."""

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID):
        self.model_id = model_id
        self._pipeline = None

    def _load_pipeline(self, device: str = "cuda"):
        """Lazy-load the SDXL Turbo pipeline."""
        import torch
        from diffusers import AutoPipelineForText2Image

        logger.info(f"[CoverArt] Loading {self.model_id} on {device}...")

        dtype = torch.float16 if device == "cuda" else torch.float32

        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
        )
        self._pipeline.to(device)

        # Disable safety checker for local use (album art, not user-facing content moderation)
        if hasattr(self._pipeline, "safety_checker"):
            self._pipeline.safety_checker = None

        logger.info(f"[CoverArt] Model loaded successfully on {device}")

    def _unload_pipeline(self):
        """Unload the pipeline and free VRAM."""
        import torch

        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[CoverArt] Pipeline unloaded, VRAM freed")

    def generate(
        self,
        title: str,
        style: str,
        lyrics: str,
        output_path: str,
        width: int = 512,
        height: int = 512,
    ) -> Optional[str]:
        """Generate a cover art image and save it.

        Args:
            title: Song title.
            style: Style/genre tags.
            lyrics: Song lyrics.
            output_path: Where to save the image (should end in ``.webp``).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            The *output_path* on success, or ``None`` on failure.
        """
        import torch

        prompt = _build_prompt(title, style, lyrics)

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._load_pipeline(device)

            logger.info(f"[CoverArt] Generating {width}x{height} image...")

            image = self._pipeline(  # type: ignore[misc]
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0,  # SDXL Turbo is distilled — no CFG needed
                width=width,
                height=height,
            ).images[0]

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as WebP
            image.save(output_path, format="WEBP", quality=85)
            logger.info(f"[CoverArt] Saved to {output_path}")

            return output_path

        except torch.cuda.OutOfMemoryError:
            logger.warning("[CoverArt] CUDA OOM — retrying on CPU")
            self._unload_pipeline()
            try:
                self._load_pipeline("cpu")
                image = self._pipeline(  # type: ignore[misc]
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    width=width,
                    height=height,
                ).images[0]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path, format="WEBP", quality=85)
                logger.info(f"[CoverArt] Saved (CPU fallback) to {output_path}")
                return output_path
            except Exception as cpu_err:
                logger.error(f"[CoverArt] CPU fallback failed: {cpu_err}")
                return None

        except Exception as e:
            logger.error(f"[CoverArt] Generation failed: {e}")
            return None

        finally:
            self._unload_pipeline()


# Module-level singleton (created on first use)
_generator: Optional[CoverArtGenerator] = None


def get_generator() -> CoverArtGenerator:
    """Return the module-level :class:`CoverArtGenerator` singleton."""
    global _generator
    if _generator is None:
        _generator = CoverArtGenerator()
    return _generator
