"""HTTP route for AI cover art generation using SDXL Turbo."""

from __future__ import annotations

import asyncio
import os
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from loguru import logger


def register_cover_art_routes(
    app: FastAPI,
    *,
    get_project_root: Callable[[], str],
) -> None:
    """Register the ``/v1/cover_art/generate`` endpoint."""

    @app.post("/v1/cover_art/generate")
    async def generate_cover_art(request: Request):
        """Generate album cover art for a song.

        Request body:
            title:   Song title
            style:   Genre/mood tags
            lyrics:  Song lyrics (used for theme extraction)
            song_id: Unique song identifier (used for filename)
        """
        body = await request.json()
        title = body.get("title", "")
        style = body.get("style", "")
        lyrics = body.get("lyrics", "")
        song_id = body.get("song_id", "")

        if not song_id:
            raise HTTPException(400, "song_id is required")

        if not title and not style:
            raise HTTPException(400, "At least title or style is required")

        # Output path — save into the temp_audio_dir so /v1/audio can serve it
        output_dir = getattr(request.app.state, 'temp_audio_dir', None)
        if not output_dir:
            # Fallback to output/ under project root
            output_dir = os.path.join(get_project_root(), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{song_id}_cover.webp")

        def _do_generate():
            from acestep.core.cover_art import get_generator

            generator = get_generator()
            return generator.generate(
                title=title,
                style=style,
                lyrics=lyrics,
                output_path=output_path,
            )

        try:
            logger.info(f"[CoverArt] Generating cover for song_id={song_id}")
            result_path = await asyncio.get_event_loop().run_in_executor(
                None, _do_generate
            )

            if result_path is None:
                raise HTTPException(500, "Cover art generation failed")

            # Return a URL path that can be served via the existing /v1/audio endpoint
            cover_url = f"/v1/audio?path={result_path}"
            logger.info(f"[CoverArt] Done: {cover_url}")
            return {"cover_url": cover_url}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[CoverArt] Generation failed: {e}", exc_info=True)
            raise HTTPException(500, f"Cover art generation failed: {e}")
