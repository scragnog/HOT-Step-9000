"""
Lireek lyrics management API routes.

Mounted at ``/api/lireek/`` — provides endpoints for the full lyrics
pipeline: Genius fetch → profiling → generation → refinement → export.
"""

import json
import logging
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from acestep.api.lireek.schemas import (
    LyricsSearchRequest,
    BuildProfileRequest,
    GenerateFromProfileRequest,
    RefineRequest,
    AlbumPresetUpdate,
)

logger = logging.getLogger(__name__)


# ── Request models ────────────────────────────────────────────────────────────

class FetchLyricsRequest(BaseModel):
    artist: str = Field(..., description="Artist name or Genius URL")
    album: Optional[str] = Field(None, description="Album name or Genius album URL")
    max_songs: int = Field(10, ge=1, le=50)


class GenerateRequest(BaseModel):
    profile_id: int
    provider: str = "gemini"
    model: Optional[str] = None
    extra_instructions: Optional[str] = None


class RefineGenerationRequest(BaseModel):
    provider: str = "gemini"
    model: Optional[str] = None


class UpdateMetadataRequest(BaseModel):
    title: Optional[str] = None
    caption: Optional[str] = None
    bpm: Optional[int] = None
    key: Optional[str] = None
    duration: Optional[int] = None
    subject: Optional[str] = None


class SlopScanRequest(BaseModel):
    text: str


# ── Route registration ───────────────────────────────────────────────────────

def register_lireek_routes(app: FastAPI) -> None:
    """Register all Lireek lyrics management routes."""

    # ── Artists ────────────────────────────────────────────────────────────

    @app.get("/api/lireek/artists")
    async def list_artists():
        from acestep.api.lireek.lireek_db import list_artists
        return {"artists": list_artists()}

    @app.delete("/api/lireek/artists/{artist_id}")
    async def delete_artist(artist_id: int):
        from acestep.api.lireek.lireek_db import delete_artist
        if not delete_artist(artist_id):
            raise HTTPException(status_code=404, detail="Artist not found")
        return {"deleted": True}

    # ── Lyrics Sets ───────────────────────────────────────────────────────

    @app.get("/api/lireek/lyrics-sets")
    async def list_lyrics_sets(artist_id: Optional[int] = None):
        from acestep.api.lireek.lireek_db import get_lyrics_sets
        return {"lyrics_sets": get_lyrics_sets(artist_id)}

    @app.get("/api/lireek/lyrics-sets/{lyrics_set_id}")
    async def get_lyrics_set(lyrics_set_id: int):
        from acestep.api.lireek.lireek_db import get_lyrics_set
        ls = get_lyrics_set(lyrics_set_id)
        if not ls:
            raise HTTPException(status_code=404, detail="Lyrics set not found")
        return ls

    @app.delete("/api/lireek/lyrics-sets/{lyrics_set_id}")
    async def delete_lyrics_set(lyrics_set_id: int):
        from acestep.api.lireek.lireek_db import delete_lyrics_set
        if not delete_lyrics_set(lyrics_set_id):
            raise HTTPException(status_code=404, detail="Lyrics set not found")
        return {"deleted": True}

    @app.delete("/api/lireek/lyrics-sets/{lyrics_set_id}/songs/{song_index}")
    async def remove_song(lyrics_set_id: int, song_index: int):
        from acestep.api.lireek.lireek_db import remove_song_from_set
        result = remove_song_from_set(lyrics_set_id, song_index)
        if not result:
            raise HTTPException(status_code=404, detail="Song or lyrics set not found")
        return result

    # ── Genius Fetch ──────────────────────────────────────────────────────

    @app.post("/api/lireek/fetch-lyrics")
    async def fetch_lyrics_endpoint(req: FetchLyricsRequest):
        """Fetch lyrics from Genius and store in the database."""
        from acestep.api.lireek.genius_service import fetch_lyrics
        from acestep.api.lireek.lireek_db import (
            get_or_create_artist, save_lyrics_set,
        )

        logger.info("fetch-lyrics request: artist=%r, album=%r, max_songs=%d",
                     req.artist, req.album, req.max_songs)

        try:
            result = fetch_lyrics(
                artist_name=req.artist,
                album_name=req.album,
                max_songs=req.max_songs,
            )
        except ValueError as e:
            logger.warning("fetch_lyrics ValueError: %s", e)
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error("Genius fetch failed: %s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        logger.info("fetch_lyrics returned %d songs for artist=%r album=%r",
                     len(result.songs), result.artist, result.album)

        # Store in DB
        artist = get_or_create_artist(result.artist)
        songs_data = [s.model_dump() for s in result.songs]
        logger.info("Saving %d songs to DB for artist_id=%d",
                     len(songs_data), artist["id"])
        lyrics_set = save_lyrics_set(
            artist_id=artist["id"],
            album=result.album,
            songs=songs_data,
            max_songs=req.max_songs,
        )

        logger.info("Saved lyrics_set id=%d with total_songs=%d",
                     lyrics_set["id"], lyrics_set["total_songs"])

        return {
            "artist": artist,
            "lyrics_set": lyrics_set,
            "songs_fetched": len(result.songs),
        }

    # ── Profiles ──────────────────────────────────────────────────────────

    @app.get("/api/lireek/profiles")
    async def list_profiles(lyrics_set_id: Optional[int] = None):
        from acestep.api.lireek.lireek_db import get_profiles
        return {"profiles": get_profiles(lyrics_set_id)}

    @app.get("/api/lireek/profiles/{profile_id}")
    async def get_profile(profile_id: int):
        from acestep.api.lireek.lireek_db import get_profile
        profile = get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile

    @app.delete("/api/lireek/profiles/{profile_id}")
    async def delete_profile(profile_id: int):
        from acestep.api.lireek.lireek_db import delete_profile
        if not delete_profile(profile_id):
            raise HTTPException(status_code=404, detail="Profile not found")
        return {"deleted": True}

    @app.post("/api/lireek/lyrics-sets/{lyrics_set_id}/build-profile")
    async def build_profile_endpoint(lyrics_set_id: int, req: BuildProfileRequest):
        """Build an artist profile from a lyrics set using LLM analysis."""
        from acestep.api.lireek.lireek_db import get_lyrics_set, save_profile
        from acestep.api.lireek.schemas import SongLyrics
        from acestep.api.lireek.profiler_service import build_profile
        from acestep.api.lireek.generation_service import make_llm_caller

        ls = get_lyrics_set(lyrics_set_id)
        if not ls:
            raise HTTPException(status_code=404, detail="Lyrics set not found")

        songs_raw = ls.get("songs") or []
        if isinstance(songs_raw, str):
            songs_raw = json.loads(songs_raw)
        songs = [SongLyrics(**s) for s in songs_raw]

        if not songs:
            raise HTTPException(status_code=400, detail="Lyrics set has no songs")

        llm_caller = make_llm_caller(req.provider, req.model)

        try:
            profile = build_profile(
                artist=ls.get("artist_name", "Unknown"),
                album=ls.get("album"),
                songs=songs,
                llm_call=llm_caller,
            )
        except Exception as e:
            logger.error("Profile build failed: %s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        saved = save_profile(
            lyrics_set_id=lyrics_set_id,
            provider=req.provider,
            model=req.model or "",
            profile_data=profile.model_dump(),
        )
        return saved

    # ── Generations ───────────────────────────────────────────────────────

    @app.get("/api/lireek/generations")
    async def list_generations(profile_id: Optional[int] = None):
        from acestep.api.lireek.lireek_db import get_generations
        return {"generations": get_generations(profile_id=profile_id)}

    @app.get("/api/lireek/generations/all")
    async def list_all_generations():
        """List all generations with full artist/album context."""
        from acestep.api.lireek.lireek_db import get_all_generations_with_context
        return {"generations": get_all_generations_with_context()}

    @app.post("/api/lireek/profiles/{profile_id}/generate")
    async def generate_lyrics_endpoint(profile_id: int, req: GenerateRequest):
        """Generate new lyrics from a profile."""
        from acestep.api.lireek.lireek_db import (
            get_profile, get_generations, save_generation,
        )
        from acestep.api.lireek.schemas import LyricsProfile
        from acestep.api.lireek.generation_service import generate_lyrics

        profile_row = get_profile(profile_id)
        if not profile_row:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = profile_row.get("profile_data", {})
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)
        profile = LyricsProfile(**profile_data)

        # Get used subjects/bpms/keys for variety
        existing = get_generations(profile_id=profile_id)
        used_subjects = [g.get("subject", "") for g in existing if g.get("subject")]
        used_bpms = [g.get("bpm", 0) for g in existing if g.get("bpm")]
        used_keys = [g.get("key", "") for g in existing if g.get("key")]
        used_titles = [g.get("title", "") for g in existing if g.get("title")]

        try:
            result = generate_lyrics(
                profile=profile,
                provider_name=req.provider,
                model=req.model,
                extra_instructions=req.extra_instructions,
                used_subjects=used_subjects,
                used_bpms=used_bpms,
                used_keys=used_keys,
                used_titles=used_titles,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Generation failed: %s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        saved = save_generation(
            profile_id=profile_id,
            provider=result.provider,
            model=result.model,
            lyrics=result.lyrics,
            extra_instructions=req.extra_instructions,
            title=result.title,
            subject=result.subject,
            bpm=result.bpm,
            key=result.key,
            caption=result.caption,
            duration=result.duration,
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
        )
        return saved

    @app.post("/api/lireek/generations/{generation_id}/refine")
    async def refine_lyrics_endpoint(generation_id: int, req: RefineGenerationRequest):
        """Refine an existing generation."""
        from acestep.api.lireek.lireek_db import (
            get_generations, get_profile, save_generation,
        )
        from acestep.api.lireek.schemas import LyricsProfile
        from acestep.api.lireek.generation_service import refine_lyrics

        # Find the original generation
        all_gens = get_generations()
        original = next((g for g in all_gens if g["id"] == generation_id), None)
        if not original:
            raise HTTPException(status_code=404, detail="Generation not found")

        # Load profile for style context
        profile = None
        profile_row = get_profile(original["profile_id"])
        if profile_row:
            pd = profile_row.get("profile_data", {})
            if isinstance(pd, str):
                pd = json.loads(pd)
            profile = LyricsProfile(**pd)

        artist_name = profile.artist if profile else "Unknown"

        try:
            result = refine_lyrics(
                original_lyrics=original["lyrics"],
                artist_name=artist_name,
                title=original.get("title", ""),
                provider_name=req.provider,
                model=req.model,
                profile=profile,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Refinement failed: %s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        saved = save_generation(
            profile_id=original["profile_id"],
            provider=result.provider,
            model=result.model,
            lyrics=result.lyrics,
            title=result.title,
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
            parent_generation_id=generation_id,
        )
        return saved

    @app.patch("/api/lireek/generations/{generation_id}/metadata")
    async def update_generation_metadata_endpoint(generation_id: int, req: UpdateMetadataRequest):
        """Update metadata fields on an existing generation."""
        from acestep.api.lireek.lireek_db import update_generation_metadata
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        result = update_generation_metadata(generation_id, **updates)
        if not result:
            raise HTTPException(status_code=404, detail="Generation not found")
        return result

    @app.delete("/api/lireek/generations/{generation_id}")
    async def delete_generation(generation_id: int):
        from acestep.api.lireek.lireek_db import delete_generation
        if not delete_generation(generation_id):
            raise HTTPException(status_code=404, detail="Generation not found")
        return {"deleted": True}

    # ── Export ────────────────────────────────────────────────────────────

    @app.post("/api/lireek/generations/{generation_id}/export")
    async def export_generation_endpoint(generation_id: int):
        """Export a generation as JSON + TXT files."""
        from acestep.api.lireek.lireek_db import get_generations
        from acestep.api.lireek.export_service import export_generation_with_context

        all_gens = get_generations()
        gen = next((g for g in all_gens if g["id"] == generation_id), None)
        if not gen:
            raise HTTPException(status_code=404, detail="Generation not found")

        try:
            folder = export_generation_with_context(gen)
        except Exception as e:
            logger.error("Export failed: %s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        if folder is None:
            raise HTTPException(status_code=404, detail="Could not resolve profile/album for export")

        return {"exported": True, "path": str(folder)}

    # ── Album Presets ─────────────────────────────────────────────────────

    @app.get("/api/lireek/lyrics-sets/{lyrics_set_id}/preset")
    async def get_album_preset(lyrics_set_id: int):
        from acestep.api.lireek.lireek_db import get_album_preset
        preset = get_album_preset(lyrics_set_id)
        if not preset:
            return {"preset": None}
        return {"preset": preset}

    @app.put("/api/lireek/lyrics-sets/{lyrics_set_id}/preset")
    async def upsert_album_preset(lyrics_set_id: int, req: AlbumPresetUpdate):
        from acestep.api.lireek.lireek_db import upsert_album_preset
        preset = upsert_album_preset(
            lyrics_set_id=lyrics_set_id,
            adapter_path=req.adapter_path,
            adapter_scale=req.adapter_scale,
            adapter_group_scales=req.adapter_group_scales,
            matchering_reference_path=req.matchering_reference_path,
        )
        return {"preset": preset}

    @app.delete("/api/lireek/lyrics-sets/{lyrics_set_id}/preset")
    async def delete_album_preset(lyrics_set_id: int):
        from acestep.api.lireek.lireek_db import delete_album_preset
        if not delete_album_preset(lyrics_set_id):
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"deleted": True}

    # ── Slop Detection ────────────────────────────────────────────────────

    @app.post("/api/lireek/slop-scan")
    async def slop_scan(req: SlopScanRequest):
        """Run the AI-slop detector on arbitrary text."""
        from acestep.api.lireek.slop_detector import AISlopDetector
        detector = AISlopDetector()
        return detector.scan_text(req.text)

    # ── Bulk Operations ───────────────────────────────────────────────────

    @app.post("/api/lireek/purge")
    async def purge_all():
        """Delete all profiles and generations (keeps artists + lyrics sets)."""
        from acestep.api.lireek.lireek_db import purge_profiles_and_generations
        return purge_profiles_and_generations()

    # ── Audio Generation Mapping ──────────────────────────────────────────

    @app.post("/api/lireek/generations/{generation_id}/audio")
    async def link_audio_generation(generation_id: int, job_id: str):
        """Link a lyric generation to a HOT-Step audio job."""
        from acestep.api.lireek.lireek_db import save_audio_generation
        return save_audio_generation(generation_id, job_id)

    @app.get("/api/lireek/generations/{generation_id}/audio")
    async def get_audio_generations(generation_id: int):
        from acestep.api.lireek.lireek_db import get_audio_generations
        return {"audio_generations": get_audio_generations(generation_id)}
