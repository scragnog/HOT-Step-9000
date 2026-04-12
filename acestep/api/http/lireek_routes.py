"""
Lireek lyrics management API routes.

Mounted at ``/api/lireek/`` — provides endpoints for the full lyrics
pipeline: Genius fetch → profiling → generation → refinement → export.
"""

import json
import logging
import queue
import threading
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
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


class CreateArtistRequest(BaseModel):
    name: str = Field(..., min_length=1)
    image_url: Optional[str] = None


class ManualSong(BaseModel):
    title: str
    lyrics: str


class CreateLyricsSetRequest(BaseModel):
    artist_id: int
    album: Optional[str] = None
    image_url: Optional[str] = None
    songs: list[ManualSong] = []


class AddSongRequest(BaseModel):
    title: str = Field(..., min_length=1)
    lyrics: str = Field(..., min_length=1)


class SongSelection(BaseModel):
    lyrics_set_id: int
    song_indices: list[int]


class CuratedProfileRequest(BaseModel):
    selections: list[SongSelection]
    provider: str = "gemini"
    model: Optional[str] = None


# ── Route registration ───────────────────────────────────────────────────────

def register_lireek_routes(app: FastAPI) -> None:
    """Register all Lireek lyrics management routes."""

    # ── Artists ────────────────────────────────────────────────────────────

    @app.get("/api/lireek/artists")
    async def list_artists():
        from acestep.api.lireek.lireek_db import list_artists
        return {"artists": list_artists()}

    @app.post("/api/lireek/artists/create")
    async def create_artist(req: CreateArtistRequest):
        """Create a new artist manually (without Genius)."""
        from acestep.api.lireek.lireek_db import get_or_create_artist, update_artist_image
        artist = get_or_create_artist(req.name.strip())
        if req.image_url and req.image_url.strip():
            update_artist_image(artist["id"], req.image_url.strip())
            artist["image_url"] = req.image_url.strip()
        return {"artist": artist}

    @app.delete("/api/lireek/artists/{artist_id}")
    async def delete_artist(artist_id: int):
        from acestep.api.lireek.lireek_db import delete_artist
        if not delete_artist(artist_id):
            raise HTTPException(status_code=404, detail="Artist not found")
        return {"deleted": True}

    @app.post("/api/lireek/artists/{artist_id}/refresh-image")
    async def refresh_artist_image(artist_id: int):
        """Fetch artist image from Genius and store it."""
        import asyncio
        from acestep.api.lireek.lireek_db import update_artist_image
        from acestep.api.lireek.genius_service import get_artist_image_url
        conn = None
        try:
            from acestep.api.lireek.lireek_db import _connect, _row_to_dict
            conn = _connect()
            row = conn.execute("SELECT * FROM artists WHERE id = ?", (artist_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Artist not found")
            artist = _row_to_dict(row)
            # Run in thread to avoid blocking the event loop
            image_url = await asyncio.to_thread(get_artist_image_url, artist["name"])
            if image_url:
                update_artist_image(artist_id, image_url)
                return {"image_url": image_url}
            raise HTTPException(status_code=404, detail="Could not find artist image on Genius")
        finally:
            if conn:
                conn.close()

    class SetImageRequest(BaseModel):
        image_url: str

    @app.post("/api/lireek/artists/{artist_id}/set-image")
    async def set_artist_image(artist_id: int, body: SetImageRequest):
        """Set a custom image URL for an artist (user override for wrong images)."""
        from acestep.api.lireek.lireek_db import set_artist_custom_image
        image_url = body.image_url.strip()
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        if not set_artist_custom_image(artist_id, image_url):
            raise HTTPException(status_code=404, detail="Artist not found")
        return {"image_url": image_url}

    @app.post("/api/lireek/lyrics-sets/{ls_id}/refresh-image")
    async def refresh_album_image(ls_id: int):
        """Fetch album cover art from Genius and store it."""
        import asyncio
        from acestep.api.lireek.lireek_db import (
            _connect, _row_to_dict, update_lyrics_set_image,
        )
        from acestep.api.lireek.genius_service import get_album_cover_art
        conn = None
        try:
            conn = _connect()
            row = conn.execute(
                "SELECT ls.*, a.name as artist_name "
                "FROM lyrics_sets ls JOIN artists a ON a.id = ls.artist_id "
                "WHERE ls.id = ?", (ls_id,)
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Lyrics set not found")
            ls = _row_to_dict(row)
            album_name = ls.get("album")
            artist_name = ls.get("artist_name", "")
            if not album_name:
                logger.info("[refresh-album-image] No album name for ls_id=%d, skipping", ls_id)
                raise HTTPException(status_code=400, detail="No album name — cannot search for cover art")
            logger.info("[refresh-album-image] Searching Genius for '%s' by '%s'", album_name, artist_name)
            # Run in thread to avoid blocking the event loop
            image_url = await asyncio.to_thread(get_album_cover_art, artist_name, album_name)
            if image_url:
                update_lyrics_set_image(ls_id, image_url)
                logger.info("[refresh-album-image] Found cover: %s", image_url)
                return {"image_url": image_url}
            logger.warning("[refresh-album-image] No cover found for '%s' by '%s'", album_name, artist_name)
            raise HTTPException(status_code=404, detail=f"Could not find cover art for '{album_name}' by '{artist_name}'")
        finally:
            if conn:
                conn.close()

    class SetAlbumImageRequest(BaseModel):
        image_url: str

    @app.post("/api/lireek/lyrics-sets/{ls_id}/set-image")
    async def set_album_image(ls_id: int, body: SetAlbumImageRequest):
        """Set a custom image URL for an album (user override)."""
        from acestep.api.lireek.lireek_db import update_lyrics_set_image
        image_url = body.image_url.strip()
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        if not update_lyrics_set_image(ls_id, image_url):
            raise HTTPException(status_code=404, detail="Lyrics set not found")
        return {"image_url": image_url}

    # ── Lyrics Sets ───────────────────────────────────────────────────────

    @app.post("/api/lireek/lyrics-sets/create")
    async def create_lyrics_set_manual(req: CreateLyricsSetRequest):
        """Create a lyrics set (album) manually with optional songs."""
        from acestep.api.lireek.lireek_db import save_lyrics_set, update_lyrics_set_image
        songs_data = [{"title": s.title, "album": req.album, "lyrics": s.lyrics} for s in req.songs]
        ls = save_lyrics_set(
            artist_id=req.artist_id,
            album=req.album,
            max_songs=len(songs_data),
            songs=songs_data,
        )
        if req.image_url and req.image_url.strip():
            update_lyrics_set_image(ls["id"], req.image_url.strip())
            ls["image_url"] = req.image_url.strip()
        return {"lyrics_set": ls}

    @app.post("/api/lireek/lyrics-sets/{lyrics_set_id}/add-song")
    async def add_song_to_set(lyrics_set_id: int, req: AddSongRequest):
        """Add a song to an existing lyrics set."""
        from acestep.api.lireek.lireek_db import add_song_to_set
        result = add_song_to_set(lyrics_set_id, req.title.strip(), req.lyrics)
        if not result:
            raise HTTPException(status_code=404, detail="Lyrics set not found")
        return result

    @app.get("/api/lireek/lyrics-sets")
    async def list_lyrics_sets(artist_id: Optional[int] = None, include_full: bool = False):
        from acestep.api.lireek.lireek_db import get_lyrics_sets
        return {"lyrics_sets": get_lyrics_sets(artist_id, include_full=include_full)}

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

    class EditSongRequest(BaseModel):
        lyrics: str

    @app.put("/api/lireek/lyrics-sets/{lyrics_set_id}/songs/{song_index}")
    async def edit_song(lyrics_set_id: int, song_index: int, body: EditSongRequest):
        from acestep.api.lireek.lireek_db import update_song_lyrics
        result = update_song_lyrics(lyrics_set_id, song_index, body.lyrics)
        if not result:
            raise HTTPException(status_code=404, detail="Song or lyrics set not found")
        return result

    # ── Genius Fetch ──────────────────────────────────────────────────────

    @app.post("/api/lireek/fetch-lyrics")
    async def fetch_lyrics_endpoint(req: FetchLyricsRequest):
        """Fetch lyrics from Genius and store in the database."""
        from acestep.api.lireek.genius_service import fetch_lyrics
        from acestep.api.lireek.lireek_db import (
            get_or_create_artist, save_lyrics_set, update_artist_image,
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

        # Store artist image from Genius if available
        if result.artist_image_url:
            try:
                update_artist_image(
                    artist["id"], result.artist_image_url,
                    genius_id=result.genius_artist_id,
                )
                artist["image_url"] = result.artist_image_url
                artist["genius_id"] = result.genius_artist_id
                logger.info("Stored artist image for '%s'", result.artist)
            except Exception as e:
                logger.warning("Failed to store artist image: %s", e)

        logger.info("Saved lyrics_set id=%d with total_songs=%d",
                     lyrics_set["id"], lyrics_set["total_songs"])

        return {
            "artist": artist,
            "lyrics_set": lyrics_set,
            "songs_fetched": len(result.songs),
        }

    # ── Profiles ──────────────────────────────────────────────────────────

    @app.get("/api/lireek/profiles")
    async def list_profiles(lyrics_set_id: Optional[int] = None, include_full: bool = False):
        from acestep.api.lireek.lireek_db import get_profiles
        return {"profiles": get_profiles(lyrics_set_id, include_full=include_full)}

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
    async def list_generations(profile_id: Optional[int] = None, lyrics_set_id: Optional[int] = None, include_full: bool = False):
        from acestep.api.lireek.lireek_db import get_generations
        return {"generations": get_generations(profile_id=profile_id, lyrics_set_id=lyrics_set_id, include_full=include_full)}

    @app.get("/api/lireek/generations/all")
    async def list_all_generations():
        """List all generations with full artist/album context."""
        from acestep.api.lireek.lireek_db import get_all_generations_with_context
        return {"generations": get_all_generations_with_context()}

    @app.get("/api/lireek/generations/{generation_id}")
    async def get_generation_detail(generation_id: int):
        from acestep.api.lireek.lireek_db import get_generation
        gen = get_generation(generation_id)
        if not gen:
            raise HTTPException(status_code=404, detail="Generation not found")
        return gen

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
        used_durations = [g.get("duration", 0) for g in existing if g.get("duration")]

        try:
            result = generate_lyrics(
                profile=profile,
                provider_name=req.provider,
                model=req.model,
                extra_instructions=req.extra_instructions,
                used_subjects=used_subjects,
                used_bpms=used_bpms,
                used_keys=used_keys,
                used_durations=used_durations,
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
            get_generation, get_profile, save_generation,
        )
        from acestep.api.lireek.schemas import LyricsProfile
        from acestep.api.lireek.generation_service import refine_lyrics

        # Find the original generation (get_generation returns full data incl. lyrics)
        original = get_generation(generation_id)
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

        # Ensure title ends with " - Refined" (strip existing to avoid stacking)
        refined_title = (result.title or original.get("title", "")).removesuffix(" - Refined") + " - Refined"
        saved = save_generation(
            profile_id=original["profile_id"],
            provider=result.provider,
            model=result.model,
            lyrics=result.lyrics,
            title=refined_title,
            subject=original.get("subject", ""),
            bpm=original.get("bpm", 0),
            key=original.get("key", ""),
            caption=original.get("caption", ""),
            duration=original.get("duration", 0),
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
            parent_generation_id=generation_id,
        )
        return saved

    # ── SSE Streaming Endpoints ───────────────────────────────────────────

    def _sse_stream(q: queue.Queue):
        """Yield SSE data lines from a queue until sentinel None."""
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n"

    @app.post("/api/lireek/lyrics-sets/{lyrics_set_id}/build-profile-stream")
    async def build_profile_stream(lyrics_set_id: int, req: BuildProfileRequest):
        """Build a profile with SSE streaming of LLM output."""
        from acestep.api.lireek.lireek_db import get_lyrics_set, save_profile
        from acestep.api.lireek.schemas import SongLyrics
        from acestep.api.lireek.profiler_service import build_profile
        from acestep.api.lireek.generation_service import make_streaming_llm_caller

        ls = get_lyrics_set(lyrics_set_id)
        if not ls:
            raise HTTPException(status_code=404, detail="Lyrics set not found")

        songs_raw = ls.get("songs") or []
        if isinstance(songs_raw, str):
            songs_raw = json.loads(songs_raw)
        songs = [SongLyrics(**s) for s in songs_raw]

        if not songs:
            raise HTTPException(status_code=400, detail="Lyrics set has no songs")

        q: queue.Queue = queue.Queue()

        def on_chunk(text: str):
            q.put(json.dumps({"type": "chunk", "text": text}) + "\n")

        def on_phase(phase: str):
            q.put(json.dumps({"type": "phase", "text": phase}) + "\n")

        def run():
            try:
                caller = make_streaming_llm_caller(req.provider, req.model, on_chunk=on_chunk)
                profile = build_profile(
                    artist=ls.get("artist_name", "Unknown"),
                    album=ls.get("album"),
                    songs=songs,
                    llm_call=caller,
                    on_phase=on_phase,
                )
                saved = save_profile(
                    lyrics_set_id=lyrics_set_id,
                    provider=req.provider,
                    model=req.model or "",
                    profile_data=profile.model_dump(),
                )
                q.put(json.dumps({"type": "result", "data": saved}) + "\n")
            except Exception as exc:
                logger.exception("Streaming profile build failed")
                q.put(json.dumps({"type": "error", "message": str(exc)}) + "\n")
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return StreamingResponse(_sse_stream(q), media_type="text/event-stream")

    @app.post("/api/lireek/profiles/{profile_id}/generate-stream")
    async def generate_stream(profile_id: int, req: GenerateRequest):
        """Generate lyrics with SSE streaming of LLM output."""
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

        existing = get_generations(profile_id=profile_id)
        used_subjects = [g.get("subject", "") for g in existing if g.get("subject")]
        used_bpms = [g.get("bpm", 0) for g in existing if g.get("bpm")]
        used_keys = [g.get("key", "") for g in existing if g.get("key")]
        used_titles = [g.get("title", "") for g in existing if g.get("title")]
        used_durations = [g.get("duration", 0) for g in existing if g.get("duration")]

        q: queue.Queue = queue.Queue()

        def on_chunk(text: str):
            q.put(json.dumps({"type": "chunk", "text": text}) + "\n")

        def on_phase(phase: str):
            q.put(json.dumps({"type": "phase", "text": phase}) + "\n")

        def run():
            try:
                result = generate_lyrics(
                    profile=profile,
                    provider_name=req.provider,
                    model=req.model,
                    extra_instructions=req.extra_instructions,
                    used_subjects=used_subjects,
                    used_bpms=used_bpms,
                    used_keys=used_keys,
                    used_durations=used_durations,
                    used_titles=used_titles,
                    on_chunk=on_chunk,
                    on_phase=on_phase,
                )
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
                q.put(json.dumps({"type": "result", "data": saved}) + "\n")
            except Exception as exc:
                logger.exception("Streaming generation failed")
                q.put(json.dumps({"type": "error", "message": str(exc)}) + "\n")
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return StreamingResponse(_sse_stream(q), media_type="text/event-stream")

    @app.post("/api/lireek/generations/{generation_id}/refine-stream")
    async def refine_stream(generation_id: int, req: RefineGenerationRequest):
        """Refine lyrics with SSE streaming of LLM output."""
        from acestep.api.lireek.lireek_db import (
            get_generation, get_profile, save_generation,
        )
        from acestep.api.lireek.schemas import LyricsProfile
        from acestep.api.lireek.generation_service import refine_lyrics

        original = get_generation(generation_id)
        if not original:
            raise HTTPException(status_code=404, detail="Generation not found")

        profile = None
        profile_row = get_profile(original["profile_id"])
        if profile_row:
            pd = profile_row.get("profile_data", {})
            if isinstance(pd, str):
                pd = json.loads(pd)
            profile = LyricsProfile(**pd)

        artist_name = profile.artist if profile else "Unknown"

        q: queue.Queue = queue.Queue()

        def on_chunk(text: str):
            q.put(json.dumps({"type": "chunk", "text": text}) + "\n")

        def run():
            try:
                result = refine_lyrics(
                    original_lyrics=original["lyrics"],
                    artist_name=artist_name,
                    title=original.get("title", ""),
                    provider_name=req.provider,
                    model=req.model,
                    profile=profile,
                    on_chunk=on_chunk,
                )
                # Ensure title ends with " - Refined" (strip existing to avoid stacking)
                refined_title = (result.title or original.get("title", "")).removesuffix(" - Refined") + " - Refined"
                saved = save_generation(
                    profile_id=original["profile_id"],
                    provider=result.provider,
                    model=result.model,
                    lyrics=result.lyrics,
                    title=refined_title,
                    subject=original.get("subject", ""),
                    bpm=original.get("bpm", 0),
                    key=original.get("key", ""),
                    caption=original.get("caption", ""),
                    duration=original.get("duration", 0),
                    system_prompt=result.system_prompt,
                    user_prompt=result.user_prompt,
                    parent_generation_id=generation_id,
                )
                q.put(json.dumps({"type": "result", "data": saved}) + "\n")
            except Exception as exc:
                logger.exception("Streaming refinement failed")
                q.put(json.dumps({"type": "error", "message": str(exc)}) + "\n")
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return StreamingResponse(_sse_stream(q), media_type="text/event-stream")

    @app.post("/api/lireek/skip-thinking")
    async def skip_thinking():
        """Signal the LLM to stop thinking and produce output."""
        from acestep.api.llm.provider_manager import skip_thinking as _skip
        _skip()
        return {"status": "skip_requested"}

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
        raw = get_album_preset(lyrics_set_id)
        if not raw:
            return {"preset": None}
        # Map DB column names → frontend-expected keys
        scales = raw.get("adapter_scales") or {}
        preset = {
            "id": raw.get("id"),
            "lyrics_set_id": raw.get("lyrics_set_id"),
            "adapter_path": raw.get("adapter_path"),
            "adapter_scale": scales.get("scale", 1.0),
            "adapter_group_scales": scales.get("group_scales"),
            "reference_track_path": raw.get("matchering_ref_path"),
            "audio_cover_strength": raw.get("audio_cover_strength"),
            "updated_at": raw.get("updated_at", ""),
        }
        return {"preset": preset}

    @app.put("/api/lireek/lyrics-sets/{lyrics_set_id}/preset")
    async def upsert_album_preset(lyrics_set_id: int, req: AlbumPresetUpdate):
        from acestep.api.lireek.lireek_db import upsert_album_preset
        # Combine adapter_scale + adapter_group_scales into single DB blob
        adapter_scales = {
            "scale": req.adapter_scale,
            "group_scales": req.adapter_group_scales or {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0},
        }
        raw = upsert_album_preset(
            lyrics_set_id=lyrics_set_id,
            adapter_path=req.adapter_path,
            adapter_scales=adapter_scales,
            matchering_ref_path=req.reference_track_path,
            audio_cover_strength=req.audio_cover_strength,
        )
        # Map back for response
        scales = raw.get("adapter_scales") or {}
        preset = {
            "id": raw.get("id"),
            "lyrics_set_id": raw.get("lyrics_set_id"),
            "adapter_path": raw.get("adapter_path"),
            "adapter_scale": scales.get("scale", 1.0),
            "adapter_group_scales": scales.get("group_scales"),
            "reference_track_path": raw.get("matchering_ref_path"),
            "audio_cover_strength": raw.get("audio_cover_strength"),
            "updated_at": raw.get("updated_at", ""),
        }
        return {"preset": preset}

    @app.delete("/api/lireek/lyrics-sets/{lyrics_set_id}/preset")
    async def delete_album_preset(lyrics_set_id: int):
        from acestep.api.lireek.lireek_db import delete_album_preset
        if not delete_album_preset(lyrics_set_id):
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"deleted": True}

    # ── Slop Detection ────────────────────────────────────────────────────

    @app.get("/api/lireek/presets")
    async def list_all_presets():
        """List all album presets (bulk view for QueuePanel)."""
        from acestep.api.lireek.lireek_db import list_all_album_presets
        raw_list = list_all_album_presets()
        mapped = []
        for raw in raw_list:
            scales = raw.get("adapter_scales") or {}
            mapped.append({
                "id": raw.get("id"),
                "lyrics_set_id": raw.get("lyrics_set_id"),
                "adapter_path": raw.get("adapter_path"),
                "adapter_scale": scales.get("scale", 1.0),
                "adapter_group_scales": scales.get("group_scales"),
                "reference_track_path": raw.get("matchering_ref_path"),
                "audio_cover_strength": raw.get("audio_cover_strength"),
                "updated_at": raw.get("updated_at", ""),
            })
        return {"presets": mapped}


    @app.post("/api/lireek/slop-scan")
    async def slop_scan(req: SlopScanRequest):
        """Run the AI-slop detector on arbitrary text."""
        from acestep.api.lireek.slop_detector import AISlopDetector
        detector = AISlopDetector()
        return detector.scan_text(req.text)

    # ── Curated Cross-Album Profiling ────────────────────────────────────

    @app.post("/api/lireek/artists/{artist_id}/curated-profile")
    async def build_curated_profile(artist_id: int, req: CuratedProfileRequest):
        """Create a curated lyrics set from songs across albums, then build a profile."""
        from acestep.api.lireek.lireek_db import (
            get_lyrics_set as _get_ls, save_lyrics_set, save_profile,
            _connect,
        )
        from acestep.api.lireek.schemas import SongLyrics
        from acestep.api.lireek.profiler_service import build_profile
        from acestep.api.lireek.generation_service import make_llm_caller

        # Assemble songs from selections
        curated_songs_data: list[dict] = []
        curated_song_objects: list[SongLyrics] = []
        for sel in req.selections:
            ls = _get_ls(sel.lyrics_set_id)
            if not ls:
                continue
            songs_raw = ls.get("songs", [])
            if isinstance(songs_raw, str):
                songs_raw = json.loads(songs_raw)
            for idx in sel.song_indices:
                if 0 <= idx < len(songs_raw):
                    song = songs_raw[idx]
                    curated_songs_data.append(song)
                    curated_song_objects.append(SongLyrics(**song))

        if not curated_song_objects:
            raise HTTPException(status_code=400, detail="No valid songs selected")

        # Create a curated lyrics_set
        curated_ls = save_lyrics_set(
            artist_id=artist_id,
            album="[Curated Selection]",
            max_songs=len(curated_songs_data),
            songs=curated_songs_data,
        )

        # Get artist name
        conn = _connect()
        try:
            row = conn.execute("SELECT name FROM artists WHERE id = ?", (artist_id,)).fetchone()
            artist_name = row["name"] if row else "Unknown"
        finally:
            conn.close()

        # Build profile
        try:
            llm_caller = make_llm_caller(req.provider, req.model)
            profile = build_profile(
                artist=artist_name,
                album="[Curated Selection]",
                songs=curated_song_objects,
                llm_call=llm_caller,
            )
        except Exception as e:
            logger.error("Curated profile build failed: %s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        saved = save_profile(
            lyrics_set_id=curated_ls["id"],
            provider=req.provider,
            model=req.model or "",
            profile_data=profile.model_dump(),
        )

        return {"lyrics_set": curated_ls, "profile": saved}

    @app.post("/api/lireek/artists/{artist_id}/curated-profile-stream")
    async def build_curated_profile_stream(artist_id: int, req: CuratedProfileRequest):
        """Create a curated lyrics set and build profile with SSE streaming."""
        from acestep.api.lireek.lireek_db import (
            get_lyrics_set as _get_ls, save_lyrics_set, save_profile,
            _connect,
        )
        from acestep.api.lireek.schemas import SongLyrics
        from acestep.api.lireek.profiler_service import build_profile
        from acestep.api.lireek.generation_service import make_streaming_llm_caller

        # Assemble songs
        curated_songs_data: list[dict] = []
        curated_song_objects: list[SongLyrics] = []
        for sel in req.selections:
            ls = _get_ls(sel.lyrics_set_id)
            if not ls:
                continue
            songs_raw = ls.get("songs", [])
            if isinstance(songs_raw, str):
                songs_raw = json.loads(songs_raw)
            for idx in sel.song_indices:
                if 0 <= idx < len(songs_raw):
                    song = songs_raw[idx]
                    curated_songs_data.append(song)
                    curated_song_objects.append(SongLyrics(**song))

        if not curated_song_objects:
            raise HTTPException(status_code=400, detail="No valid songs selected")

        curated_ls = save_lyrics_set(
            artist_id=artist_id,
            album="[Curated Selection]",
            max_songs=len(curated_songs_data),
            songs=curated_songs_data,
        )

        conn = _connect()
        try:
            row = conn.execute("SELECT name FROM artists WHERE id = ?", (artist_id,)).fetchone()
            artist_name = row["name"] if row else "Unknown"
        finally:
            conn.close()

        q: queue.Queue = queue.Queue()

        def on_chunk(text: str):
            q.put(json.dumps({"type": "chunk", "text": text}) + "\n")

        def on_phase(phase: str):
            q.put(json.dumps({"type": "phase", "text": phase}) + "\n")

        def run():
            try:
                caller = make_streaming_llm_caller(req.provider, req.model, on_chunk=on_chunk)
                profile = build_profile(
                    artist=artist_name,
                    album="[Curated Selection]",
                    songs=curated_song_objects,
                    llm_call=caller,
                    on_phase=on_phase,
                )
                saved = save_profile(
                    lyrics_set_id=curated_ls["id"],
                    provider=req.provider,
                    model=req.model or "",
                    profile_data=profile.model_dump(),
                )
                q.put(json.dumps({"type": "result", "data": {"lyrics_set": curated_ls, "profile": saved}}) + "\n")
            except Exception as exc:
                logger.exception("Streaming curated profile build failed")
                q.put(json.dumps({"type": "error", "message": str(exc)}) + "\n")
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return StreamingResponse(_sse_stream(q), media_type="text/event-stream")

    # ── Bulk Operations ───────────────────────────────────────────────────

    @app.post("/api/lireek/purge")
    async def purge_all():
        """Delete all profiles and generations (keeps artists + lyrics sets)."""
        from acestep.api.lireek.lireek_db import purge_profiles_and_generations
        return purge_profiles_and_generations()

    # ── Audio Generation Mapping ──────────────────────────────────────────

    @app.post("/api/lireek/generations/{generation_id}/audio")
    async def link_audio_generation(generation_id: int, body: dict):
        """Link a lyric generation to a HOT-Step audio job."""
        from acestep.api.lireek.lireek_db import save_audio_generation
        job_id = body.get("job_id") or body.get("hotstep_job_id")
        if not job_id:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": "job_id is required"})
        return save_audio_generation(generation_id, job_id)

    @app.get("/api/lireek/generations/{generation_id}/audio")
    async def get_audio_generations(generation_id: int):
        from acestep.api.lireek.lireek_db import get_audio_generations
        return {"audio_generations": get_audio_generations(generation_id)}

    @app.delete("/api/lireek/audio-generations/{ag_id}")
    async def delete_audio_generation_route(ag_id: int):
        """Delete an audio generation record."""
        from acestep.api.lireek.lireek_db import delete_audio_generation
        deleted = delete_audio_generation(ag_id)
        if not deleted:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={"error": "Audio generation not found"})
        return {"deleted": True}

    @app.delete("/api/lireek/audio-generations")
    async def delete_all_audio_generations_route():
        """Delete ALL audio generation records (music only, preserves lyrics/profiles)."""
        from acestep.api.lireek.lireek_db import delete_all_audio_generations
        count = delete_all_audio_generations()
        return {"deleted": True, "count": count}

    @app.get("/api/lireek/recent-songs")
    async def get_recent_songs(limit: int = 30):
        """Return recent Lireek audio generations across all artists."""
        from acestep.api.lireek.lireek_db import get_recent_audio_generations
        return {"songs": get_recent_audio_generations(limit)}

    @app.patch("/api/lireek/audio-generations/resolve")
    async def resolve_audio_generation(request: Request):
        """Store resolved audio URL and cover URL on an audio_generation record."""
        from acestep.api.lireek.lireek_db import update_audio_generation_urls_by_job
        body = await request.json()
        job_id = body.get("job_id")
        audio_url = body.get("audio_url")
        cover_url = body.get("cover_url")
        if not job_id or not audio_url:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": "job_id and audio_url required"})
        updated = update_audio_generation_urls_by_job(job_id, audio_url, cover_url)
        return {"updated": updated}

    # ── Song Management ───────────────────────────────────────────────────

    @app.delete("/api/lireek/lyrics-sets/{lyrics_set_id}/songs/{song_index}")
    async def remove_song(lyrics_set_id: int, song_index: int):
        """Remove a song from a lyrics set by index."""
        from acestep.api.lireek.lireek_db import remove_song_from_set
        result = remove_song_from_set(lyrics_set_id, song_index)
        if not result:
            raise HTTPException(404, "Lyrics set or song index not found")
        return result

    # ── Generation Editing ────────────────────────────────────────────────

    @app.patch("/api/lireek/generations/{generation_id}")
    async def patch_generation(generation_id: int, body: dict):
        """Partial update of a generation's fields."""
        from acestep.api.lireek.lireek_db import update_generation
        result = update_generation(generation_id, **body)
        if not result:
            raise HTTPException(404, "Generation not found or no valid fields")
        return result

    @app.delete("/api/lireek/generations/{generation_id}")
    async def delete_generation_endpoint(generation_id: int):
        """Delete a single generation."""
        from acestep.api.lireek.lireek_db import delete_generation
        if not delete_generation(generation_id):
            raise HTTPException(404, "Generation not found")
        return {"deleted": True}

    # ── Prompt Management ─────────────────────────────────────────────────

    # Register all prompt defaults on mount
    from acestep.api.lireek.prompt_manager import register_default, list_prompts, load_prompt, save_prompt, reset_prompt
    from acestep.api.lireek.generation_service import (
        GENERATION_SYSTEM_PROMPT,
        REFINEMENT_SYSTEM_PROMPT,
        SONG_METADATA_SYSTEM_PROMPT,
    )
    from acestep.api.lireek.profiler_service import (
        _PROFILE_PROMPT_1,
        _PROFILE_PROMPT_2,
        _PROFILE_PROMPT_3,
        _SUBJECT_SYSTEM_PROMPT,
    )
    register_default("generation", GENERATION_SYSTEM_PROMPT)
    register_default("refinement", REFINEMENT_SYSTEM_PROMPT)
    register_default("metadata", SONG_METADATA_SYSTEM_PROMPT)
    register_default("profiler_1_themes", _PROFILE_PROMPT_1)
    register_default("profiler_2_tone", _PROFILE_PROMPT_2)
    register_default("profiler_3_imagery", _PROFILE_PROMPT_3)
    register_default("profiler_4_subjects", _SUBJECT_SYSTEM_PROMPT)

    @app.get("/api/lireek/prompts")
    async def get_prompts():
        """List all prompts with name, source, and content."""
        return {"prompts": list_prompts()}

    @app.get("/api/lireek/prompts/{name}")
    async def get_prompt(name: str):
        """Get a single prompt by name."""
        content = load_prompt(name)
        if not content:
            raise HTTPException(404, f"Prompt '{name}' not found")
        return {"name": name, "content": content}

    @app.put("/api/lireek/prompts/{name}")
    async def update_prompt(name: str, body: dict):
        """Save a prompt override to file."""
        content = body.get("content", "")
        if not content:
            raise HTTPException(400, "content is required")
        path = save_prompt(name, content)
        return {"name": name, "path": str(path), "source": "file"}

    @app.delete("/api/lireek/prompts/{name}")
    async def delete_prompt(name: str):
        """Reset a prompt to its default (delete the file override)."""
        reset_prompt(name)
        return {"name": name, "status": "reset"}
