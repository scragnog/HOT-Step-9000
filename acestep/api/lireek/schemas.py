"""
Pydantic schemas for the Lireek lyrics sub-system.

Ported from Lireek's ``app/models/schemas.py`` with additions for
album presets and audio generation mapping.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


# ── Provider type ─────────────────────────────────────────────────────────────

LLM_PROVIDER = Literal["gemini", "openai", "anthropic", "ollama", "lmstudio", "unsloth"]


# ── Lyrics acquisition ────────────────────────────────────────────────────────

class LyricsSearchRequest(BaseModel):
    artist: str = Field(..., description="Artist name to search for")
    album: Optional[str] = Field(None, description="Album name (optional)")
    max_songs: int = Field(10, ge=1, le=50, description="Max songs when no album specified")


class SongLyrics(BaseModel):
    title: str
    album: Optional[str] = None
    lyrics: str


class LyricsSearchResponse(BaseModel):
    artist: str
    album: Optional[str] = None
    songs: list[SongLyrics]
    total_songs: int
    artist_image_url: Optional[str] = None
    genius_artist_id: Optional[int] = None


# ── Profile ───────────────────────────────────────────────────────────────────

class LyricsProfile(BaseModel):
    artist: str
    album: Optional[str] = None
    themes: list[str] = []
    common_subjects: list[str] = []
    rhyme_schemes: list[str] = []
    avg_verse_lines: float = 0.0
    avg_chorus_lines: float = 0.0
    vocabulary_notes: str = ""
    tone_and_mood: str = ""
    structural_patterns: str = ""
    additional_notes: str = ""
    raw_summary: str = ""
    # Enhanced fields
    structure_blueprints: list[str] = []
    perspective: str = ""
    meter_stats: dict = {}
    vocabulary_stats: dict = {}
    representative_excerpts: list[str] = []
    narrative_techniques: str = ""
    imagery_patterns: str = ""
    signature_devices: str = ""
    emotional_arc: str = ""
    rhyme_quality: dict = {}
    song_subjects: dict = {}
    subject_categories: list[str] = []
    repetition_stats: dict = {}
    raw_llm_response: str = ""


# ── Generation ────────────────────────────────────────────────────────────────

class GenerationRequest(BaseModel):
    artist: str
    album: Optional[str] = None
    profile: LyricsProfile
    provider: LLM_PROVIDER = "gemini"
    model: Optional[str] = None
    extra_instructions: Optional[str] = Field(None, description="Extra guidance for generation")


class GenerationResponse(BaseModel):
    lyrics: str
    provider: str
    model: str
    title: str = ""
    subject: str = ""
    bpm: int = 0
    key: str = ""
    caption: str = ""
    duration: int = 0
    system_prompt: str = ""
    user_prompt: str = ""


# ── API request models ────────────────────────────────────────────────────────

class BuildProfileRequest(BaseModel):
    provider: LLM_PROVIDER = "gemini"
    model: Optional[str] = None


class GenerateFromProfileRequest(BaseModel):
    provider: LLM_PROVIDER = "gemini"
    model: Optional[str] = None
    extra_instructions: Optional[str] = None


class RefineRequest(BaseModel):
    provider: LLM_PROVIDER = "gemini"
    model: Optional[str] = None


# ── Stored entities ───────────────────────────────────────────────────────────

class SavedArtist(BaseModel):
    id: int
    name: str
    created_at: str
    lyrics_set_count: int = 0


class SavedLyricsSetSummary(BaseModel):
    """Lyrics set without the full song texts (for list views)."""
    id: int
    artist_id: int
    artist_name: str
    album: Optional[str] = None
    max_songs: int = 0
    total_songs: int = 0
    fetched_at: str = ""


class SavedLyricsSet(BaseModel):
    """Full lyrics set including all song texts."""
    id: int
    artist_id: int
    artist_name: str
    album: Optional[str] = None
    max_songs: int = 0
    total_songs: int = 0
    songs: list[SongLyrics] = []
    fetched_at: str = ""


class SavedProfile(BaseModel):
    id: int
    lyrics_set_id: int
    provider: str
    model: str
    profile_data: LyricsProfile
    created_at: str = ""


class SavedGeneration(BaseModel):
    id: int
    profile_id: int
    provider: str
    model: str
    extra_instructions: Optional[str] = None
    title: str = ""
    subject: str = ""
    bpm: int = 0
    key: str = ""
    caption: str = ""
    lyrics: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    parent_generation_id: Optional[int] = None
    created_at: str = ""


# ── Album presets (HOT-Step extension) ────────────────────────────────────────

class AlbumPreset(BaseModel):
    """Persistent adapter + reference track config bound to a lyrics set (album)."""
    id: int
    lyrics_set_id: int
    adapter_path: Optional[str] = None
    adapter_scale: float = 1.0
    adapter_group_scales: Optional[dict] = None
    reference_track_path: Optional[str] = None
    audio_cover_strength: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""


class AlbumPresetUpdate(BaseModel):
    """Payload for creating/updating an album preset."""
    adapter_path: Optional[str] = None
    adapter_scale: float = 1.0
    adapter_group_scales: Optional[dict] = None
    reference_track_path: Optional[str] = None
    audio_cover_strength: Optional[float] = None


# ── Audio generation mapping (HOT-Step extension) ────────────────────────────

class AudioGenerationMapping(BaseModel):
    """Links a lyric generation → HOT-Step audio job."""
    id: int
    generation_id: int
    job_id: str
    created_at: str = ""
