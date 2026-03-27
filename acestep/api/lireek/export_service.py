"""
Export service — writes generation data as JSON + human-readable TXT files.

Ported from Lireek's ``export_service.py``.

Folder structure::

    {export_dir}/{Artist}/{Based on Album}/
        {sanitised_title}_{id}.json
        {sanitised_title}_{id}.txt
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default export directory (can be overridden via settings)
DEFAULT_EXPORT_DIR = r"D:\Ace-Step-Latest\Lyrics"


def _sanitize_dirname(name: str) -> str:
    """Make a string safe for use as a Windows directory name."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.rstrip('. ')
    return name.strip() or "Unknown"


def _sanitize_filename(name: str) -> str:
    """Convert a string into a safe filename (lowercase, underscores)."""
    name = name.strip().lower()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'[\s\-]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name[:80] or "untitled"


def _build_json(gen: dict, artist: str, album: Optional[str]) -> dict:
    """Build the ACE-Step-compatible JSON structure."""
    return {
        "title": gen.get("title", ""),
        "caption": gen.get("caption", ""),
        "lyrics": gen.get("lyrics", ""),
        "bpm": gen.get("bpm", 0),
        "keyscale": gen.get("key", ""),
        "duration": gen.get("duration", 0),
        "metadata": {
            "artist": artist,
            "album": album or "",
            "subject": gen.get("subject", ""),
            "provider": gen.get("provider", ""),
            "model": gen.get("model", ""),
            "lireek_id": gen.get("id", 0),
            "created_at": gen.get("created_at", ""),
        },
    }


def _build_txt(gen: dict, artist: str, album: Optional[str]) -> str:
    """Build a human-readable TXT representation."""
    album_label = album or "Top Songs"
    lines = [
        f"Title: {gen.get('title', 'Untitled')}",
        f"Artist style: {artist} (Based on {album_label})",
    ]
    meta_parts = []
    if gen.get("bpm"):
        meta_parts.append(f"BPM: {gen['bpm']}")
    if gen.get("key"):
        meta_parts.append(f"Key: {gen['key']}")
    if meta_parts:
        lines.append(" | ".join(meta_parts))
    if gen.get("subject"):
        lines.append(f"Subject: {gen['subject']}")
    if gen.get("caption"):
        lines.append(f"\nCaption:\n{gen['caption']}")
    lines.append("\n---\n")
    lines.append(gen.get("lyrics", ""))
    return "\n".join(lines)


def export_generation(
    gen: dict,
    artist: str,
    album: Optional[str],
    export_dir: Optional[str] = None,
) -> Path:
    """Write a generation as JSON + TXT to the export directory.

    Returns the directory where files were written.
    """
    if export_dir is None:
        from acestep.api.lireek.lireek_db import get_setting
        export_dir = get_setting("export_dir") or DEFAULT_EXPORT_DIR

    album_label = album or "Top Songs"
    folder = Path(export_dir) / _sanitize_dirname(artist) / f"Based on {_sanitize_dirname(album_label)}"
    folder.mkdir(parents=True, exist_ok=True)

    title_slug = _sanitize_filename(gen.get("title", ""))
    gen_id = gen.get("id", 0)
    basename = f"{title_slug}_{gen_id}" if title_slug != "untitled" else f"generation_{gen_id}"

    if gen.get("parent_generation_id"):
        ref_num = gen.get("refinement_number", 1)
        basename = f"{basename}_refined_{ref_num}"

    # JSON
    json_path = folder / f"{basename}.json"
    json_data = _build_json(gen, artist, album)
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # TXT
    txt_path = folder / f"{basename}.txt"
    txt_data = _build_txt(gen, artist, album)
    txt_path.write_text(txt_data, encoding="utf-8")

    logger.info("Exported generation %d → %s", gen_id, folder)
    return folder


def export_generation_with_context(
    gen: dict,
    export_dir: Optional[str] = None,
) -> Optional[Path]:
    """Export a single generation, resolving artist/album from the DB."""
    from acestep.api.lireek.lireek_db import get_profile, get_lyrics_set, get_generations_by_profile

    profile = get_profile(gen["profile_id"])
    if not profile:
        logger.warning("Cannot export generation %d: profile %d not found", gen["id"], gen["profile_id"])
        return None

    lyrics_set = get_lyrics_set(profile["lyrics_set_id"])
    if not lyrics_set:
        logger.warning("Cannot export generation %d: lyrics_set not found", gen["id"])
        return None

    artist = lyrics_set.get("artist_name", "Unknown Artist")
    album = lyrics_set.get("album")

    # Calculate refinement number for refined variants
    if gen.get("parent_generation_id"):
        all_gens = get_generations_by_profile(gen["profile_id"])
        siblings = sorted(
            [g for g in all_gens if g.get("parent_generation_id") == gen["parent_generation_id"]],
            key=lambda g: g["id"],
        )
        gen["refinement_number"] = next(
            (i + 1 for i, g in enumerate(siblings) if g["id"] == gen["id"]),
            1,
        )

    return export_generation(gen, artist, album, export_dir)
