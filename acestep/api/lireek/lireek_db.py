"""
SQLite-based storage for the Lireek lyrics component.

Creates and manages ``data/hotstep_lyrics.db`` — a self-contained database
alongside (but separate from) the main HOT-Step data stores.

Tables
------
- artists        – unique artist names (case-insensitive)
- lyrics_sets    – fetched lyrics per artist + album (songs as JSON)
- profiles       – LLM-generated stylistic profiles per lyrics_set
- generations    – generated lyrics with metadata (BPM, key, caption …)
- album_presets  – persistent adapter + matchering bindings per album
- audio_generations – maps lyric generation_id → HOT-Step job_id
- settings       – key-value config store
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_DB_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "data"
_DB_PATH: Path = _DB_DIR / "hotstep_lyrics.db"


def get_db_path() -> Path:
    """Return the resolved database path (useful for tests / logging)."""
    return _DB_PATH


def set_db_path(path: Path) -> None:
    """Override the database path (for tests only)."""
    global _DB_DIR, _DB_PATH
    _DB_PATH = path
    _DB_DIR = path.parent


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS artists (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    image_url   TEXT,
    genius_id   INTEGER,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lyrics_sets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    artist_id   INTEGER NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    album       TEXT,
    max_songs   INTEGER NOT NULL DEFAULT 10,
    songs       TEXT    NOT NULL,   -- JSON array of {title, album, lyrics}
    fetched_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lyrics_set_id   INTEGER NOT NULL REFERENCES lyrics_sets(id) ON DELETE CASCADE,
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    profile_data    TEXT    NOT NULL,   -- JSON of full LyricsProfile
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS generations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id              INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    provider                TEXT    NOT NULL,
    model                   TEXT    NOT NULL,
    extra_instructions      TEXT,
    title                   TEXT    NOT NULL DEFAULT '',
    subject                 TEXT    NOT NULL DEFAULT '',
    bpm                     INTEGER NOT NULL DEFAULT 0,
    key                     TEXT    NOT NULL DEFAULT '',
    caption                 TEXT    NOT NULL DEFAULT '',
    duration                INTEGER NOT NULL DEFAULT 0,
    lyrics                  TEXT    NOT NULL,
    system_prompt           TEXT    NOT NULL DEFAULT '',
    user_prompt             TEXT    NOT NULL DEFAULT '',
    parent_generation_id    INTEGER REFERENCES generations(id) ON DELETE SET NULL,
    created_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS album_presets (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    lyrics_set_id           INTEGER NOT NULL UNIQUE REFERENCES lyrics_sets(id) ON DELETE CASCADE,
    adapter_path            TEXT,
    adapter_scales          TEXT,   -- JSON: {scale, group_scales: {self_attn, cross_attn, mlp}}
    matchering_ref_path     TEXT,
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audio_generations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id       INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    hotstep_job_id      TEXT    NOT NULL,   -- UUID from HOT-Step queue
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


# ── Init ──────────────────────────────────────────────────────────────────────

_MIGRATIONS = [
    # (check_sql, migrate_sql_list)
    (
        "SELECT image_url FROM artists LIMIT 0",
        [
            "ALTER TABLE artists ADD COLUMN image_url TEXT",
            "ALTER TABLE artists ADD COLUMN genius_id INTEGER",
        ],
    ),
    (
        "SELECT image_url FROM lyrics_sets LIMIT 0",
        [
            "ALTER TABLE lyrics_sets ADD COLUMN image_url TEXT",
        ],
    ),
    (
        "SELECT audio_url FROM audio_generations LIMIT 0",
        [
            "ALTER TABLE audio_generations ADD COLUMN audio_url TEXT",
            "ALTER TABLE audio_generations ADD COLUMN cover_url TEXT",
        ],
    ),
]


def init_db() -> None:
    """Create the database file and tables if they don't exist."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        # Run migrations for existing databases
        for check_sql, migrate_sqls in _MIGRATIONS:
            try:
                conn.execute(check_sql)
            except sqlite3.OperationalError:
                for sql in migrate_sqls:
                    try:
                        conn.execute(sql)
                    except sqlite3.OperationalError:
                        pass  # column already exists
                conn.commit()
                logger.info("Applied migration: %s", migrate_sqls[0][:60])
        logger.info("Lireek database initialised at %s", _DB_PATH)
    finally:
        conn.close()


# ── Artists ───────────────────────────────────────────────────────────────────

def get_or_create_artist(name: str) -> dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM artists WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            return _row_to_dict(row)

        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "INSERT INTO artists (name, created_at) VALUES (?, ?)",
            (name, now),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "name": name, "created_at": now}
    finally:
        conn.close()


def list_artists() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT a.*, COUNT(ls.id) AS lyrics_set_count "
            "FROM artists a LEFT JOIN lyrics_sets ls ON ls.artist_id = a.id "
            "GROUP BY a.id ORDER BY a.name"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_artist_image(artist_id: int, image_url: str, genius_id: Optional[int] = None) -> bool:
    """Update the artist's image URL and optionally their Genius ID."""
    conn = _connect()
    try:
        if genius_id is not None:
            conn.execute(
                "UPDATE artists SET image_url = ?, genius_id = ? WHERE id = ?",
                (image_url, genius_id, artist_id),
            )
        else:
            conn.execute(
                "UPDATE artists SET image_url = ? WHERE id = ?",
                (image_url, artist_id),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def delete_artist(artist_id: int) -> bool:
    """Delete an artist and all cascading children."""
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM artists WHERE id = ?", (artist_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def set_artist_custom_image(artist_id: int, image_url: str) -> bool:
    """Set a custom image URL for an artist (user override)."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "UPDATE artists SET image_url = ? WHERE id = ?",
            (image_url, artist_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ── Lyrics Sets ───────────────────────────────────────────────────────────────

def save_lyrics_set(
    artist_id: int,
    album: Optional[str],
    max_songs: int,
    songs: list[dict],
) -> dict[str, Any]:
    conn = _connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        songs_json = json.dumps(songs, ensure_ascii=False)
        cursor = conn.execute(
            "INSERT INTO lyrics_sets (artist_id, album, max_songs, songs, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (artist_id, album, max_songs, songs_json, now),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "artist_id": artist_id,
            "album": album,
            "max_songs": max_songs,
            "total_songs": len(songs),
            "fetched_at": now,
        }
    finally:
        conn.close()


def update_lyrics_set_image(lyrics_set_id: int, image_url: str) -> bool:
    """Set an image URL on a lyrics_set (album cover)."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "UPDATE lyrics_sets SET image_url = ? WHERE id = ?",
            (image_url, lyrics_set_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_lyrics_sets(artist_id: Optional[int] = None, include_full: bool = False) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        if artist_id:
            rows = conn.execute(
                "SELECT ls.*, a.name as artist_name FROM lyrics_sets ls "
                "JOIN artists a ON a.id = ls.artist_id "
                "WHERE ls.artist_id = ? ORDER BY ls.fetched_at DESC",
                (artist_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ls.*, a.name as artist_name FROM lyrics_sets ls "
                "JOIN artists a ON a.id = ls.artist_id "
                "ORDER BY ls.fetched_at DESC"
            ).fetchall()

        results = []
        for r in rows:
            d = _row_to_dict(r)
            songs_raw = d.get("songs", "[]")
            songs = json.loads(songs_raw) if isinstance(songs_raw, str) else songs_raw
            d["total_songs"] = len(songs) if isinstance(songs, list) else 0
            if not include_full:
                # Strip full lyrics — only return titles + char counts for listings
                d["songs"] = json.dumps([{"title": s.get("title", ""), "chars": len(s.get("lyrics", ""))} for s in songs]) if isinstance(songs, list) else "[]"
            results.append(d)
        return results
    finally:
        conn.close()


def get_lyrics_set(lyrics_set_id: int) -> Optional[dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT ls.*, a.name as artist_name FROM lyrics_sets ls "
            "JOIN artists a ON a.id = ls.artist_id "
            "WHERE ls.id = ?",
            (lyrics_set_id,),
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        d["songs"] = json.loads(d["songs"])
        d["total_songs"] = len(d["songs"])
        return d
    finally:
        conn.close()


def delete_lyrics_set(lyrics_set_id: int) -> bool:
    conn = _connect()
    try:
        cursor = conn.execute(
            "DELETE FROM lyrics_sets WHERE id = ?", (lyrics_set_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def remove_song_from_set(lyrics_set_id: int, song_index: int) -> Optional[dict[str, Any]]:
    """Remove a single song by index from a lyrics set. Returns updated set or None."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT songs FROM lyrics_sets WHERE id = ?", (lyrics_set_id,)
        ).fetchone()
        if not row:
            return None

        songs = json.loads(row["songs"])
        if song_index < 0 or song_index >= len(songs):
            return None

        songs.pop(song_index)
        conn.execute(
            "UPDATE lyrics_sets SET songs = ? WHERE id = ?",
            (json.dumps(songs, ensure_ascii=False), lyrics_set_id),
        )
        conn.commit()
        return get_lyrics_set(lyrics_set_id)
    finally:
        conn.close()


def update_song_lyrics(lyrics_set_id: int, song_index: int, new_lyrics: str) -> Optional[dict[str, Any]]:
    """Update the lyrics of a single song by index within a lyrics set."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT songs FROM lyrics_sets WHERE id = ?", (lyrics_set_id,)
        ).fetchone()
        if not row:
            return None

        songs = json.loads(row["songs"])
        if song_index < 0 or song_index >= len(songs):
            return None

        songs[song_index]["lyrics"] = new_lyrics
        conn.execute(
            "UPDATE lyrics_sets SET songs = ? WHERE id = ?",
            (json.dumps(songs, ensure_ascii=False), lyrics_set_id),
        )
        conn.commit()
        return get_lyrics_set(lyrics_set_id)
    finally:
        conn.close()


def add_song_to_set(lyrics_set_id: int, title: str, lyrics: str) -> Optional[dict[str, Any]]:
    """Append a song to a lyrics set's JSON songs array."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT songs, album FROM lyrics_sets WHERE id = ?", (lyrics_set_id,)
        ).fetchone()
        if not row:
            return None

        songs = json.loads(row["songs"])
        songs.append({"title": title, "album": row["album"], "lyrics": lyrics})
        conn.execute(
            "UPDATE lyrics_sets SET songs = ?, max_songs = ? WHERE id = ?",
            (json.dumps(songs, ensure_ascii=False), len(songs), lyrics_set_id),
        )
        conn.commit()
        return get_lyrics_set(lyrics_set_id)
    finally:
        conn.close()


# ── Profiles ──────────────────────────────────────────────────────────────────

def save_profile(
    lyrics_set_id: int,
    provider: str,
    model: str,
    profile_data: dict,
) -> dict[str, Any]:
    conn = _connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        profile_json = json.dumps(profile_data, ensure_ascii=False)
        cursor = conn.execute(
            "INSERT INTO profiles (lyrics_set_id, provider, model, profile_data, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (lyrics_set_id, provider, model, profile_json, now),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "lyrics_set_id": lyrics_set_id,
            "provider": provider,
            "model": model,
            "profile_data": profile_data,
            "created_at": now,
        }
    finally:
        conn.close()


def get_profiles(lyrics_set_id: Optional[int] = None, include_full: bool = False) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        if lyrics_set_id:
            rows = conn.execute(
                "SELECT * FROM profiles WHERE lyrics_set_id = ? ORDER BY created_at DESC",
                (lyrics_set_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM profiles ORDER BY created_at DESC"
            ).fetchall()

        results = []
        for r in rows:
            d = _row_to_dict(r)
            if include_full:
                # Parse JSON profile_data for full view
                pd = d.get("profile_data")
                if isinstance(pd, str):
                    try:
                        d["profile_data"] = json.loads(pd)
                    except json.JSONDecodeError:
                        d["profile_data"] = {}
            else:
                # Strip heavy profile_data from list
                d.pop("profile_data", None)
            results.append(d)
        return results
    finally:
        conn.close()


def get_profile(profile_id: int) -> Optional[dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        d["profile_data"] = json.loads(d["profile_data"])
        return d
    finally:
        conn.close()


def delete_profile(profile_id: int) -> bool:
    """Delete a profile and all its generations (cascade)."""
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ── Generations ───────────────────────────────────────────────────────────────

def save_generation(
    profile_id: int,
    provider: str,
    model: str,
    lyrics: str,
    extra_instructions: Optional[str] = None,
    title: str = "",
    subject: str = "",
    bpm: int = 0,
    key: str = "",
    caption: str = "",
    duration: int = 0,
    system_prompt: str = "",
    user_prompt: str = "",
    parent_generation_id: Optional[int] = None,
) -> dict[str, Any]:
    conn = _connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "INSERT INTO generations "
            "(profile_id, provider, model, extra_instructions, title, subject, "
            "bpm, key, caption, duration, lyrics, system_prompt, user_prompt, "
            "parent_generation_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (profile_id, provider, model, extra_instructions, title, subject,
             bpm, key, caption, duration, lyrics, system_prompt, user_prompt,
             parent_generation_id, now),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "profile_id": profile_id,
            "provider": provider,
            "model": model,
            "extra_instructions": extra_instructions,
            "title": title,
            "subject": subject,
            "bpm": bpm,
            "key": key,
            "caption": caption,
            "duration": duration,
            "lyrics": lyrics,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "parent_generation_id": parent_generation_id,
            "created_at": now,
        }
    finally:
        conn.close()


def get_generations(
    profile_id: Optional[int] = None,
    lyrics_set_id: Optional[int] = None,
    include_full: bool = False,
) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        if profile_id:
            rows = conn.execute(
                "SELECT * FROM generations WHERE profile_id = ? ORDER BY created_at DESC",
                (profile_id,),
            ).fetchall()
        elif lyrics_set_id:
            rows = conn.execute(
                "SELECT g.* FROM generations g "
                "JOIN profiles p ON p.id = g.profile_id "
                "WHERE p.lyrics_set_id = ? ORDER BY g.created_at DESC",
                (lyrics_set_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM generations ORDER BY created_at DESC"
            ).fetchall()
        results = []
        for r in rows:
            d = _row_to_dict(r)
            if not include_full:
                # Strip heavy text from list — fetch via get_generation(id) for full data
                d.pop("lyrics", None)
                d.pop("system_prompt", None)
                d.pop("user_prompt", None)
            results.append(d)
        return results
    finally:
        conn.close()


def get_generation(generation_id: int) -> Optional[dict[str, Any]]:
    """Get a single generation with full data (including lyrics)."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT g.*, a.name AS artist_name, ls.album "
            "FROM generations g "
            "JOIN profiles p ON p.id = g.profile_id "
            "JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id "
            "JOIN artists a ON a.id = ls.artist_id "
            "WHERE g.id = ?",
            (generation_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def get_all_generations_with_context() -> list[dict[str, Any]]:
    """Return all generations joined with artist/album info."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT g.*, a.name AS artist_name, ls.album "
            "FROM generations g "
            "JOIN profiles p ON p.id = g.profile_id "
            "JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id "
            "JOIN artists a ON a.id = ls.artist_id "
            "ORDER BY g.created_at DESC"
        ).fetchall()
        results = []
        for r in rows:
            d = _row_to_dict(r)
            d.pop("lyrics", None)
            d.pop("system_prompt", None)
            d.pop("user_prompt", None)
            results.append(d)
        return results
    finally:
        conn.close()


def update_generation_metadata(
    generation_id: int, bpm: int, key: str, caption: str, duration: int = 0,
) -> None:
    """Update BPM, key, caption, duration on an existing generation row."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE generations SET bpm = ?, key = ?, caption = ?, duration = ? WHERE id = ?",
            (bpm, key, caption, duration, generation_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_generation(generation_id: int, **fields) -> Optional[dict[str, Any]]:
    """General partial update. Accepts any combination of: title, subject, bpm, key, caption, duration, lyrics."""
    allowed = {"title", "subject", "bpm", "key", "caption", "duration", "lyrics"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return None

    conn = _connect()
    try:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [generation_id]
        conn.execute(
            f"UPDATE generations SET {set_clause} WHERE id = ?", values
        )
        conn.commit()

        row = conn.execute("SELECT * FROM generations WHERE id = ?", (generation_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_generation(generation_id: int) -> bool:
    """Delete a single generation by ID."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "DELETE FROM generations WHERE id = ?", (generation_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def purge_profiles_and_generations() -> dict[str, int]:
    """Delete ALL generations and ALL profiles.

    Leaves artists and lyrics_sets intact.
    """
    conn = _connect()
    try:
        gen_cursor = conn.execute("DELETE FROM generations")
        gen_count = gen_cursor.rowcount
        prof_cursor = conn.execute("DELETE FROM profiles")
        prof_count = prof_cursor.rowcount
        conn.commit()
        logger.info("Purged %d generations and %d profiles", gen_count, prof_count)
        return {"generations_deleted": gen_count, "profiles_deleted": prof_count}
    finally:
        conn.close()


# ── Album Presets ─────────────────────────────────────────────────────────────

def get_album_preset(lyrics_set_id: int) -> Optional[dict[str, Any]]:
    """Get the album preset for a lyrics set."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM album_presets WHERE lyrics_set_id = ?",
            (lyrics_set_id,),
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        if d.get("adapter_scales"):
            d["adapter_scales"] = json.loads(d["adapter_scales"])
        return d
    finally:
        conn.close()


def upsert_album_preset(
    lyrics_set_id: int,
    adapter_path: Optional[str] = None,
    adapter_scales: Optional[dict] = None,
    matchering_ref_path: Optional[str] = None,
) -> dict[str, Any]:
    """Create or update the album preset for a lyrics set.

    Only the fields that are explicitly provided (not ``None``) are written.
    Existing values for omitted fields are preserved, so updating just the
    adapter won't clear the matchering reference and vice-versa.
    """
    conn = _connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        scales_json = json.dumps(adapter_scales, ensure_ascii=False) if adapter_scales else None

        # Check if a preset already exists for this lyrics_set
        existing = conn.execute(
            "SELECT * FROM album_presets WHERE lyrics_set_id = ?",
            (lyrics_set_id,),
        ).fetchone()

        if existing is None:
            # Fresh INSERT — write everything (None columns stay NULL)
            conn.execute(
                "INSERT INTO album_presets "
                "(lyrics_set_id, adapter_path, adapter_scales, matchering_ref_path, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (lyrics_set_id, adapter_path, scales_json, matchering_ref_path, now),
            )
        else:
            # Partial UPDATE — only overwrite columns that were explicitly given
            updates: list[str] = ["updated_at = ?"]
            values: list[Any] = [now]

            if adapter_path is not None:
                updates.append("adapter_path = ?")
                values.append(adapter_path)
            if adapter_scales is not None:
                updates.append("adapter_scales = ?")
                values.append(scales_json)
            if matchering_ref_path is not None:
                updates.append("matchering_ref_path = ?")
                values.append(matchering_ref_path)

            values.append(lyrics_set_id)
            conn.execute(
                f"UPDATE album_presets SET {', '.join(updates)} WHERE lyrics_set_id = ?",
                values,
            )

        conn.commit()
        return get_album_preset(lyrics_set_id) or {}
    finally:
        conn.close()


def delete_album_preset(lyrics_set_id: int) -> bool:
    """Delete the album preset for a lyrics set."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "DELETE FROM album_presets WHERE lyrics_set_id = ?", (lyrics_set_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def list_all_album_presets() -> list[dict[str, Any]]:
    """Return every album preset row with adapter_scales JSON unpacked."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM album_presets ORDER BY updated_at DESC"
        ).fetchall()
        results = []
        for r in rows:
            d = _row_to_dict(r)
            if d.get("adapter_scales"):
                d["adapter_scales"] = json.loads(d["adapter_scales"])
            results.append(d)
        return results
    finally:
        conn.close()


# ── Audio Generations (mapping lyrics → HOT-Step jobs) ────────────────────────

def save_audio_generation(generation_id: int, hotstep_job_id: str) -> dict[str, Any]:
    """Record a HOT-Step audio job linked to a lyric generation."""
    conn = _connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "INSERT INTO audio_generations (generation_id, hotstep_job_id, created_at) "
            "VALUES (?, ?, ?)",
            (generation_id, hotstep_job_id, now),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "generation_id": generation_id,
            "hotstep_job_id": hotstep_job_id,
            "created_at": now,
        }
    finally:
        conn.close()


def get_audio_generations(generation_id: int) -> list[dict[str, Any]]:
    """Get all audio generation jobs for a lyric generation."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM audio_generations WHERE generation_id = ? ORDER BY created_at DESC",
            (generation_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def delete_audio_generation(ag_id: int) -> bool:
    """Delete an audio generation record by ID."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "DELETE FROM audio_generations WHERE id = ?", (ag_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_all_audio_generations() -> int:
    """Delete ALL audio generation records (music, not lyrics). Returns count deleted."""
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM audio_generations")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_recent_audio_generations(limit: int = 30) -> list[dict[str, Any]]:
    """Return recent audio generations across ALL artists with full context.

    Joins: audio_generations → generations → profiles → lyrics_sets → artists.
    Returns pre-resolved audio_url and cover_url when available.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT
                ag.id              AS ag_id,
                ag.hotstep_job_id,
                ag.audio_url,
                ag.cover_url,
                ag.created_at      AS ag_created_at,
                g.id               AS generation_id,
                g.title            AS song_title,
                g.subject,
                g.caption,
                g.lyrics,
                g.duration,
                ls.id              AS lyrics_set_id,
                ls.album,
                ls.image_url       AS album_image,
                a.id               AS artist_id,
                a.name             AS artist_name,
                a.image_url        AS artist_image
            FROM audio_generations ag
            JOIN generations g   ON g.id  = ag.generation_id
            JOIN profiles   p   ON p.id  = g.profile_id
            JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
            JOIN artists    a   ON a.id  = ls.artist_id
            ORDER BY ag.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_audio_generation_urls(
    ag_id: int, audio_url: str, cover_url: str | None = None
) -> bool:
    """Store the resolved audio URL and cover art URL on an audio_generation record."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "UPDATE audio_generations SET audio_url = ?, cover_url = ? WHERE id = ?",
            (audio_url, cover_url, ag_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_audio_generation_urls_by_job(
    hotstep_job_id: str, audio_url: str, cover_url: str | None = None
) -> bool:
    """Store resolved URLs by HOT-Step job ID (convenience for queue store)."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "UPDATE audio_generations SET audio_url = ?, cover_url = ? "
            "WHERE hotstep_job_id = ?",
            (audio_url, cover_url, hotstep_job_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ── Settings ──────────────────────────────────────────────────────────────────

def get_setting(key: str, default: str = "") -> str:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def set_setting(key: str, value: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_settings() -> dict[str, str]:
    """Return all settings as a dictionary."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        conn.close()
