"""
Lyrics acquisition via the Genius API.

Ported from Lireek's ``genius_service.py``. Uses httpx + BeautifulSoup for
scraping lyrics from Genius song pages with browser-like headers.

Configuration is read from the Lireek settings DB, falling back to env vars.
"""

import os
import re
import logging
import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from acestep.api.lireek.schemas import SongLyrics, LyricsSearchResponse

logger = logging.getLogger(__name__)

API_ROOT = "https://api.genius.com"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_genius_token() -> str:
    """Get the Genius access token from DB settings or env."""
    from acestep.api.lireek.lireek_db import get_setting
    token = get_setting("genius_access_token") or os.environ.get("GENIUS_ACCESS_TOKEN", "")
    if not token:
        raise ValueError(
            "GENIUS_ACCESS_TOKEN is not set. "
            "Please configure it in LLM Settings or your .env file."
        )
    return token


def _get_auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_get_genius_token()}"}


def _clean_lyrics(raw: str) -> str:
    """Strip Genius-specific artefacts from raw lyrics text."""
    if not raw:
        return ""
    raw = re.sub(
        r"^\d+\s*Contributors?.*?Lyrics.*?(?=\[)",
        "", raw, count=1, flags=re.IGNORECASE | re.DOTALL,
    )
    raw = re.sub(
        r"^\d+\s*Contributors?.*?Lyrics\s*\n?",
        "", raw, count=1, flags=re.IGNORECASE | re.DOTALL,
    )
    raw = re.sub(r"You might also like\s*", "", raw)
    raw = re.sub(r"\d*Embed$", "", raw.strip())
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _scrape_lyrics(song_url: str) -> Optional[str]:
    """Scrape lyrics from a Genius song page using httpx + BeautifulSoup."""
    resp = httpx.get(
        song_url,
        headers=_BROWSER_HEADERS,
        follow_redirects=True,
        timeout=15,
    )
    resp.raise_for_status()
    html = BeautifulSoup(resp.text.replace("<br/>", "\n"), "html.parser")

    containers = html.find_all("div", attrs={"data-lyrics-container": "true"})
    if containers:
        return "\n".join(c.get_text() for c in containers)

    div = html.find("div", class_=re.compile(r"^lyrics$|Lyrics__Root"))
    if div is None:
        logger.warning("Could not find lyrics div on page: %s", song_url)
        return None
    return div.get_text()


# ── Authenticated Genius API calls ───────────────────────────────────────────

def _api_search(query: str, per_page: int = 20) -> list[dict]:
    headers = _get_auth_headers()
    resp = httpx.get(
        f"{API_ROOT}/search",
        headers=headers,
        params={"q": query, "per_page": per_page},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["response"]["hits"]


def _api_get_artist_songs(artist_id: int, per_page: int = 20, sort: str = "popularity") -> list[dict]:
    headers = _get_auth_headers()
    resp = httpx.get(
        f"{API_ROOT}/artists/{artist_id}/songs",
        headers=headers,
        params={"per_page": per_page, "page": 1, "sort": sort},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["response"]["songs"]


def _api_get_album_tracks(album_id: int) -> list[dict]:
    headers = _get_auth_headers()
    tracks = []
    page = 1
    while page:
        resp = httpx.get(
            f"{API_ROOT}/albums/{album_id}/tracks",
            headers=headers,
            params={"per_page": 50, "page": page},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()["response"]
        tracks.extend(data["tracks"])
        page = data.get("next_page")
    return tracks


def _get_artist_id_from_url(url: str) -> Optional[int]:
    """Scrape the artist ID directly from a Genius artist page URL."""
    try:
        resp = httpx.get(url, headers=_BROWSER_HEADERS, follow_redirects=True, timeout=15)
        resp.raise_for_status()
        match = re.search(r'\\?"artist_id\\?":\s*(\d+)', resp.text)
        if match:
            return int(match.group(1))
        match2 = re.search(r'content="genius://artists/(\d+)"', resp.text)
        if match2:
            return int(match2.group(1))
    except Exception as e:
        logger.warning("Failed to extract artist ID from URL: %s", e)
    return None


def _api_get_artist_details(artist_id: int) -> dict:
    headers = _get_auth_headers()
    resp = httpx.get(f"{API_ROOT}/artists/{artist_id}", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()["response"]["artist"]


def _find_artist_id(artist_name: str) -> Optional[int]:
    hits = _api_search(artist_name, per_page=5)
    name_lower = artist_name.lower()
    for hit in hits:
        result = hit.get("result", {})
        primary = result.get("primary_artist", {})
        if primary.get("name", "").lower() == name_lower:
            return primary["id"]
    if hits:
        return hits[0]["result"]["primary_artist"]["id"]
    return None


_BONUS_PATTERNS = re.compile(
    r"\([^)]*(?:Demo|Live|Outtake|Cassette|Remix|Acoustic|Remaster|Session)[^)]*\)",
    re.IGNORECASE,
)


def _is_bonus_track(title: str) -> bool:
    return bool(_BONUS_PATTERNS.search(title))


def _slugify_for_genius(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


def _find_album_id_by_page(album_name: str, artist_name: str) -> Optional[int]:
    """Fallback: find an album by scraping its Genius page URL."""
    artist_slug = _slugify_for_genius(artist_name)
    album_slug = _slugify_for_genius(album_name)
    url = f"https://genius.com/albums/{artist_slug}/{album_slug}"
    logger.info("Trying direct album page: %s", url)
    artist_lower = artist_name.lower()

    try:
        resp = httpx.get(url, headers=_BROWSER_HEADERS, follow_redirects=True, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        track_links = soup.select("div.chart_row a.u-display_block")
        if not track_links:
            return None

        for link in track_links[:3]:
            h3 = link.find("h3")
            title_text = h3.get_text(strip=True) if h3 else link.get_text(strip=True)
            title_text = re.sub(r"\s*Lyrics$", "", title_text).strip()
            if not title_text:
                continue
            try:
                hits = _api_search(f"{title_text} {artist_name}", per_page=5)
                for hit in hits:
                    result = hit.get("result", {})
                    song_id = result.get("id")
                    hit_artist = result.get("primary_artist", {}).get("name", "").lower()
                    if not song_id or hit_artist != artist_lower:
                        continue
                    headers = _get_auth_headers()
                    sresp = httpx.get(f"{API_ROOT}/songs/{song_id}", headers=headers, timeout=15)
                    sresp.raise_for_status()
                    song = sresp.json()["response"]["song"]
                    album = song.get("album")
                    if album and album_name.lower() in album.get("name", "").lower():
                        logger.info("Found album via page fallback: '%s' (ID: %d)", album["name"], album["id"])
                        return album["id"]
            except Exception as e:
                logger.warning("Failed to look up song '%s': %s", title_text[:30], e)
    except Exception as e:
        logger.warning("Failed to fetch album page: %s", e)
    return None


def _find_album_id(album_name: str, artist_name: str) -> Optional[int]:
    """Search for an album and return its Genius ID."""
    headers = _get_auth_headers()
    query = f"{album_name} {artist_name}"
    hits = _api_search(query, per_page=10)

    album_lower = album_name.lower()
    artist_lower = artist_name.lower()

    sorted_hits = sorted(
        hits,
        key=lambda h: h.get("result", {}).get("primary_artist", {}).get("name", "").lower() == artist_lower,
        reverse=True,
    )

    candidates: list[tuple[int, str]] = []
    seen_ids: set[int] = set()

    for hit in sorted_hits:
        song_id = hit.get("result", {}).get("id")
        if not song_id:
            continue
        try:
            resp = httpx.get(f"{API_ROOT}/songs/{song_id}", headers=headers, timeout=15)
            resp.raise_for_status()
            song = resp.json()["response"]["song"]
            album = song.get("album")
            if album and album_lower in album.get("name", "").lower():
                aid = album["id"]
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    candidates.append((aid, album.get("name", "")))
        except Exception as e:
            logger.warning("Failed to fetch song %s details: %s", song_id, e)
        if len(candidates) >= 3:
            break

    if candidates:
        for aid, aname in candidates:
            if aname.lower() == album_lower:
                return aid
        candidates.sort(key=lambda c: len(c[1]))
        return candidates[0][0]

    logger.info("Search-based album lookup failed, trying page URL fallback...")
    return _find_album_id_by_page(album_name, artist_name)


def _scrape_album_page_tracks(
    album_name: str = "",
    artist_name: str = "",
    url: Optional[str] = None,
) -> list[dict]:
    """Directly scrape track titles and song URLs from a Genius album page."""
    if not url:
        artist_slug = _slugify_for_genius(artist_name)
        album_slug = _slugify_for_genius(album_name)
        url = f"https://genius.com/albums/{artist_slug}/{album_slug}"

    try:
        resp = httpx.get(url, headers=_BROWSER_HEADERS, follow_redirects=True, timeout=15)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        track_links = soup.select("div.chart_row a.u-display_block")
        if not track_links:
            return []

        tracks = []
        for link in track_links:
            href = link.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = f"https://genius.com{href}"
            h3 = link.find("h3")
            title = h3.get_text(strip=True) if h3 else link.get_text(strip=True)
            title = re.sub(r"\s*Lyrics$", "", title).strip()
            if title:
                tracks.append({"title": title, "url": href})
        return tracks
    except Exception as e:
        logger.warning("Failed to scrape album page: %s", e)
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_lyrics(
    artist_name: str,
    album_name: Optional[str] = None,
    max_songs: int = 10,
) -> LyricsSearchResponse:
    """Fetch lyrics for an artist, optionally scoped to a specific album."""
    scrape_delay = 0.3
    songs: list[SongLyrics] = []
    artist_id = None

    # ── Detect artist URL ──────────────────────────────────────────────────
    if "genius.com/artists/" in artist_name:
        url_match = re.search(r'(https?://(?:www\.)?genius\.com/artists/[^\s]+)', artist_name)
        url = url_match.group(1) if url_match else artist_name.strip()
        artist_id = _get_artist_id_from_url(url)
        if artist_id:
            try:
                artist_details = _api_get_artist_details(artist_id)
                actual_name = artist_details.get("name")
                if actual_name:
                    logger.info("Resolved URL '%s' to artist '%s' (ID: %d)", url, actual_name, artist_id)
                    artist_name = actual_name
            except Exception as e:
                logger.warning("Could not fetch artist details for ID %d: %s", artist_id, e)

    # ── Detect album URL ──────────────────────────────────────────────────
    album_url: Optional[str] = None
    if album_name and "genius.com/albums/" in album_name:
        url_match = re.search(r'(https?://(?:www\.)?genius\.com/albums/[^\s]+)', album_name)
        album_url = url_match.group(1) if url_match else album_name.strip()
        parts = album_url.rstrip("/").split("/")
        if len(parts) >= 2:
            album_slug = parts[-1]
            artist_slug = parts[-2]
            album_name = album_slug.replace("-", " ").title()
            if not artist_id and artist_name == album_name:
                artist_name = artist_slug.replace("-", " ").title()

    if album_name:
        logger.info("Fetching album '%s' by '%s'", album_name, artist_name)

        if album_url:
            album_id_resolved = None
        else:
            album_id_resolved = _find_album_id(album_name, artist_name)

        if album_url is None and album_id_resolved is not None:
            tracks = _api_get_album_tracks(album_id_resolved)
            seen_titles: set[str] = set()

            for track in tracks:
                song_info = track.get("song", {})
                url = song_info.get("url")
                title = song_info.get("title", "Unknown")
                if not url:
                    continue
                if song_info.get("lyrics_state") != "complete":
                    continue
                if song_info.get("instrumental"):
                    continue
                if _is_bonus_track(title):
                    continue
                base_title = re.sub(r"\s*\(.*\)", "", title).strip().lower()
                if base_title in seen_titles:
                    continue
                seen_titles.add(base_title)

                try:
                    time.sleep(scrape_delay)
                    raw_lyrics = _scrape_lyrics(url)
                    if raw_lyrics:
                        songs.append(SongLyrics(title=title, album=album_name, lyrics=_clean_lyrics(raw_lyrics)))
                except Exception as e:
                    logger.warning("Failed to scrape lyrics for '%s': %s", title, e)
        else:
            page_tracks = _scrape_album_page_tracks(
                album_name=album_name,
                artist_name=artist_name,
                url=album_url,
            )
            for pt in page_tracks:
                title = pt["title"]
                url = pt["url"]
                if _is_bonus_track(title):
                    continue
                try:
                    time.sleep(scrape_delay)
                    raw_lyrics = _scrape_lyrics(url)
                    if raw_lyrics:
                        songs.append(SongLyrics(title=title, album=album_name, lyrics=_clean_lyrics(raw_lyrics)))
                except Exception as e:
                    logger.warning("Failed to scrape lyrics for '%s': %s", title, e)
    else:
        logger.info("Fetching up to %d songs by '%s'", max_songs, artist_name)
        if artist_id is None:
            artist_id = _find_artist_id(artist_name)
        if artist_id is None:
            raise ValueError(f"Could not find artist '{artist_name}' on Genius.")

        api_songs = _api_get_artist_songs(artist_id, per_page=max_songs)
        for song_info in api_songs[:max_songs]:
            url = song_info.get("url")
            title = song_info.get("title", "Unknown")
            song_album = song_info.get("album")
            album_title = song_album.get("name") if song_album else None
            if not url:
                continue
            try:
                time.sleep(scrape_delay)
                raw_lyrics = _scrape_lyrics(url)
                if raw_lyrics:
                    songs.append(SongLyrics(title=title, album=album_title, lyrics=_clean_lyrics(raw_lyrics)))
            except Exception as e:
                logger.warning("Failed to scrape lyrics for '%s': %s", title, e)

    if not songs:
        raise ValueError(
            f"No lyrics found for '{artist_name}'"
            + (f" – album '{album_name}'" if album_name else "")
            + ". Please check the spelling and try again."
        )

    return LyricsSearchResponse(
        artist=artist_name,
        album=album_name,
        songs=songs,
        total_songs=len(songs),
    )
